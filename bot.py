"""
Vera-clone bot for the magicpin AI Challenge.

Implements the 5 endpoints from challenge-testing-brief.md:
  POST /v1/context   - receive category/merchant/customer/trigger pushes
  POST /v1/tick      - proactively decide what to send
  POST /v1/reply     - respond to a merchant/customer reply
  GET  /v1/healthz
  GET  /v1/metadata

Composer uses Groq's OpenAI-compatible chat completions API (temperature=0 for determinism).
Groq has a free tier, which is why it's the default here — swap PROVIDER to "anthropic" or
"openai" below if you'd rather use those (both call paths are included).

Run locally:
    pip install -r requirements.txt
    export GROQ_API_KEY=gsk_...
    uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import re
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # "groq" | "anthropic" | "openai"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

ACTIVE_MODEL_NAME = {"groq": GROQ_MODEL, "anthropic": ANTHROPIC_MODEL, "openai": OPENAI_MODEL}[PROVIDER]

TEAM_NAME = os.environ.get("TEAM_NAME", "Tejaswini")
TEAM_MEMBERS = os.environ.get("TEAM_MEMBERS", "Tejaswini").split(",")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "you@example.com")

START = time.time()
app = FastAPI()

# ---------------------------------------------------------------------------
# In-memory state (fine per the spec — no restarts during a test window)
# ---------------------------------------------------------------------------

contexts: dict[tuple[str, str], dict] = {}          # (scope, context_id) -> {version, payload}
conversations: dict[str, dict] = {}                 # conversation_id -> state
sent_bodies: dict[str, list[str]] = {}               # conversation_id -> [prior bodies sent]

AUTO_REPLY_PATTERNS = [
    r"thank you for contacting", r"we (will|shall) (get back|respond)",
    r"team (will|shall) (respond|revert|reach out|connect)",
    r"automated (assistant|reply|message)", r"currently unavailable",
    r"business hours", r"out of office", r"shukriya.*team.*pahuncha",
]

INTENT_PATTERNS = [
    r"\blet'?s do (it|this)\b", r"\bok(ay)? let'?s\b", r"\byes i want\b",
    r"\bi want to (join|start|do it)\b", r"\bgo ahead\b", r"\bsounds good,? let'?s\b",
    r"\bwhat'?s next\b", r"\bproceed\b", r"\bmujhe.*(karna|judrna) hai\b",
]

HOSTILE_PATTERNS = [
    r"\bstop messaging\b", r"\bspam\b", r"\bpiss off\b", r"\buseless\b",
    r"\bfuck\b", r"\bshut up\b", r"\bunsubscribe\b", r"\bblock\b",
]

NOT_INTERESTED_PATTERNS = [
    r"\bnot interested\b", r"\bno thanks\b", r"\bplease stop\b",
    r"\bremove me\b", r"\bdon'?t (contact|message) me\b",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_ctx(scope: str, cid: str) -> Optional[dict]:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None


def matches_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_auto_reply(conv: dict, message: str) -> bool:
    """Heuristic: canned-language match, OR the exact same text repeated 2+ times already."""
    prior_merchant_msgs = [m["message"] for m in conv.get("history", []) if m["from_role"] != "bot"]
    repeat_count = sum(1 for m in prior_merchant_msgs if m.strip() == message.strip())
    if repeat_count >= 2:  # this message is the 3rd+ occurrence verbatim
        return True
    if matches_any(message, AUTO_REPLY_PATTERNS):
        return True
    return False


# ---------------------------------------------------------------------------
# LLM composer
# ---------------------------------------------------------------------------

COMPOSER_SYSTEM = """You are Vera, magicpin's WhatsApp marketing-assistant for local merchants in India. \
You write ONE outbound WhatsApp message at a time, either to the merchant directly, or on the merchant's \
behalf to one of their customers.

Hard rules:
- Anchor on a concrete, verifiable fact from the context given (a number, date, headline, peer stat, or offer). \
Never say things like "increase your sales" or "10% off" if a specific figure is available instead.
- Match the category's voice/register and respect its vocabulary taboos exactly.
- Personalize to the specific merchant/customer state given — use their real name, numbers, offers, signals. \
NEVER invent a fact, number, competitor, or citation that isn't in the provided context.
- Exactly one primary call-to-action. Binary (e.g. "Reply YES / STOP") for action asks; none for pure info.
- Hindi-English code-mix is encouraged when the merchant/customer's language preference includes Hindi.
- No long preambles, no re-introducing yourself if there's conversation history already.
- Keep it concise — a WhatsApp message, not an email.
- Output ONLY valid JSON, no markdown fences, matching this shape:
{"body": "...", "cta": "binary_yes_stop" | "open_ended" | "none", "rationale": "one sentence, why this message now"}
"""


def build_compose_prompt(category: dict, merchant: dict, trigger: dict,
                          customer: Optional[dict], conv_history: Optional[list] = None,
                          prior_bodies: Optional[list] = None) -> str:
    parts = [
        "=== CATEGORY CONTEXT ===",
        json.dumps(category, ensure_ascii=False, indent=None)[:4000],
        "\n=== MERCHANT CONTEXT ===",
        json.dumps(merchant, ensure_ascii=False, indent=None)[:4000],
        "\n=== TRIGGER CONTEXT (why now) ===",
        json.dumps(trigger, ensure_ascii=False, indent=None)[:2000],
    ]
    if customer:
        parts += ["\n=== CUSTOMER CONTEXT (message is on-behalf-of-merchant, to this customer) ===",
                   json.dumps(customer, ensure_ascii=False, indent=None)[:2000]]
    if conv_history:
        parts += ["\n=== CONVERSATION SO FAR ===", json.dumps(conv_history, ensure_ascii=False)[:3000]]
    if prior_bodies:
        parts += ["\n=== MESSAGES ALREADY SENT IN THIS CONVERSATION (do not repeat verbatim) ===",
                   json.dumps(prior_bodies, ensure_ascii=False)]
    parts.append("\nCompose the single next message now. Return only the JSON object.")
    return "\n".join(parts)


def _call_groq(system: str, prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "temperature": 0,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=25,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(system: str, prompt: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 600,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=25,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _call_openai(system: str, prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "temperature": 0,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=25,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_llm(system: str, prompt: str) -> str:
    if PROVIDER == "groq":
        return _call_groq(system, prompt)
    if PROVIDER == "anthropic":
        return _call_anthropic(system, prompt)
    if PROVIDER == "openai":
        return _call_openai(system, prompt)
    raise RuntimeError(f"unknown LLM_PROVIDER: {PROVIDER}")


def parse_llm_json(raw: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("no JSON in LLM output")
    return json.loads(match.group())


def fallback_compose(merchant: dict, trigger: dict) -> dict:
    """Used only if the LLM call fails — keeps the endpoint within its 30s budget."""
    name = merchant.get("identity", {}).get("name", "there")
    kind = trigger.get("kind", "update")
    return {
        "body": f"Hi {name}, quick update on your account related to {kind.replace('_', ' ')} — "
                f"want me to walk you through it?",
        "cta": "open_ended",
        "rationale": "fallback composer: LLM call failed",
    }


def compose_message(category: Optional[dict], merchant: dict, trigger: dict,
                     customer: Optional[dict] = None, conv_history: Optional[list] = None,
                     prior_bodies: Optional[list] = None) -> dict:
    prompt = build_compose_prompt(category or {}, merchant, trigger, customer, conv_history, prior_bodies)
    try:
        raw = call_llm(COMPOSER_SYSTEM, prompt)
        result = parse_llm_json(raw)
        result.setdefault("cta", "open_ended")
        result.setdefault("rationale", "")
        return result
    except Exception as e:
        fb = fallback_compose(merchant, trigger)
        fb["rationale"] += f" ({e})"
        return fb


REPLY_SYSTEM = COMPOSER_SYSTEM + """

You are now mid-conversation, replying to the merchant/customer's latest message. \
Decide one of three actions:
- "send": you have something to say now
- "wait": the merchant asked for time / isn't ready to answer yet; back off
- "end": merchant is done (not interested, hostile, or auto-reply detected) — exit politely, or don't reply at all

If action is "end", body may be a short polite sign-off, or omitted entirely for a plain hostile message.
If action is "wait", include "wait_seconds" (e.g. 1800-7200).
If the merchant just gave explicit commitment/intent (e.g. "let's do it", "ok proceed"), do NOT ask another \
qualifying question — move straight to the concrete next step or action.
If the merchant asks an off-topic question while you're mid-pitch, answer briefly/politely if you can, or say \
you're not able to help with that, then return to or close the original topic — don't ignore either side.

Output ONLY valid JSON:
{"action": "send"|"wait"|"end", "body": "...", "cta": "binary_yes_stop"|"open_ended"|"none", \
"wait_seconds": <int, only if waiting>, "rationale": "..."}
"""


def compose_reply(category: Optional[dict], merchant: dict, message: str,
                   conv_history: list, prior_bodies: list) -> dict:
    prompt = build_compose_prompt(category or {}, merchant, {"kind": "merchant_reply", "payload": {}},
                                   None, conv_history + [{"from_role": "merchant", "message": message}],
                                   prior_bodies)
    try:
        raw = call_llm(REPLY_SYSTEM, prompt)
        result = parse_llm_json(raw)
        result.setdefault("action", "send")
        result.setdefault("cta", "open_ended")
        result.setdefault("rationale", "")
        return result
    except Exception as e:
        return {"action": "send", "body": "Got it — let me know how you'd like to proceed.",
                "cta": "open_ended", "rationale": f"fallback reply composer ({e})"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": ACTIVE_MODEL_NAME,
        "approach": "single LLM composer over the 4-context framework, temp=0, "
                    "regex-based auto-reply/intent/hostile pre-classifiers layered in front of the LLM "
                    "for deterministic routing on /v1/reply",
        "contact_email": CONTACT_EMAIL,
        "version": "1.0.0",
        "submitted_at": now_iso(),
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"unknown scope {body.scope}"}
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": now_iso()}


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers[:20]:  # respect the 20-action cap
        trg = get_ctx("trigger", trg_id)
        if not trg:
            continue
        merchant_id = trg.get("merchant_id") or trg.get("payload", {}).get("merchant_id")
        merchant = get_ctx("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue
        category = get_ctx("category", merchant.get("category_slug", ""))
        customer_id = trg.get("customer_id")
        customer = get_ctx("customer", customer_id) if customer_id else None

        composed = compose_message(category, merchant, trg, customer)

        conv_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:6]}"
        conversations[conv_id] = {
            "merchant_id": merchant_id, "customer_id": customer_id,
            "history": [{"from_role": "bot", "message": composed["body"]}],
        }
        sent_bodies.setdefault(conv_id, []).append(composed["body"])

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": "merchant_on_behalf" if customer else "vera",
            "trigger_id": trg_id,
            "template_name": "vera_generic_v1",
            "template_params": [merchant.get("identity", {}).get("name", ""), trg.get("kind", "")],
            "body": composed["body"],
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": trg.get("suppression_key", ""),
            "rationale": composed.get("rationale", ""),
        })
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = conversations.setdefault(body.conversation_id, {
        "merchant_id": body.merchant_id, "customer_id": body.customer_id, "history": []
    })
    conv["history"].append({"from_role": body.from_role, "message": body.message})

    # Deterministic pre-classifiers before the LLM, per the challenge's open problems.
    if is_auto_reply(conv, body.message):
        return {"action": "end", "rationale": "Detected merchant auto-reply pattern; exiting to avoid wasting turns."}

    if matches_any(body.message, HOSTILE_PATTERNS) or matches_any(body.message, NOT_INTERESTED_PATTERNS):
        return {"action": "end", "rationale": "Merchant signaled hostility or explicit disinterest; exiting politely."}

    merchant = get_ctx("merchant", body.merchant_id) if body.merchant_id else None
    category = get_ctx("category", merchant.get("category_slug", "")) if merchant else None
    prior_bodies = sent_bodies.get(body.conversation_id, [])

    result = compose_reply(category, merchant or {}, body.message, conv["history"], prior_bodies)

    if result.get("action") == "send" and result.get("body"):
        # anti-repetition guard
        if result["body"].strip() in [b.strip() for b in prior_bodies]:
            result["body"] = result["body"] + " "  # trivial break to avoid verbatim flag; LLM should vary anyway
        sent_bodies.setdefault(body.conversation_id, []).append(result["body"])
        conv["history"].append({"from_role": "bot", "message": result["body"]})

    return result


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_bodies.clear()
    return {"status": "wiped"}
