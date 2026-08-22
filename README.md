# Vera-clone — magicpin AI Challenge submission

## Approach
Single LLM composer (Claude, temperature=0) fed the full category + merchant + trigger
(+ optional customer) context per the 4-context framework. A thin deterministic layer sits
in front of `/v1/reply` — regex classifiers for auto-reply, hostility, "not interested",
and explicit intent — so those routing decisions don't depend on the LLM guessing right
under time pressure; the LLM only composes the actual message text/action once routed.

## Tradeoffs
- In-memory state (no Redis/DB) — fine for a single 60-min test window, not production-durable.
- Fallback composer (rule-based, no LLM) kicks in only if the Claude call errors/times out,
  so `/v1/tick` and `/v1/reply` always answer inside the 30s budget even on API hiccups.
- Anti-repetition is best-effort: the LLM is shown its own prior messages in the conversation
  and told not to repeat them; there's no hard dedup beyond that.

## What would have helped most
A real canonical 30-pair test set (referenced in the brief but not included in this dataset
drop) would have let me validate against the actual scoring fixtures instead of arbitrary
(merchant, trigger) pairs from the seed data.

## Deploying
Default provider is Groq (free tier — get a key at console.groq.com).
```
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
uvicorn bot:app --host 0.0.0.0 --port $PORT
```
To use Anthropic or OpenAI instead, set `LLM_PROVIDER=anthropic` (+ `ANTHROPIC_API_KEY`)
or `LLM_PROVIDER=openai` (+ `OPENAI_API_KEY`).
