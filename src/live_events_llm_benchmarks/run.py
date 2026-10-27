"""Run a system configuration over a query file and write predictions as JSONL.

Systems (the ablation ladder):
    single   one generic role, no classifier (plain tool-calling LLM baseline)
    single-union  one role holding every routed role's instructions, no classifier (content control)
    routed   lite-model role classifier + specialized roles (FigureOut default)
    single-union-neutral, routed-neutral
             type-neutral ablation: as single-union / routed, but discovery prompts do not name an event_type

Orthogonal switches: --validator (pre-LLM gate), --max-roles, --cache, --lite-llm-version.

Examples:
    python -m live_events_llm_benchmarks.run --system single --llm openai --llm-version gpt-4.1-mini
    python -m live_events_llm_benchmarks.run --system routed --llm claude --llm-version claude-sonnet-5-5 \
        --lite-llm-version claude-haiku-5-5 --max-roles 2 --repeats 3
    python -m live_events_llm_benchmarks.run --system routed --validator nlq-tfidf ...

--validator takes nlq-tfidf, nlq-embed (calibrated nlq-validator), their -uncal
variants, or any importable module:function returning True to let a query through.
Rejected queries never reach the LLM and cost no tokens.
"""
import argparse
import asyncio
import contextvars
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from figureout import FigureOut

from live_events_llm_benchmarks.mcp_server import bench_mcp
from live_events_llm_benchmarks.oracle import load_queries
from live_events_llm_benchmarks.roles import ROUTED_SYSTEMS, SYSTEMS


# Tool calls made while running the current item. asyncio tasks copy the context,
# so calls from figureout's parallel role and tool tasks land in the right list.
_tool_calls: contextvars.ContextVar[list | None] = contextvars.ContextVar("tool_calls", default=None)


def log_tool_calls(fo: FigureOut) -> None:
    """Record each tool call's name, arguments and match count. figureout doesn't report arguments."""
    execute = fo._execute_tool

    async def logged(tool_name, tool_args):
        call = {"name": tool_name, "arguments": tool_args}
        try:
            out = await execute(tool_name, tool_args)
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            calls = _tool_calls.get()
            if calls is not None:
                calls.append(call)
        try:
            parsed = json.loads(out)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            call.update({k: parsed[k] for k in ("total", "returned") if k in parsed})
        return out

    fo._execute_tool = logged


def build_engine(args) -> FigureOut:
    fo = FigureOut(
        llm=args.llm,
        llm_version=args.llm_version,
        lite_llm_version=args.lite_llm_version,
        mcp_server=bench_mcp,
        roles=SYSTEMS[args.system],
        max_roles=args.max_roles,
        cache_enabled=args.cache,
        verbose=True,
        inject_date=False,  # each item carries its own anchor date via context
        max_tool_rounds=args.max_tool_rounds,
        reasoning_effort=args.reasoning_effort,
    )
    log_tool_calls(fo)
    return fo


def load_validator(spec: str | None):
    if not spec:
        return None
    from live_events_llm_benchmarks.validator import make_gate  # nlq-validator is only needed when gating
    return make_gate(spec)


async def run_item(fo, args, validator, item, repeat, sem) -> dict:
    async with sem:
        record = {"id": item["id"], "repeat": repeat}
        calls = []
        _tool_calls.set(calls)
        start = time.perf_counter()
        try:
            accepted, record["validator_score"] = validator(item["query"]) if validator else (True, None)
            if not accepted:
                record.update(response={"events": []}, rejected_by="validator", roles_selected=[],
                              input_tokens=0, output_tokens=0, classifier_input_tokens=0,
                              classifier_output_tokens=0, tools_used=[])
            else:
                anchor = item.get("anchor_date", args.anchor_date)
                out = await fo.run(
                    item["query"],
                    role=None if args.system in ROUTED_SYSTEMS else "assistant",
                    context=f"Today's date is {anchor}.",
                )
                debug = out["debug"]
                if args.system in ROUTED_SYSTEMS and "classifier_input_tokens" not in debug:
                    warnings.warn("figureout does not report classifier tokens; token totals undercount. "
                                  "Upgrade figureout.", stacklevel=1)
                cached = debug.get("cached", False)
                record.update(
                    response=out["response"],
                    rejected_by="router" if args.system in ROUTED_SYSTEMS and not debug["roles_selected"] else None,
                    # routing is only scored when a classifier chose the roles
                    roles_selected=debug.get("roles_selected") if args.system in ROUTED_SYSTEMS else None,
                    # input/output totals include the classifier; a cache hit spends no tokens
                    input_tokens=0 if cached else debug["input_tokens"],
                    output_tokens=0 if cached else debug["output_tokens"],
                    classifier_input_tokens=0 if cached else debug.get("classifier_input_tokens", 0),
                    classifier_output_tokens=0 if cached else debug.get("classifier_output_tokens", 0),
                    tools_used=debug.get("tools_used", []),
                    cached=cached,
                )
                if isinstance(out["response"], dict) and "error" in out["response"]:
                    # figureout reports role failures (rate limits, unparseable output) in the response
                    record["error"] = f"RoleFailure: {out['response']['error']}"
        except Exception as exc:  # keep going; failures are scored as misses
            record.update(error=f"{type(exc).__name__}: {exc}", response=None)
        record["tool_calls"] = calls
        record["latency_s"] = round(time.perf_counter() - start, 3)
        return record


async def main(args):
    load_dotenv()  # .env in the working directory
    items = load_queries(args.queries)
    if args.limit:
        items = items[: args.limit]
    fo = build_engine(args)
    validator = load_validator(args.validator)
    sem = asyncio.Semaphore(args.concurrency)

    out_path = Path(args.out) if args.out else (
        Path("runs") / f"{args.system}_{args.llm}_{args.llm_version or 'default'}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items()}
    config["started_at"] = datetime.now(timezone.utc).isoformat()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"config": config}) + "\n")
        for repeat in range(args.repeats):
            tasks = [run_item(fo, args, validator, item, repeat, sem) for item in items]
            for coro in asyncio.as_completed(tasks):
                record = await coro
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                status = "ERR" if record.get("error") else ("gated" if record.get("rejected_by") == "validator" else "ok")
                print(f"[{repeat}] {record['id']:<14} {status:<5} {record['latency_s']:.2f}s")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=list(SYSTEMS), required=True)
    parser.add_argument("--llm", required=True, help="figureout LLM name: openai, claude, gemini, meta, mistral, groq")
    parser.add_argument("--llm-version", default=None)
    parser.add_argument("--lite-llm-version", default=None)
    parser.add_argument("--max-roles", type=int, default=1)
    parser.add_argument("--max-tool-rounds", type=int, default=3)
    parser.add_argument("--reasoning-effort", default=None,
                        help="OpenAI only, e.g. none (gpt-6-luna rejects tools with reasoning on chat completions)")
    parser.add_argument("--cache", action="store_true", help="enable FigureOut's response cache (off by default)")
    parser.add_argument("--validator", default=None,
                        help="pre-LLM gate: nlq-tfidf, nlq-embed, nlq-tfidf-uncal, nlq-embed-uncal, or module:function")
    parser.add_argument("--queries", default="queries/seed.jsonl")
    parser.add_argument("--anchor-date", default="2027-01-13", help="fallback when an item has no anchor_date")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=None)
    asyncio.run(main(parser.parse_args()))
