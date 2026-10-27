"""nlq-validator as a pre-LLM gate for off-topic and injection queries.

The validator is trained on validator_data/train_in_domain.txt and its threshold
is calibrated on the dev_* files. None of these overlap with the benchmark
queries, so validator accuracy on queries/*.jsonl is a held-out result.

Gate specs accepted by run.py --validator:
    nlq-tfidf, nlq-embed                 calibrated threshold (default use)
    nlq-tfidf-uncal, nlq-embed-uncal     nlq-validator's default threshold (0.25)
    module:function                      any callable(query) -> bool

Standalone evaluation (no LLM calls, costs nothing):
    python -m live_events_llm_benchmarks.validator --kind tfidf --queries queries/seed.jsonl
    python -m live_events_llm_benchmarks.validator --kind embed --no-calibrate --table
"""
import argparse
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Callable

from nlq_validator import NLQValidator

from live_events_llm_benchmarks.oracle import load_queries

DATA_DIR = Path(__file__).parent / "validator_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are an assistant for a live event booking platform. You help users find concerts, music festivals, "
    "sports games, standup comedy, theater, musicals, opera, family and kids shows, and movies, filtered by "
    "date, city, price, genre, and ticket availability. You also help with add-ons such as parking, VIP "
    "upgrades, food and beverage packages, merchandise, and ticket insurance."
)

Gate = Callable[[str], tuple[bool, float | None]]


def _lines(name: str) -> list[str]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build(kind: str = "tfidf", calibrate: bool = True) -> NLQValidator:
    if kind not in ("tfidf", "embed"):
        raise ValueError(f"kind must be 'tfidf' or 'embed', got {kind!r}")
    v = NLQValidator.from_training_file(
        DATA_DIR / "train_in_domain.txt",
        SYSTEM_PROMPT,
        embedding_model=EMBEDDING_MODEL if kind == "embed" else None,
    )
    if calibrate:
        v.apply_calibration(v.calibrate(_lines("dev_in_domain.txt"), _lines("dev_off_domain.txt")))
    return v


def make_gate(spec: str) -> Gate:
    """Turn a --validator spec into a callable returning (accepted, score)."""
    if spec.startswith("nlq-"):
        kind = spec[len("nlq-"):]
        calibrate = not kind.endswith("-uncal")
        v = build(kind.removesuffix("-uncal"), calibrate)

        def gate(query: str) -> tuple[bool, float | None]:
            return v.validate(query).is_valid, v.score(query)
        return gate

    module, func = spec.split(":")
    fn = getattr(importlib.import_module(module), func)

    def custom_gate(query: str) -> tuple[bool, float | None]:
        return bool(fn(query)), None
    return custom_gate


def evaluate(v: NLQValidator, items: list[dict]) -> dict:
    """Accept/reject accuracy on benchmark items: expect=reject is the negative class."""
    tp = fp = fn = tn = 0
    by_slice = defaultdict(lambda: [0, 0])  # slice -> [accepted, total]
    for item in items:
        accepted = v.validate(item["query"]).is_valid
        in_domain = item["expect"] != "reject"
        tp += accepted and in_domain
        fp += accepted and not in_domain
        fn += (not accepted) and in_domain
        tn += (not accepted) and not in_domain
        by_slice[item["slice"]][0] += accepted
        by_slice[item["slice"]][1] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": v.threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "off_topic_rejected": tn / (tn + fp) if tn + fp else 0.0,
        "in_domain_wrongly_rejected": fn,
        "accept_rate_by_slice": {s: a / n for s, (a, n) in sorted(by_slice.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["tfidf", "embed"], default="tfidf")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--queries", default="queries/seed.jsonl")
    parser.add_argument("--table", action="store_true", help="print the dev-set calibration sweep")
    args = parser.parse_args()

    v = build(args.kind, calibrate=not args.no_calibrate)
    if args.table:
        v.calibrate(_lines("dev_in_domain.txt"), _lines("dev_off_domain.txt")).summary()
    r = evaluate(v, load_queries(args.queries))
    print(f"\nkind={args.kind} calibrated={not args.no_calibrate} threshold={r['threshold']:.4f}")
    print(f"accept precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1']:.3f}")
    print(f"off-topic/injection rejected: {100 * r['off_topic_rejected']:.1f}%  "
          f"in-domain wrongly rejected: {r['in_domain_wrongly_rejected']}")
    for s, rate in r["accept_rate_by_slice"].items():
        print(f"  {s:<12} accepted {100 * rate:.0f}%")
