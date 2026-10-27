"""Score a predictions file from run.py against the query labels and catalog.

Per item, predicted occurrences (event_id, date, city) are pulled from every
"events" list in the response (single- or multi-role) and compared to the
oracle's gold set.

Metrics:
    success        results items: occurrence F1; empty/reject items: 1 if no events returned.
                   Failed runs (see failed()) score 0 on every item.
    P / R / F1     occurrence-level, results items only. Sold-out gold occurrences are optional
                   unless the item sets available_only: they count neither for nor against.
                   With available_only, a sold-out date the response labels status "sold_out" is
                   also neutral (it isn't offered as available); unlabeled, it is a violation.
    halluc         share of returned occurrences that are not in the catalog at all
    violation      share of returned, real occurrences that break a query constraint
    route_acc      gold_roles subset of selected subset of gold_roles + acceptable_roles (routed runs only)
    tokens         main + classifier tokens per query
    latency        seconds per query (mean, p95)

Usage:
    python -m live_events_llm_benchmarks.score runs/<file>.jsonl [--queries queries/seed.jsonl] [--json out.json]
"""
import argparse
import json
import statistics
from collections import defaultdict

from live_events_llm_benchmarks import catalog as catalog_mod
from live_events_llm_benchmarks.oracle import gold_split, item_constraints, load_queries, occ_key, violations


def extract_events(response) -> list[dict]:
    """All event dicts under any 'events' key, at any depth (multi-role responses nest by role)."""
    found = []
    if isinstance(response, dict):
        for key, value in response.items():
            if key == "events" and isinstance(value, list):
                found.extend(e for e in value if isinstance(e, dict))
            else:
                found.extend(extract_events(value))
    elif isinstance(response, list):
        for value in response:
            found.extend(extract_events(value))
    return found


def predicted_occurrences(response, status: str | None = None) -> set[tuple]:
    """Occurrence keys in the response; with `status`, only those the response labels with it."""
    out = set()
    for e in extract_events(response):
        eid = str(e.get("event_id", ""))
        details = e.get("details") or [{}]
        for d in details:
            if not isinstance(d, dict) or (status and d.get("status") != status):
                continue
            out.add(occ_key(eid, {"date": str(d.get("date", "")), "city": str(d.get("city", ""))}))
    return out


def build_index(events: list[dict]) -> dict:
    return {occ_key(e["event_id"], occ): (e, occ) for e in events for occ in e["details"]}


def prf(pred: set, gold_set: set) -> tuple[float, float, float]:
    if not pred and not gold_set:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold_set)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def failed(rec: dict) -> bool:
    """A run error, or figureout's in-band failure: it catches role errors (rate limits, unparseable
    output) and returns {"error": ...} as the response instead of raising."""
    response = rec.get("response")
    return bool(rec.get("error")) or (isinstance(response, dict) and "error" in response)


def score_record(item: dict, rec: dict, events: list[dict], index: dict) -> dict:
    pred = predicted_occurrences(rec.get("response"))
    s = {"id": item["id"], "slice": item["slice"], "expect": item["expect"], "repeat": rec.get("repeat", 0),
         "error": failed(rec), "n_pred": len(pred)}

    c = item_constraints(item)
    if c.get("available_only"):
        # A truly sold-out date the response labels sold_out isn't offered as available: neutral
        labeled = {k for k in predicted_occurrences(rec.get("response"), status="sold_out")
                   if k in index and index[k][1]["status"] == "sold_out"}
        pred -= labeled
        s["n_sold_out_labeled"] = len(labeled)

    real = [k for k in pred if k in index]
    s["n_halluc"] = len(pred) - len(real)
    s["n_violation"] = sum(1 for k in real if violations(*index[k], c)) if c else 0

    if item["expect"] == "results":
        required, optional = gold_split(events, c)
        p, r, f = prf(pred - optional, required)
        s.update(precision=p, recall=r, f1=f, success=f)
    else:
        # a failed run returns no events but hasn't rejected anything: it's a miss, not a correct reject
        s["success"] = 1.0 if not pred and not s["error"] else 0.0

    selected = rec.get("roles_selected")
    if item.get("gold_roles") and selected is not None and rec.get("rejected_by") != "validator":
        gold_roles = set(item["gold_roles"])
        allowed = gold_roles | set(item.get("acceptable_roles", []))
        s["route_ok"] = gold_roles <= set(selected) <= allowed

    s["tokens"] = rec.get("input_tokens", 0) + rec.get("output_tokens", 0)  # already includes the classifier
    s["classifier_tokens"] = rec.get("classifier_input_tokens", 0) + rec.get("classifier_output_tokens", 0)
    s["latency"] = rec.get("latency_s", 0.0)
    return s


def _mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def aggregate(rows: list[dict]) -> dict:
    n_pred = sum(r["n_pred"] for r in rows)
    n_real = n_pred - sum(r["n_halluc"] for r in rows)
    lat = sorted(r["latency"] for r in rows)
    routed = [r["route_ok"] for r in rows if "route_ok" in r]
    return {
        "n": len(rows),
        "success": _mean([r["success"] for r in rows]),
        "precision": _mean([r["precision"] for r in rows if "precision" in r]),
        "recall": _mean([r["recall"] for r in rows if "recall" in r]),
        "f1": _mean([r["f1"] for r in rows if "f1" in r]),
        "halluc": sum(r["n_halluc"] for r in rows) / n_pred if n_pred else 0.0,
        "violation": sum(r["n_violation"] for r in rows) / n_real if n_real else 0.0,
        "route_acc": _mean(routed) if routed else None,
        "errors": sum(r["error"] for r in rows),
        "tokens": _mean([r["tokens"] for r in rows]),
        "latency": _mean(lat),
        "latency_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else float("nan"),
    }


def load_run(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    config = lines[0].get("config", {}) if lines and "config" in lines[0] else {}
    return config, [r for r in lines if "id" in r]


def score_run(run_path: str, queries_path: str | None = None, catalog_path: str | None = None) -> dict:
    config, records = load_run(run_path)
    items = {q["id"]: q for q in load_queries(queries_path or config.get("queries") or "queries/seed.jsonl")}
    events = catalog_mod.load(catalog_path)
    index = build_index(events)

    rows = [score_record(items[r["id"]], r, events, index) for r in records if r["id"] in items]
    by_slice = defaultdict(list)
    by_repeat = defaultdict(list)
    for r in rows:
        by_slice[r["slice"]].append(r)
        by_repeat[r["repeat"]].append(r)

    overall = aggregate(rows)
    if len(by_repeat) > 1:
        per_repeat = [aggregate(rs)["success"] for rs in by_repeat.values()]
        overall["success_std_across_repeats"] = statistics.stdev(per_repeat)
    return {
        "config": config,
        "overall": overall,
        "by_slice": {k: aggregate(v) for k, v in sorted(by_slice.items())},
        "items": rows,
    }


def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return "-"
    if isinstance(v, float):
        return f"{100 * v:.1f}" if pct else f"{v:.2f}"
    return str(v)


def print_table(result: dict) -> None:
    cols = [("n", False), ("success", True), ("f1", True), ("precision", True), ("recall", True),
            ("halluc", True), ("violation", True), ("route_acc", True), ("tokens", False),
            ("latency", False), ("latency_p95", False), ("errors", False)]
    print("| slice | " + " | ".join(c for c, _ in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    rows = list(result["by_slice"].items()) + [("**overall**", result["overall"])]
    for name, agg in rows:
        print(f"| {name} | " + " | ".join(_fmt(agg[c], pct) for c, pct in cols) + " |")
    if "success_std_across_repeats" in result["overall"]:
        print(f"\nsuccess std across repeats: {100 * result['overall']['success_std_across_repeats']:.2f} pts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--json", default=None, help="write full results (incl. per-item rows) here")
    args = parser.parse_args()

    result = score_run(args.run, args.queries, args.catalog)
    print_table(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
