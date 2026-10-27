"""Gold answers from constraint labels.

A benchmark item carries machine-readable constraints; the gold answer is every
catalog occurrence (event_id, date, city) that satisfies all of them. Nobody
hand-lists gold events, so labels stay consistent if the catalog is regenerated.

Constraint keys (all optional):
    type            event type, e.g. "Concert"
    genre           one genre the event must carry
    artist          performer name (case-insensitive exact match)
    city / region / country
    date_from, date_to   inclusive, YYYY-MM-DD (resolve relative dates against
                         the item's anchor_date when labeling). Without date_from,
                         scoring uses anchor_date: past events never count as gold.
    max_price       cheapest ticket (price_min) must be <= this
    available_only  true -> exclude sold_out occurrences. Without it, sold-out
                    occurrences are optional when scoring: returning or omitting
                    them is not penalized (see gold_split).
    venue           venue name (case-insensitive exact match)

Usage:
    python -m live_events_llm_benchmarks.oracle check queries/seed.jsonl   # gold sizes, label errors
    python -m live_events_llm_benchmarks.oracle propose --n 20 --k 3       # constraint sets to write queries for
"""
import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from live_events_llm_benchmarks import catalog as catalog_mod

CONSTRAINT_KEYS = {"type", "genre", "artist", "city", "region", "country", "date_from", "date_to",
                   "max_price", "available_only", "venue"}
EXPECT_VALUES = {"results", "empty", "reject"}


def _eq(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def event_violations(event: dict, c: dict) -> list[str]:
    """Event-level constraints this event fails."""
    v = []
    if "type" in c and not _eq(event["type"], c["type"]):
        v.append("type")
    if "genre" in c and not any(_eq(g, c["genre"]) for g in event["genre"]):
        v.append("genre")
    if "artist" in c and not _eq(event["artist"], c["artist"]):
        v.append("artist")
    return v


def occurrence_violations(occ: dict, c: dict) -> list[str]:
    """Occurrence-level constraints this date/venue fails."""
    v = []
    for key in ("city", "region", "country", "venue"):
        if key in c and not _eq(occ.get(key, ""), c[key]):
            v.append(key)
    if "date_from" in c and occ["date"] < c["date_from"]:
        v.append("date")
    elif "date_to" in c and occ["date"] > c["date_to"]:
        v.append("date")
    if "max_price" in c and occ["price_min"] > c["max_price"]:
        v.append("price")
    if c.get("available_only") and occ["status"] == "sold_out":
        v.append("availability")
    return v


def violations(event: dict, occ: dict, c: dict) -> list[str]:
    return event_violations(event, c) + occurrence_violations(occ, c)


def occ_key(event_id: str, occ: dict) -> tuple[str, str, str]:
    return (event_id, occ["date"], occ["city"].split(",")[0].strip().lower())


def item_constraints(item: dict) -> dict:
    """Constraints to score an item by: the labels, with date_from defaulting to anchor_date.

    The model is told today's date, so occurrences before it are not a valid answer
    even when the query states no date.
    """
    c = dict(item.get("constraints", {}))
    if "date_from" not in c and item.get("anchor_date"):
        c["date_from"] = item["anchor_date"]
    return c


def gold(events: list[dict], constraints: dict) -> set[tuple[str, str, str]]:
    out = set()
    for e in events:
        if event_violations(e, constraints):
            continue
        for occ in e["details"]:
            if not occurrence_violations(occ, constraints):
                out.add(occ_key(e["event_id"], occ))
    return out


def gold_split(events: list[dict], constraints: dict) -> tuple[set, set]:
    """(required, optional) gold occurrences. Sold-out ones are optional unless available_only is set."""
    full = gold(events, constraints)
    if constraints.get("available_only"):
        return full, set()
    required = gold(events, {**constraints, "available_only": True})
    return required, full - required


def validate_item(item: dict) -> list[str]:
    """Schema problems with a benchmark item."""
    errs = []
    for key in ("id", "query", "slice", "expect"):
        if key not in item:
            errs.append(f"missing '{key}'")
    if item.get("expect") not in EXPECT_VALUES:
        errs.append(f"expect must be one of {sorted(EXPECT_VALUES)}")
    unknown = set(item.get("constraints", {})) - CONSTRAINT_KEYS
    if unknown:
        errs.append(f"unknown constraint keys {sorted(unknown)}")
    if item.get("expect") in ("results", "empty") and not item.get("constraints"):
        errs.append("results/empty items need constraints")
    return errs


def load_queries(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(path: str, events: list[dict], max_gold: int) -> int:
    """Validate a query file against the catalog. Returns number of problems."""
    items = load_queries(path)
    problems = 0
    seen = set()
    for item in items:
        errs = validate_item(item)
        if item.get("id") in seen:
            errs.append("duplicate id")
        seen.add(item.get("id"))
        if not errs and item["expect"] != "reject":
            required, optional = gold_split(events, item_constraints(item))
            n = len(required) + len(optional)
            if item["expect"] == "results" and not required:
                errs.append("expect=results but gold is empty" if n == 0 else
                            "expect=results but every gold occurrence is sold out (optional)")
            if item["expect"] == "empty" and n > 0:
                errs.append(f"expect=empty but gold has {n}")
            if n > max_gold:
                errs.append(f"gold has {n} occurrences (> {max_gold}); tighten constraints")
            status = f"gold={n}" + (f" ({len(optional)} sold out, optional)" if optional else "")
        else:
            status = "reject" if not errs else ""
        if errs:
            problems += 1
            print(f"FAIL {item.get('id')}: {'; '.join(errs)}")
        else:
            print(f"ok   {item['id']:<12} {item['slice']:<16} {status}")
    print(f"\n{len(items)} items, {problems} with problems")
    return problems


def propose(events: list[dict], n: int, k: int, seed: int, max_gold: int) -> None:
    """Sample satisfiable constraint sets with k constraints and 1..max_gold gold occurrences.

    Write the natural-language query yourself; do not generate it with a model you evaluate.
    """
    rng = random.Random(seed)
    found = 0
    tries = 0
    while found < n and tries < n * 500:
        tries += 1
        e = rng.choice(events)
        occ = rng.choice(e["details"])
        d = date.fromisoformat(occ["date"])
        pool = {
            "type": e["type"],
            "genre": rng.choice(e["genre"]),
            "artist": e["artist"],
            "city": occ["city"],
            "country": occ["country"],
            "date_from": (d - timedelta(days=rng.choice([0, 0, 1, 3, 7]))).isoformat(),
            "date_to": (d + timedelta(days=rng.choice([0, 0, 1, 3, 7]))).isoformat(),
            "max_price": float(int(occ["price_min"] // 10 * 10 + 10)),
            "available_only": True,
        }
        keys = rng.sample(sorted(pool), k)
        # date_from/date_to only make sense together; widen to a window
        if "date_from" in keys or "date_to" in keys:
            keys = [x for x in keys if x not in ("date_from", "date_to")] + ["date_from", "date_to"]
        c = {key: pool[key] for key in keys}
        if c.get("available_only") and occ["status"] == "sold_out":
            continue
        g = gold(events, c)
        if 1 <= len(g) <= max_gold:
            found += 1
            print(json.dumps({"constraints": c, "gold_size": len(g)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--max-gold", type=int, default=10)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_check = sub.add_parser("check")
    p_check.add_argument("queries")
    p_prop = sub.add_parser("propose")
    p_prop.add_argument("--n", type=int, default=20)
    p_prop.add_argument("--k", type=int, default=3, help="number of constraints")
    p_prop.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    events = catalog_mod.load(args.catalog)
    if args.cmd == "check":
        sys.exit(1 if check(args.queries, events, args.max_gold) else 0)
    propose(events, args.n, args.k, args.seed, args.max_gold)
