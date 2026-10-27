"""Join sampled candidates (scripts/sample_queries.py) with hand-written wording into benchmark items.

The wording file has one query per candidate, in candidate order. Blank lines and lines starting with '#' are
skipped. After the candidates come reject items written as `offtopic: <query>` or `injection: <query>`, with an
optional ` || <note>`. Ids continue from the highest id per slice in the --existing files.

Usage:
    python scripts/build_queries.py candidates.jsonl wording.txt --author claude-opus-5-5 \\
        --existing queries/seed.jsonl queries/bench.jsonl >> queries/bench.jsonl
"""
import argparse
import json
import re
import sys

from live_events_llm_benchmarks.oracle import load_queries

PREFIX = {"name_misleads": "name"}
REJECT_SLICES = ("offtopic", "injection")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates")
    parser.add_argument("wording")
    parser.add_argument("--author", required=True, help="who wrote the wording, e.g. claude-opus-5-5 or human")
    parser.add_argument("--existing", nargs="*", default=[])
    args = parser.parse_args()
    sys.stdout.reconfigure(newline="\n")  # query files use LF on every platform

    cands = load_queries(args.candidates)
    lines = [ln.strip() for ln in open(args.wording, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    texts, rejects = lines[:len(cands)], lines[len(cands):]
    if len(texts) != len(cands) or any(re.match(rf"({'|'.join(REJECT_SLICES)}):", t) for t in texts):
        raise SystemExit(f"{len(cands)} candidates but the wording has a different number of query lines")

    counter = {}
    for it in (it for path in args.existing for it in load_queries(path)):
        prefix, n = it["id"].rsplit("-", 1)
        counter[prefix] = max(counter.get(prefix, 0), int(n))

    def next_id(slice_):
        prefix = PREFIX.get(slice_, slice_)
        counter[prefix] = counter.get(prefix, 0) + 1
        return f"{prefix}-{counter[prefix]:03d}"

    for c, text in zip(cands, texts):
        item = {"id": next_id(c["slice"]), "slice": c["slice"], "query": text, "anchor_date": c["anchor_date"],
                "expect": c["expect"], "constraints": c["constraints"]}
        if c.get("gold_roles"):
            item["gold_roles"] = c["gold_roles"]
        if c.get("note"):
            item["note"] = c["note"]
        item["author"] = args.author
        print(json.dumps(item))
    for line in rejects:
        slice_, rest = line.split(":", 1)
        if slice_ not in REJECT_SLICES:
            raise SystemExit(f"reject line must start with one of {REJECT_SLICES}: {line}")
        text, _, note = rest.partition(" || ")
        item = {"id": next_id(slice_), "slice": slice_, "query": text.strip(), "expect": "reject"}
        if note:
            item["note"] = note.strip()
        item["author"] = args.author
        print(json.dumps(item))
