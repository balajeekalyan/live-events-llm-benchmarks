"""Check that each query's wording states what its constraint labels say. Complements `oracle check`, which only
checks labels against the catalog.

Flags: artist/city/country not named; a price that differs from max_price (or a price with no max_price);
an explicit date range whose months or days aren't in the text; available_only with no availability wording;
type words in name_misleads queries; the same wording twice. Relative-date items are skipped for dates (their
windows come from phrases like "next week"). Heuristic, so a flagged item may be fine: read it.

Usage:
    python scripts/lint_queries.py queries/seed.jsonl queries/bench.jsonl
"""
import re
import sys
from datetime import date, timedelta

from live_events_llm_benchmarks.oracle import load_queries

MONTHS = ["january", "february", "march", "april", "may", "june", "july"]
COUNTRY_WORDS = {"USA": ["US", "U.S.", "United States", "America"], "UK": ["UK", "U.K.", "United Kingdom", "Britain"]}
TYPE_CUES = (r"\b(concert|gig|movie|film|screen|showtime|game|match|play(s|ing)?|comed|stand-?up|festival|theat|"
             r"musical|opera|ballet|kids|family|show)\w*")
AVAILABLE = (r"sold out|left|available|still|seats|buy|book|get (me )?(into|tickets|seats)|need tickets|"
             r"tickets (for|to)|want tickets")
RELATIVE = r"\b(this|next) (week|weekend|month)\b|\btonight\b|\btomorrow\b|\bin the next\b"


def month_names(text):
    found = set()
    for i, m in enumerate(MONTHS, 1):
        if re.search(rf"\b{m[:3]}\w*\b", text, re.I):
            found.add(i)
    return found


def problems(item):
    q, c = item["query"], item.get("constraints", {})
    out = []
    for key in ("artist", "city"):
        if key in c and c[key].lower() not in q.lower():
            out.append(f"{key} '{c[key]}' not in text")
    if "country" in c and not any(w.lower() in q.lower() for w in [c["country"], *COUNTRY_WORDS.get(c["country"], [])]):
        out.append(f"country '{c['country']}' not in text")

    # a price is "$N", "N or less", or a number too big to be a day
    prices = [float(m.group(2)) for m in re.finditer(r"(\$)?\b(\d{2,3})\b(?! *(?:to|through|and) *\d)(?=( or less)?)", q)
              if m.group(1) or m.group(3) or int(m.group(2)) > 31]
    if item["expect"] != "reject":
        if "max_price" in c and prices != [c["max_price"]]:
            out.append(f"price {prices} != max_price {c['max_price']}")
        if "max_price" not in c and prices:
            out.append(f"price {prices} in text but no max_price")

    if "date_from" in c and item["slice"] != "relative" and not re.search(RELATIVE, q, re.I):
        d1, d2 = date.fromisoformat(c["date_from"]), date.fromisoformat(c["date_to"])
        rest_of_month = "rest of" in q.lower() and d1 == date.fromisoformat(item.get("anchor_date", c["date_from"]))
        whole_month = (d1.day == 1 or rest_of_month) and (d2 + timedelta(days=1)).month != d2.month
        if {d1.month, d2.month} - month_names(q) and not rest_of_month:
            out.append(f"month of {d1}..{d2} not in text")
        days = {int(x) for x in re.findall(r"\b(\d{1,2})\b", q)}
        if not whole_month and {d1.day, d2.day} - days:
            out.append(f"days of {d1}..{d2} not in text")
    if c.get("available_only") and not re.search(AVAILABLE, q, re.I):
        out.append("available_only but no availability wording")
    if item["slice"] == "name_misleads" and re.search(TYPE_CUES, q.replace(c.get("artist", ""), ""), re.I):
        out.append("type cue in a name_misleads query")
    return out


if __name__ == "__main__":
    items = [it for path in sys.argv[1:] for it in load_queries(path)]
    seen = {}
    n = 0
    for it in items:
        errs = problems(it)
        key = it["query"].strip().lower()
        if key in seen:
            errs.append(f"same wording as {seen[key]}")
        seen.setdefault(key, it["id"])
        if errs:
            n += 1
            print(f"{it['id']:<16} {it['query']}\n{'':<16} {'; '.join(errs)}")
    print(f"\n{len(items)} items, {n} flagged")
    sys.exit(1 if n else 0)
