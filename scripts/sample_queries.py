"""Sample constraint sets per slice for a batch of benchmark queries.

Code picks what is tested; a writer adds the wording (scripts/build_queries.py).
Prints one JSON object per candidate: constraints, gold sizes, and context for writing the query.
Artists and constraint sets already used in --exclude files are skipped, so batches don't repeat each other.

Usage:
    python scripts/sample_queries.py --seed 1 > candidates.jsonl              # batch 1 of queries/bench.jsonl
    python scripts/sample_queries.py --seed 2 --exclude queries/seed.jsonl queries/bench.jsonl \\
        --quota single=30 name_misleads=48 multi=90 relative=38 empty=22 soldout=22 multirole=22   # batch 2
"""
import argparse
import json
import random
from datetime import date, timedelta

from live_events_llm_benchmarks import catalog as catalog_mod
from live_events_llm_benchmarks.oracle import gold_split, item_constraints, load_queries

ROLE = {"Concert": "music_artist_discovery", "Music Festival": "music_festival_discovery", "Sports": "sports_discovery",
        "Standup Comedy": "standup_comedy_discovery", "Theater": "theater_shows_discovery",
        "Family and Children": "kids_family_shows_discovery", "Movie": "movie_discovery"}
FAMILY_WORDS = ("Circus", "Carnival", "Parade")  # names that read as family shows; a third of these events are
MAX_GOLD = 10
TODAY = "2027-01-13"  # default anchor_date, as in queries/seed.jsonl
DEFAULT_QUOTA = {"single": 10, "name_misleads": 12, "multi": 30, "relative": 12, "empty": 8, "soldout": 8, "multirole": 8}
PHRASES = ["this weekend", "next weekend", "next week", "in the next week", "in the next 10 days",
           "in the next month", "tomorrow", "tonight"]
ADDONS = ["parking", "VIP upgrades", "merch", "food packages", "drink packages", "ticket insurance"]

GENRE_TYPES = {}
for _t, _spec in catalog_mod.TYPES.items():
    for _g in _spec[1]:
        GENRE_TYPES.setdefault(_g, set()).add(_t)


class Sampler:
    def __init__(self, seed: int, exclude: list[dict]):
        self.rng = random.Random(seed)
        self.events = catalog_mod.load()
        self.used_artists = {it["constraints"]["artist"] for it in exclude if "artist" in it.get("constraints", {})}
        self.seen = {self._key(it["constraints"], it.get("anchor_date")) for it in exclude if it.get("constraints")}

    @staticmethod
    def _key(c, anchor_date):
        return json.dumps([c, anchor_date], sort_keys=True)

    def add_genre(self, c, e):
        """Genre constraint, plus the type when the genre exists under several types (keeps gold_roles unambiguous)."""
        g = self.rng.choice(e["genre"])
        c["genre"] = g
        if len(GENRE_TYPES[g]) > 1:
            c["type"] = e["type"]

    def sizes(self, c, anchor_date):
        req, opt = gold_split(self.events, item_constraints({"constraints": c, "anchor_date": anchor_date}))
        return len(req), len(opt)

    def ok_results(self, c, anchor_date):
        req, opt = self.sizes(c, anchor_date)
        return req >= 1 and req + opt <= MAX_GOLD

    def fresh(self, e):
        if e["artist"] in self.used_artists:
            return False
        self.used_artists.add(e["artist"])
        return True

    def emit(self, slice_, c, anchor_date, expect="results", roles=None, **ctx):
        """Print the candidate; False (and nothing printed) if this constraint set was already used."""
        key = self._key(c, anchor_date)
        if key in self.seen:
            return False
        self.seen.add(key)
        req, opt = self.sizes(c, anchor_date)
        out = {"slice": slice_, "anchor_date": anchor_date,
               "anchor_weekday": date.fromisoformat(anchor_date).strftime("%A"),
               "expect": expect, "constraints": c, "gold_required": req, "gold_optional": opt}
        if roles:
            out["gold_roles"] = roles
        out.update(ctx)
        print(json.dumps(out))
        return True

    def single(self, n):
        pool = [e for e in self.events if not e["artist"].endswith(FAMILY_WORDS)]
        self.rng.shuffle(pool)
        done = 0
        for e in pool:
            c = {"artist": e["artist"]}
            if done < n and self.ok_results(c, TODAY) and self.fresh(e):
                done += self.emit("single", c, TODAY, roles=[ROLE[e["type"]]], type=e["type"], genre=e["genre"])

    def name_misleads(self, n):
        """Two thirds name/type mismatches (routing unscored), one third controls where they agree."""
        n_mis = n * 2 // 3
        n_ctrl = n - n_mis
        pool = [e for e in self.events if e["artist"].endswith(FAMILY_WORDS)]
        self.rng.shuffle(pool)
        mis = ctrl = 0
        for e in pool:
            c = {"artist": e["artist"]}
            if not self.ok_results(c, TODAY):
                continue
            family = e["type"] == "Family and Children"
            if family and ctrl < n_ctrl and self.fresh(e):
                ctrl += self.emit("name_misleads", c, TODAY, roles=[ROLE[e["type"]]], type=e["type"], genre=e["genre"],
                                  note="control: name and type agree")
            elif not family and mis < n_mis and self.fresh(e):
                mis += self.emit("name_misleads", c, TODAY, type=e["type"], genre=e["genre"],
                                 note=f"a {e['type']} despite the name; type is not inferable from the query, "
                                      "so routing is unscored")
        if mis < n_mis or ctrl < n_ctrl:
            raise SystemExit(f"name_misleads: catalog ran out ({mis}/{n_mis} mismatches, {ctrl}/{n_ctrl} controls)")

    def multi(self, n):
        rng = self.rng
        done = 0
        while done < n:
            e = rng.choice(self.events)
            occ = rng.choice(e["details"])
            if occ["status"] == "sold_out" or occ["date"] < "2027-01-20":
                continue
            d = date.fromisoformat(occ["date"])
            c = {}
            if rng.random() < 0.6:
                self.add_genre(c, e)
                if rng.random() < 0.4:
                    c["type"] = e["type"]
            else:
                c["type"] = e["type"]
            extras = rng.sample(["place", "dates", "price", "available"], rng.choice([1, 2, 2, 3]))
            if "place" in extras:
                if rng.random() < 0.7:
                    c["city"] = occ["city"]
                else:
                    c["country"] = occ["country"]
            if "dates" in extras:
                if rng.random() < 0.5:
                    start = d.replace(day=1)
                    end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                else:
                    start = d - timedelta(days=rng.choice([0, 1, 2, 3]))
                    end = d + timedelta(days=rng.choice([0, 1, 2, 4]))
                start = max(start, date.fromisoformat(TODAY))  # never ask about the past
                c["date_from"], c["date_to"] = start.isoformat(), end.isoformat()
            if "price" in extras:
                c["max_price"] = float(int(occ["price_min"] // 10 * 10 + rng.choice([10, 20, 30])))
            if "available" in extras:
                c["available_only"] = True
            if self.ok_results(c, TODAY):
                done += self.emit("multi", c, TODAY, roles=[ROLE[e["type"]]], type=e["type"])

    @staticmethod
    def window(phrase, a):
        wd = a.weekday()  # Monday 0
        sat = a + timedelta(days=(5 - wd) % 7)
        if phrase == "this weekend":
            return sat, sat + timedelta(days=1)
        if phrase == "next weekend":  # the weekend after this one
            return sat + timedelta(days=7), sat + timedelta(days=8)
        if phrase == "next week":  # Monday to Sunday
            mon = a + timedelta(days=7 - wd)
            return mon, mon + timedelta(days=6)
        if phrase == "in the next week":
            return a, a + timedelta(days=7)
        if phrase == "in the next 10 days":
            return a, a + timedelta(days=10)
        if phrase == "in the next month":
            return a, a.replace(month=a.month + 1)
        if phrase == "tomorrow":
            return a + timedelta(days=1), a + timedelta(days=1)
        return a, a  # tonight

    def relative(self, n):
        rng = self.rng
        done = 0
        while done < n:
            a = date(2027, 1, 4) + timedelta(days=rng.randrange(0, 120))
            if a.weekday() >= 5 or a.day > 28:
                continue  # weekend anchors make "this weekend" ambiguous; day > 28 has no same day next month
            phrase = PHRASES[done % len(PHRASES)]
            start, end = self.window(phrase, a)
            e = rng.choice(self.events)
            occs = [o for o in e["details"] if start.isoformat() <= o["date"] <= end.isoformat()]
            if not occs:
                continue
            occ = rng.choice(occs)
            c = {}
            r = rng.random()
            if r < 0.4:
                self.add_genre(c, e)
            elif r < 0.8:
                c["type"] = e["type"]
            c["city"] = occ["city"]
            c["date_from"], c["date_to"] = start.isoformat(), end.isoformat()
            if self.ok_results(c, a.isoformat()):
                done += self.emit("relative", c, a.isoformat(),
                                  roles=[ROLE[e["type"]]] if c.get("type") or c.get("genre") else None,
                                  phrase=phrase, type=e["type"])

    def empty(self, n):
        rng = self.rng
        done = 0
        cities = sorted({o["city"] for e in self.events for o in e["details"]})
        while done < n:
            e = rng.choice(self.events)
            c = {"city": rng.choice(cities)}
            self.add_genre(c, e)
            if rng.random() < 0.5:
                c["type"] = e["type"]
            d = date.fromisoformat(TODAY) + timedelta(days=rng.randrange(0, 150))
            c["date_from"], c["date_to"] = d.isoformat(), (d + timedelta(days=rng.choice([1, 2, 6]))).isoformat()
            if self.sizes(c, TODAY) == (0, 0):
                done += self.emit("empty", c, TODAY, expect="empty", roles=[ROLE[e["type"]]], type=e["type"],
                                  start_weekday=d.strftime("%A"))

    def soldout(self, n):
        done = 0
        pool = [(e, o) for e in self.events for o in e["details"] if o["status"] == "sold_out" and o["date"] >= "2027-01-20"]
        self.rng.shuffle(pool)
        for e, o in pool:
            if done >= n:
                break
            c = {"artist": e["artist"], "city": o["city"], "date_from": o["date"], "date_to": o["date"],
                 "available_only": True}
            if self.sizes(c, TODAY) == (0, 0) and self.fresh(e):
                done += self.emit("soldout", c, TODAY, expect="empty", roles=[ROLE[e["type"]]], type=e["type"],
                                  venue=o["venue"], note="only occurrence is sold out")

    def multirole(self, n):
        rng = self.rng
        done = 0
        while done < n:
            e = rng.choice(self.events)
            occ = rng.choice(e["details"])
            start = date.fromisoformat(occ["date"]).replace(day=1)
            end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            c = {"city": occ["city"], "date_from": max(start, date.fromisoformat(TODAY)).isoformat(),
                 "date_to": end.isoformat()}
            self.add_genre(c, e)
            if self.ok_results(c, TODAY):
                done += self.emit("multirole", c, TODAY, roles=[ROLE[e["type"]], "addons_selection"], type=e["type"],
                                  addon=ADDONS[(done + 1) % len(ADDONS)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exclude", nargs="*", default=[], help="query files whose artists and constraint sets to skip")
    parser.add_argument("--quota", nargs="*", default=[], help="slice=N overrides, e.g. multi=90")
    args = parser.parse_args()

    quota = dict(DEFAULT_QUOTA)
    for q in args.quota:
        name, n = q.split("=")
        if name not in quota:
            raise SystemExit(f"unknown slice {name}; one of {sorted(quota)}")
        quota[name] = int(n)
    sampler = Sampler(args.seed, [it for path in args.exclude for it in load_queries(path)])
    for name in DEFAULT_QUOTA:  # fixed order keeps a seed reproducible
        getattr(sampler, name)(quota[name])
