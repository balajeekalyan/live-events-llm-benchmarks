"""Deterministic synthetic event catalog for the live-events benchmark.

Performer and venue names are synthetic on purpose: real artist names let a model
answer from prior knowledge instead of tool data, which hides hallucination.

Usage:
    python -m live_events_llm_benchmarks.catalog                 # rewrites the packaged data/catalog.json (seed 2027)
    python -m live_events_llm_benchmarks.catalog --seed 7 --n 2000 --out catalog_alt.json
"""
import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

WINDOW_START = date(2027, 1, 1)
WINDOW_END = date(2027, 6, 30)

# (city, region, country)
CITIES = [
    ("New York", "NY", "USA"), ("Boston", "MA", "USA"), ("Chicago", "IL", "USA"),
    ("Los Angeles", "CA", "USA"), ("San Francisco", "CA", "USA"), ("San Diego", "CA", "USA"),
    ("Seattle", "WA", "USA"), ("Portland", "OR", "USA"), ("Denver", "CO", "USA"),
    ("Austin", "TX", "USA"), ("Dallas", "TX", "USA"), ("Houston", "TX", "USA"),
    ("Atlanta", "GA", "USA"), ("Miami", "FL", "USA"), ("Orlando", "FL", "USA"),
    ("Nashville", "TN", "USA"), ("Philadelphia", "PA", "USA"), ("Washington", "DC", "USA"),
    ("Las Vegas", "NV", "USA"), ("Phoenix", "AZ", "USA"),
    ("Toronto", "ON", "Canada"), ("Montreal", "QC", "Canada"), ("Vancouver", "BC", "Canada"),
    ("London", "", "UK"), ("Manchester", "", "UK"), ("Glasgow", "", "UK"), ("Birmingham", "", "UK"),
    ("Dublin", "", "Ireland"), ("Paris", "", "France"), ("Berlin", "", "Germany"),
    ("Amsterdam", "", "Netherlands"), ("Barcelona", "", "Spain"), ("Lisbon", "", "Portugal"),
    ("Sydney", "NSW", "Australia"), ("Melbourne", "VIC", "Australia"),
]

VENUE_KINDS = ["Arena", "Hall", "Theatre", "Pavilion", "Stadium", "Amphitheater", "Club", "Center"]
VENUE_PREFIXES = ["Harbor", "Union", "Crescent", "Granite", "Maple", "Lantern", "Summit", "Riverside",
                  "Beacon", "Foundry", "Orchard", "Ironworks", "Meridian", "Copper", "Juniper", "Atlas"]

# type -> (share of catalog, genres, (low, high) cheapest-ticket price, occurrences range)
TYPES = {
    "Concert": (0.25, ["Pop", "R&B", "Rock", "Hip-Hop", "Country", "Jazz", "Indie", "Afrobeats", "Electronic", "Folk", "Soul"], (35, 180), (2, 8)),
    "Sports": (0.15, ["Basketball", "Soccer", "Baseball", "Hockey", "Tennis", "MMA", "Boxing", "Wrestling", "Motorsport"], (25, 250), (1, 6)),
    "Standup Comedy": (0.12, ["Comedy", "Improv"], (25, 90), (2, 7)),
    "Theater": (0.13, ["Musical", "Drama", "Opera", "Ballet"], (45, 220), (3, 10)),
    "Music Festival": (0.07, ["EDM", "House", "Techno", "Indie", "Hip-Hop", "Country", "Rock", "Folk"], (90, 400), (1, 2)),
    "Family and Children": (0.13, ["Circus", "Ice Show", "Children's Music", "Puppetry", "Interactive", "Educational"], (15, 75), (2, 8)),
    "Movie": (0.15, ["Action", "Drama", "Comedy", "Horror", "Sci-Fi", "Animation", "Thriller", "Documentary", "Romance"], (9, 25), (3, 10)),
}

SYLLABLES = ["ka", "lo", "mi", "ra", "ven", "sol", "tia", "dor", "ell", "ari", "zen", "nox", "lu", "mar",
             "ise", "qua", "rio", "bel", "tor", "una", "fen", "ost", "yra", "cal", "dex", "orin", "sa", "vel"]
TEAM_NOUNS = ["Comets", "Harriers", "Foxes", "Tide", "Rangers", "Owls", "Miners", "Blaze", "Pilots", "Wolves",
              "Herons", "Lynx", "Mariners", "Thunder", "Royals", "Stags"]
SHOW_WORDS = ["Midnight", "Paper", "Velvet", "Glass", "Hollow", "Silver", "Wild", "Northern", "Electric",
              "Quiet", "Golden", "Crimson", "Distant", "Little", "Broken", "Bright"]
SHOW_NOUNS = ["Garden", "Lighthouse", "Carnival", "Orchard", "Kingdom", "Machine", "River", "Letters",
              "Circus", "Parade", "Harbor", "Echo", "Lantern", "Comet", "Atlas", "Reverie"]

STATUSES = ["on_sale", "few_left", "sold_out"]
STATUS_WEIGHTS = [0.70, 0.18, 0.12]


def _word(rng, n_syll):
    return "".join(rng.choice(SYLLABLES) for _ in range(n_syll)).capitalize()


def _performer(rng, event_type):
    if event_type == "Sports":
        return f"{_word(rng, 2)} {rng.choice(TEAM_NOUNS)}"
    if event_type in ("Concert", "Standup Comedy"):
        return f"{_word(rng, 2)} {_word(rng, rng.choice([2, 3]))}"
    # 16x16 word pairs is too few for ~600 shows; mix in a synthetic proper noun
    if rng.random() < 0.5:
        return f"{rng.choice(SHOW_WORDS)} {rng.choice(SHOW_NOUNS)}"
    return f"The {_word(rng, 2)} {rng.choice(SHOW_NOUNS)}"


def _venues(rng):
    venues = {}
    for city, _, _ in CITIES:
        k = rng.randint(2, 4)
        venues[city] = [f"{rng.choice(VENUE_PREFIXES)} {kind}" for kind in rng.sample(VENUE_KINDS, k)]
    return venues


def generate(seed: int = 2027, n_events: int = 1200) -> dict:
    rng = random.Random(seed)
    venues = _venues(rng)
    span = (WINDOW_END - WINDOW_START).days
    used_names = set()
    events = []

    type_names = list(TYPES)
    shares = [TYPES[t][0] for t in type_names]
    for i in range(n_events):
        event_type = rng.choices(type_names, weights=shares)[0]
        _, genres, (lo, hi), (occ_lo, occ_hi) = TYPES[event_type]

        name = _performer(rng, event_type)
        while name in used_names:
            name = _performer(rng, event_type)
        used_names.add(name)

        # Most events tour a few cities; some stay in one city (residencies, runs).
        n_occ = rng.randint(occ_lo, occ_hi)
        home = rng.choice(CITIES)
        tour = rng.random() < 0.6
        details = []
        seen = set()
        for _ in range(n_occ):
            city, region, country = rng.choice(CITIES) if tour else home
            d = WINDOW_START + timedelta(days=rng.randint(0, span))
            if (d, city) in seen:
                continue
            seen.add((d, city))
            price_min = round(rng.uniform(lo, hi))
            details.append({
                "date": d.isoformat(),
                "city": city,
                "region": region,
                "country": country,
                "venue": rng.choice(venues[city]),
                "showtimes": sorted(rng.sample(["13:00", "15:30", "18:00", "19:00", "19:30", "20:00", "21:30"],
                                               3 if event_type == "Movie" else 1)),
                "price_min": float(price_min),
                "price_max": float(round(price_min * rng.uniform(1.5, 6))),
                "status": rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0],
            })
        details.sort(key=lambda x: (x["date"], x["city"]))

        events.append({
            "event_id": f"E{i + 1:04d}",
            "name": name,
            "artist": name,
            "type": event_type,
            "genre": rng.sample(genres, rng.choice([1, 1, 2])),
            "details": details,
        })

    return {
        "meta": {
            "seed": seed,
            "n_events": n_events,
            "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
            "generator": "benchmarks/live_events/catalog.py",
        },
        "events": events,
    }


def load(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else DATA_DIR / "catalog.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["events"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--out", default=str(DATA_DIR / "catalog.json"))
    args = parser.parse_args()

    catalog = generate(args.seed, args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=1, ensure_ascii=False)
    n_occ = sum(len(e["details"]) for e in catalog["events"])
    print(f"Wrote {len(catalog['events'])} events / {n_occ} occurrences to {out}")
