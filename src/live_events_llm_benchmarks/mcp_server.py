"""MCP tools over the benchmark catalog.

Filtering reuses the oracle's matchers, so if the model extracts the right
arguments the tool returns exactly the gold set. Errors therefore come from
argument extraction, tool choice, or what the model does with the result,
which is what the benchmark is meant to measure.
"""
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from live_events_llm_benchmarks import catalog as catalog_mod
from live_events_llm_benchmarks.oracle import event_violations, occurrence_violations

ADDONS_PATH = Path(__file__).parent / "data" / "addons.json"  # copied from figureout's event-sports-booking example

COUNTRY_ALIASES = {
    "us": "USA", "u.s.": "USA", "united states": "USA", "america": "USA",
    "uk": "UK", "united kingdom": "UK", "england": "UK", "britain": "UK", "great britain": "UK",
}

# Genres per event type, from the catalog generator, so the model can match the catalog's exact names
GENRES = "; ".join(f"{t}: {', '.join(spec[1])}" for t, spec in catalog_mod.TYPES.items())

bench_mcp = FastMCP("Live Events Benchmark MCP")
_events = None


def events() -> list[dict]:
    global _events
    if _events is None:
        _events = catalog_mod.load()
    return _events


def _constraints(event_type, genre, artist, city, country, from_date, to_date, max_price, available_only) -> dict:
    c = {}
    if event_type:
        c["type"] = event_type
    if genre:
        c["genre"] = genre
    if artist:
        c["artist"] = artist
    if city:
        c["city"] = city.split(",")[0].strip()
    if country:
        c["country"] = COUNTRY_ALIASES.get(country.strip().lower(), country)
    if from_date:
        c["date_from"] = from_date
    if to_date:
        c["date_to"] = to_date
    if max_price:
        c["max_price"] = float(max_price)
    if available_only:
        c["available_only"] = True
    return c


@bench_mcp.tool(
    name="search_events",
    description=(
        "Search the live events catalog. All filters are optional and combine with AND. "
        "event_type: one of Concert, Sports, Standup Comedy, Theater, Music Festival, Family and Children, Movie. "
        f"genre: exact name from this list, by event type ({GENRES}). "
        "artist: performer, team, or show name; when the user names one, don't guess event_type or genre from the name. city: city name only (e.g. 'Boston'). country: e.g. USA, UK, Canada. "
        "from_date/to_date: YYYY-MM-DD inclusive. max_price: cheapest ticket must be at or below this. "
        "available_only: true to exclude sold-out dates. Returns at most `limit` events, earliest first, with the total match count."
    ),
)
async def search_events(
    event_type: str = "",
    genre: str = "",
    artist: str = "",
    city: str = "",
    country: str = "",
    from_date: str = "",
    to_date: str = "",
    max_price: float = 0,
    available_only: bool = False,
    limit: int = 20,
) -> str:
    c = _constraints(event_type, genre, artist, city, country, from_date, to_date, max_price, available_only)
    results = []
    for e in events():
        if event_violations(e, c):
            continue
        details = [occ for occ in e["details"] if not occurrence_violations(occ, c)]
        if details:
            results.append({**e, "details": details})
    if not results:
        return json.dumps({"total": 0, "events": [], "message": "No events match these filters."})
    results.sort(key=lambda e: e["details"][0]["date"])
    limit = max(1, min(int(limit), 50))
    return json.dumps({"total": len(results), "returned": min(limit, len(results)), "events": results[:limit]})


@bench_mcp.tool(
    name="get_addons",
    description="Get add-ons and upgrades for an event booking: access (VIP lounge, parking, meet & greet), food, beverage, merch, insurance. Optionally filter by category.",
)
async def get_addons(category: str = "") -> str:
    with open(ADDONS_PATH, encoding="utf-8") as f:
        addons = json.load(f)
    if category:
        addons = [a for a in addons if a["category"].lower() == category.lower()]
    return json.dumps(addons)


if __name__ == "__main__":
    bench_mcp.run()
