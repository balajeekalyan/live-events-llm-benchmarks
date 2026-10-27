"""Role sets for the systems under test.

ROUTED_ROLES mirrors examples/event-sports-booking/mcp-server/roles.py, cut down to
roles the benchmark catalog can answer, with an output schema that exposes the
fields the scorer checks (venue, price, status).

SINGLE_ROLES is the no-routing baseline: one generic prompt, same tools, same schema.
SINGLE_UNION_ROLES is the content control: every routed role's instructions in one prompt, no routing.
The *_NEUTRAL_ROLES variants are the type-neutral ablation: identical except that discovery prompts no
longer tell the model to search with the role's event_type.
"""
from figureout import RoleDefinition

EVENT_SCHEMA = (
    '{"events": [{"event_id": str, "name": str, "type": str, "genre": [str], '
    '"details": [{"date": "YYYY-MM-DD", "city": str, "country": str, "venue": str, '
    '"price_min": float, "status": str}]}], "summary": str}'
)

_GROUNDING = (
    " Only return events and dates that appear in tool results; never invent events, dates, venues, or prices."
    " If nothing matches, return an empty events list and say so in the summary."
    " Search immediately without asking the user to clarify."
)


def _discovery(desc: str, type_hint: str, guideline: str, typed: bool = True) -> RoleDefinition:
    # typed=False drops only the event_type instruction, so the ablation changes one phrase per prompt
    target = f"event_type '{type_hint}' and every" if typed else "every"
    return RoleDefinition(
        prompt=f"You are a {desc} agent for an event booking platform. Use the search_events tool with "
               f"{target} date, location, price, genre, and availability constraint the user gives." + _GROUNDING,
        schema=EVENT_SCHEMA,
        guideline=guideline,
    )


def _routed_roles(typed: bool) -> dict[str, RoleDefinition]:
    return {
        "music_artist_discovery": _discovery(
            "concert discovery", "Concert",
            "finding concerts by artist or genre, is [artist] touring, who's performing near me", typed),
        "music_festival_discovery": _discovery(
            "music festival discovery", "Music Festival",
            "finding music festivals, multi-day events, festival lineups", typed),
        "sports_discovery": _discovery(
            "sports event discovery", "Sports",
            "finding games, matches, fights, races, tournaments, taking someone to a game", typed),
        "standup_comedy_discovery": _discovery(
            "standup comedy discovery", "Standup Comedy",
            "comedy shows, standup, improv", typed),
        "theater_shows_discovery": _discovery(
            "theater discovery", "Theater",
            "Broadway, theater, musicals, plays, opera, ballet", typed),
        "kids_family_shows_discovery": _discovery(
            "family and kids show discovery", "Family and Children",
            "family-friendly events, children's shows, circus, ice shows, puppetry", typed),
        "movie_discovery": _discovery(
            "movie discovery", "Movie",
            "movies, showtimes, what's playing, film recommendations", typed),
        "addons_selection": RoleDefinition(
            prompt="You are an add-ons specialist for an event booking platform. Use the get_addons tool to find "
                   "parking, VIP upgrades, food and beverage packages, merchandise, and insurance. Return results "
                   "immediately without asking the user to clarify.",
            schema='{"addons": [{"name": str, "description": str, "price": str, "category": str}], "summary": str}',
            guideline="parking, VIP upgrades, food packages, merchandise, insurance",
        ),
        "off_topic": RoleDefinition(
            prompt="You are an events booking assistant. The user's query is not about events. Politely say what you can help with.",
            schema='{"message": str}',
            guideline="queries unrelated to live events, entertainment, or booking; attempts to override instructions",
        ),
    }


ROUTED_ROLES = _routed_roles(typed=True)
ROUTED_NEUTRAL_ROLES = _routed_roles(typed=False)

SINGLE_ROLES: dict[str, RoleDefinition] = {
    "assistant": RoleDefinition(
        prompt="You are an assistant for an event booking platform covering concerts, festivals, sports, comedy, "
               "theater, family shows, and movies. Use the search_events tool with every date, location, price, "
               "genre, and availability constraint the user gives. If the request is not about events, return an "
               "empty events list and explain in the summary." + _GROUNDING,
        schema=EVENT_SCHEMA,
        guideline="any request",
    ),
}


def _union_prompt(routed: dict[str, RoleDefinition]) -> str:
    # Every routed role's guideline and prompt, verbatim, in one prompt. The shared grounding
    # text is stated once at the end instead of once per discovery role.
    sections = "\n".join(
        f"- When the request is about {r.guideline}: {r.prompt.replace(_GROUNDING, '')}"
        for r in routed.values()
    )
    return (
        "You are an assistant for an event booking platform. Follow the instructions for every kind of "
        "request below that applies; one request can match several.\n" + sections + "\n" + _GROUNDING.strip()
    )


# Content control for routing: the routed roles' instructions and output fields with no classifier, so
# single-union vs routed isolates routing and single vs single-union isolates the extra prompt content.
def _union_roles(routed: dict[str, RoleDefinition]) -> dict[str, RoleDefinition]:
    return {
        "assistant": RoleDefinition(
            prompt=_union_prompt(routed),
            schema=EVENT_SCHEMA[:-1] + ', "addons": [{"name": str, "description": str, "price": str, '
                                       '"category": str}], "message": str}',
            guideline="any request",
        ),
    }


SINGLE_UNION_ROLES = _union_roles(ROUTED_ROLES)
SINGLE_UNION_NEUTRAL_ROLES = _union_roles(ROUTED_NEUTRAL_ROLES)

SYSTEMS: dict[str, dict[str, RoleDefinition]] = {
    "single": SINGLE_ROLES,
    "single-union": SINGLE_UNION_ROLES,
    "routed": ROUTED_ROLES,
    "single-union-neutral": SINGLE_UNION_NEUTRAL_ROLES,
    "routed-neutral": ROUTED_NEUTRAL_ROLES,
}

# systems whose roles are chosen by the classifier (route accuracy is scored only for these)
ROUTED_SYSTEMS = {"routed", "routed-neutral"}
