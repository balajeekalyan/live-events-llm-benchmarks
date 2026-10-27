"""Offline tests for the live-events benchmark: no LLM calls.

Run from the repo root: python -m pytest -q
"""
import asyncio
import json
import re
from pathlib import Path

from live_events_llm_benchmarks import catalog as catalog_mod
from live_events_llm_benchmarks import mcp_server, score
from live_events_llm_benchmarks.oracle import gold, gold_split, item_constraints, load_queries, occ_key, validate_item, violations

SEED = Path(__file__).parents[1] / "queries" / "seed.jsonl"

EVENT = {
    "event_id": "E1", "name": "A", "artist": "Band A", "type": "Concert", "genre": ["Pop"],
    "details": [
        {"date": "2027-01-16", "city": "Boston", "region": "MA", "country": "USA", "venue": "X Hall",
         "showtimes": ["20:00"], "price_min": 40.0, "price_max": 120.0, "status": "on_sale"},
        {"date": "2027-01-20", "city": "Boston", "region": "MA", "country": "USA", "venue": "X Hall",
         "showtimes": ["20:00"], "price_min": 90.0, "price_max": 200.0, "status": "sold_out"},
    ],
}


def test_catalog_is_deterministic():
    assert catalog_mod.generate(1, 50) == catalog_mod.generate(1, 50)


def test_committed_catalog_matches_generator():
    with open(catalog_mod.DATA_DIR / "catalog.json", encoding="utf-8") as f:
        committed = json.load(f)
    meta = committed["meta"]
    assert committed == catalog_mod.generate(meta["seed"], meta["n_events"])


def test_gold_applies_every_constraint():
    events = [EVENT]
    assert gold(events, {"city": "Boston"}) == {("E1", "2027-01-16", "boston"), ("E1", "2027-01-20", "boston")}
    assert gold(events, {"max_price": 50}) == {("E1", "2027-01-16", "boston")}
    assert gold(events, {"available_only": True}) == {("E1", "2027-01-16", "boston")}
    assert gold(events, {"date_from": "2027-01-17"}) == {("E1", "2027-01-20", "boston")}
    assert gold(events, {"genre": "rock"}) == set()


def test_anchor_date_is_the_default_date_from():
    item = {"constraints": {"city": "Boston"}, "anchor_date": "2027-01-17"}
    assert gold([EVENT], item_constraints(item)) == {("E1", "2027-01-20", "boston")}
    explicit = {"constraints": {"date_from": "2027-01-01"}, "anchor_date": "2027-01-17"}
    assert item_constraints(explicit)["date_from"] == "2027-01-01"


def test_sold_out_gold_is_optional_unless_available_only():
    sold_out = ("E1", "2027-01-20", "boston")
    assert gold_split([EVENT], {"city": "Boston"}) == ({("E1", "2027-01-16", "boston")}, {sold_out})
    assert gold_split([EVENT], {"city": "Boston", "available_only": True}) == ({("E1", "2027-01-16", "boston")}, set())


def test_violations_name_the_broken_constraint():
    occ = EVENT["details"][1]
    assert violations(EVENT, occ, {"max_price": 50, "available_only": True}) == ["price", "availability"]


def test_seed_queries_are_valid():
    for item in load_queries(SEED):
        assert validate_item(item) == [], item["id"]


def test_every_query_file_is_consistent_with_the_catalog(capsys):
    from live_events_llm_benchmarks.oracle import check
    events = catalog_mod.load()
    for path in sorted(SEED.parent.glob("*.jsonl")):
        assert check(str(path), events, max_gold=10) == 0, capsys.readouterr().out


def test_tool_returns_gold_for_correct_arguments():
    events = catalog_mod.load()
    c = {"type": "Theater", "genre": "Musical", "city": "London", "date_from": "2027-02-01",
         "date_to": "2027-02-28", "max_price": 80.0}
    raw = asyncio.run(mcp_server.search_events(
        event_type="Theater", genre="Musical", city="London, UK", from_date="2027-02-01",
        to_date="2027-02-28", max_price=80, limit=50))
    returned = {occ_key(e["event_id"], d) for e in json.loads(raw)["events"] for d in e["details"]}
    assert returned == gold(events, c)


def _write_run(tmp_path, records):
    path = tmp_path / "run.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"config": {"queries": str(SEED)}}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _perfect_response(item, events):
    if item["expect"] != "results":
        return {"events": []}
    by_id = {e["event_id"]: e for e in events}
    g = gold(events, item_constraints(item))
    out = {}
    for eid, d, city in g:
        occ = next(o for o in by_id[eid]["details"] if o["date"] == d and o["city"].lower() == city)
        out.setdefault(eid, {"event_id": eid, "details": []})["details"].append(occ)
    return {"events": list(out.values())}


def test_perfect_run_scores_full_marks(tmp_path):
    events = catalog_mod.load()
    items = load_queries(SEED)
    records = [{"id": it["id"], "repeat": 0, "response": _perfect_response(it, events),
                "roles_selected": it.get("gold_roles", []), "latency_s": 1.0} for it in items]
    result = score.score_run(str(_write_run(tmp_path, records)))
    assert result["overall"]["success"] == 1.0
    assert result["overall"]["halluc"] == 0.0
    assert result["overall"]["violation"] == 0.0
    assert result["overall"]["route_acc"] == 1.0


def test_invented_and_violating_events_are_counted(tmp_path):
    events = catalog_mod.load()
    item = next(it for it in load_queries(SEED) if it["id"] == "soldout-001")
    real = next(e for e in events if e["artist"] == "Rioka Vensa")
    response = {"events": [
        {"event_id": real["event_id"], "details": [{"date": "2027-03-17", "city": "Boston"}]},  # sold out
        {"event_id": "E9999", "details": [{"date": "2027-03-17", "city": "Boston"}]},  # invented
    ]}
    result = score.score_run(str(_write_run(tmp_path, [{"id": item["id"], "response": response}])))
    row = result["items"][0]
    assert row["success"] == 0.0
    assert row["n_halluc"] == 1
    assert row["n_violation"] == 1


def test_sold_out_labeled_as_sold_out_is_neutral_when_available_only(tmp_path):
    events = catalog_mod.load()
    item = next(it for it in load_queries(SEED) if it["id"] == "soldout-001")
    real = next(e for e in events if e["artist"] == "Rioka Vensa")
    labeled = {"events": [{"event_id": real["event_id"],
                           "details": [{"date": "2027-03-17", "city": "Boston", "status": "sold_out"}]}]}
    mislabeled = {"events": [{"event_id": real["event_id"],
                              "details": [{"date": "2027-03-17", "city": "Boston", "status": "on_sale"}]}]}
    rows = score.score_run(str(_write_run(tmp_path, [{"id": item["id"], "response": labeled},
                                                     {"id": item["id"], "repeat": 1, "response": mislabeled}])))["items"]
    assert (rows[0]["success"], rows[0]["n_violation"]) == (1.0, 0)
    assert (rows[1]["success"], rows[1]["n_violation"]) == (0.0, 1)


def test_failed_runs_are_errors_and_misses_on_reject_items(tmp_path):
    item = next(it for it in load_queries(SEED) if it["expect"] != "results")
    records = [{"id": item["id"], "response": {"events": []}},
               {"id": item["id"], "repeat": 1, "response": {"error": "All role executions failed."}},
               {"id": item["id"], "repeat": 2, "error": "RuntimeError: boom", "response": None}]
    rows = score.score_run(str(_write_run(tmp_path, records)))["items"]
    assert [(r["success"], r["error"]) for r in rows] == [(1.0, False), (0.0, True), (0.0, True)]


def test_multirole_response_is_searched_for_events():
    nested = {"music_artist_discovery": {"events": [{"event_id": "E1", "details": [{"date": "2027-01-16", "city": "Boston, MA"}]}]},
              "addons_selection": {"addons": []}}
    assert score.predicted_occurrences(nested) == {("E1", "2027-01-16", "boston")}


def test_validator_gate_rejects_off_topic_and_accepts_events():
    from live_events_llm_benchmarks.validator import make_gate
    gate = make_gate("nlq-tfidf")
    accepted, score_ = gate("Any comedy shows in Chicago this weekend?")
    assert accepted and score_ is not None
    assert gate("What's a good recipe for banana bread?")[0] is False


def test_validator_training_data_does_not_leak_benchmark_queries():
    from live_events_llm_benchmarks.validator import DATA_DIR
    train = set()
    for name in ("train_in_domain.txt", "dev_in_domain.txt", "dev_off_domain.txt"):
        train |= {line.strip().lower() for line in open(DATA_DIR / name, encoding="utf-8") if line.strip()}
    for item in load_queries(SEED):
        assert item["query"].strip().lower() not in train, item["id"]


def test_tool_calls_are_logged_per_item():
    from live_events_llm_benchmarks import run

    class FakeEngine:
        async def _execute_tool(self, name, args):
            await asyncio.sleep(0)
            return await mcp_server.search_events(**args)

    fo = FakeEngine()
    run.log_tool_calls(fo)

    async def item(city):
        calls = []
        run._tool_calls.set(calls)
        await asyncio.gather(fo._execute_tool("search_events", {"city": city}))  # tools run in child tasks
        return calls

    async def both():
        return await asyncio.gather(asyncio.create_task(item("Boston")), asyncio.create_task(item("Nowhere")))

    boston, nowhere = asyncio.run(both())
    assert [c["arguments"] for c in boston] == [{"city": "Boston"}] and boston[0]["total"] > 0
    assert nowhere == [{"name": "search_events", "arguments": {"city": "Nowhere"}, "total": 0}]


def test_tool_description_lists_every_catalog_genre():
    description = mcp_server.bench_mcp._tool_manager._tools["search_events"].description
    for e in catalog_mod.load():
        for g in e["genre"]:
            assert g in description, g


def test_returning_or_omitting_sold_out_is_not_penalized(tmp_path):
    events = catalog_mod.load()
    item = next(it for it in load_queries(SEED) if it["id"] == "relative-001")
    required, optional = gold_split(events, item_constraints(item))
    assert required and optional  # the item has both kinds, or this test proves nothing
    full = _perfect_response(item, events)
    on_sale_only = {"events": [e for e in full["events"]
                               if any(occ_key(e["event_id"], d) in required for d in e["details"])]}
    for e in on_sale_only["events"]:
        e["details"] = [d for d in e["details"] if occ_key(e["event_id"], d) in required]
    for response in (full, on_sale_only):
        result = score.score_run(str(_write_run(tmp_path, [{"id": item["id"], "response": response}])))
        assert result["items"][0]["success"] == 1.0


def test_type_neutral_roles_drop_only_the_event_type_instruction():
    from live_events_llm_benchmarks import roles
    for typed, neutral in ((roles.ROUTED_ROLES, roles.ROUTED_NEUTRAL_ROLES),
                           (roles.SINGLE_UNION_ROLES, roles.SINGLE_UNION_NEUTRAL_ROLES)):
        assert typed.keys() == neutral.keys()
        for name, role in typed.items():
            n = neutral[name]
            assert "event_type" not in n.prompt
            assert (n.schema, n.guideline) == (role.schema, role.guideline)
            assert re.sub(r"event_type '[^']*' and ", "", role.prompt) == n.prompt
    assert roles.ROUTED_SYSTEMS <= roles.SYSTEMS.keys()
