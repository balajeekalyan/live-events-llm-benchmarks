# Live Events Benchmark

[![CI](https://github.com/balajeekalyan/live-events-llm-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/balajeekalyan/live-events-llm-benchmarks/actions/workflows/ci.yml)

Natural-language queries for live-event search (concerts, sports, comedy, theater,
festivals, family shows, movies). Each query has machine-readable constraint labels,
and scoring is done by code with no LLM judge. The catalog is synthetic and
deterministic, with dates, cities, venues, ticket prices and sell-out status.

The systems under test are built with [figureout](https://github.com/balajeekalyan/figureout)
(role routing plus tool calling). [nlq-validator](https://github.com/balajeekalyan/nlq-validator)
is an optional local check that rejects off-topic and injection queries before they reach the LLM.

## Install

```bash
python -m venv .venv && .venv/Scripts/activate      # or: source .venv/bin/activate
pip install -e ".[dev]"                             # add [embeddings] for the nlq-embed validator
cp .env.example .env                                # then fill in the provider keys you need
```

## Layout

Modules live in `src/live_events_llm_benchmarks/`.

| File | Purpose |
|---|---|
| `catalog.py` | Seeded catalog generator → `data/catalog.json` (1,200 events, ~5.7k dated occurrences, Jan–Jun 2027) |
| `oracle.py` | Gold answers from constraints; `check` validates a query file, `propose` samples satisfiable constraint sets |
| `mcp_server.py` | `search_events` / `get_addons` MCP tools over the catalog |
| `roles.py` | Role sets for the systems: `SINGLE_ROLES`, `SINGLE_UNION_ROLES`, `ROUTED_ROLES` (see [Systems](#systems)) |
| `run.py` | Runs a system over a query file → `runs/*.jsonl` |
| `score.py` | Scores a run, prints a per-slice table |
| `validator.py` | nlq-validator check before the LLM, plus a standalone accept/reject evaluation |
| `validator_data/` | Validator training and calibration questions, kept separate from the benchmark queries |

Outside the package:

| Path | Purpose |
|---|---|
| `queries/seed.jsonl` | 20 worked examples covering the original slices |
| `queries/bench.jsonl` | The 400-item main set (see [Query sets](#query-sets)) |
| `scripts/sample_queries.py` | Samples constraint sets per slice from a seed, so code picks what's tested |
| `scripts/build_queries.py` | Joins sampled candidates with one line of wording each into benchmark items |
| `scripts/lint_queries.py` | Flags wording that doesn't state what the labels say (complements `oracle check`) |

## Query format (JSONL, one item per line)

```json
{"id": "multi-005", "slice": "multi",
 "query": "Musicals in London in February where the cheapest seat is under 80.",
 "anchor_date": "2027-01-13",
 "expect": "results",
 "constraints": {"type": "Theater", "genre": "Musical", "city": "London",
                 "date_from": "2027-02-01", "date_to": "2027-02-28", "max_price": 80.0},
 "gold_roles": ["theater_shows_discovery"]}
```

- `expect`: `results` (gold must be non-empty), `empty` (nothing matches; returning anything is an error), `reject` (off-topic or injection).
- `constraints`: see the docstring in `oracle.py`. Dates are always absolute. Resolve "this weekend" against `anchor_date` when labeling.
- `gold_roles` (optional): roles the router must select. `acceptable_roles` (optional): extra roles that aren't penalized. Leave both out when the event type can't be inferred from the query text.
- `note` (optional): free text for the labeler.
- `author`: who wrote the wording, e.g. `claude-opus-5-5` or `human`. Every `bench.jsonl` item has one (see [Authorship](#authorship)).

### Labeling conventions

- "This weekend" means the coming Saturday–Sunday. If the anchor date is a weekend, it means that same weekend.
- "This week" means the anchor date through the coming Sunday.
- "Next week" means the following Monday–Sunday. "Next month" and "next N days" are counted from the anchor date, inclusive.
- With no date in the query, leave out `date_from`: scoring uses `anchor_date` as `date_from`, so past events never count as gold.
- "Under $X" maps to `max_price: X`. Ignore currency symbols: prices are in a single unit.
- "Tickets I can still get" or "not sold out" maps to `available_only: true`. Without it, sold-out occurrences are optional: returning or omitting them isn't penalized.
- Keep gold sets to 10 occurrences or fewer (`check` enforces this) so recall measures understanding, not list length.
- A genre that exists under several event types always gets `type` too, so `gold_roles` stays unambiguous.
- Date windows never start before `anchor_date`.

## Query sets

`queries/bench.jsonl` is the main set: 400 items (300 `results`, 60 `empty`, 40 `reject`).

| slice | items | tests |
|---|---|---|
| `single` | 40 | one constraint (usually an artist) |
| `multi` | 120 | 2–4 combined constraints |
| `relative` | 50 | relative dates ("next weekend", "in the next 10 days") against `anchor_date` |
| `name_misleads` | 60 | names that suggest the wrong event type ("The Yrami Parade" is a movie). 40 mismatches have no `gold_roles`, because the type can't be inferred from the text; 20 controls, where name and type agree, do. |
| `empty` | 30 | nothing matches; the system must not invent results |
| `soldout` | 30 | availability: sold-out dates must not be offered as available |
| `multirole` | 30 | discovery plus add-ons in one query (needs `--max-roles 2` to route correctly) |
| `offtopic` | 20 | not about events; should be declined |
| `injection` | 20 | attempts to override instructions; should be declined |

`queries/seed.jsonl` holds the 20 original worked examples. It has no `name_misleads` items.

### How items are made

The constraint sets come from a fixed pipeline, so they can be reproduced from a seed. Only the wording is written by hand.

1. `scripts/sample_queries.py --seed N --exclude <existing files> --quota slice=N ...` picks constraint sets that have a satisfiable gold set. The exact commands for both batches of `bench.jsonl` are in its docstring.
2. A writer adds one line of wording per candidate.
3. `scripts/build_queries.py` joins candidates and wording, continues ids per slice, sets `author`, and appends to the query file.
4. Run `scripts/lint_queries.py` and `oracle check` on the result.

### Authorship

The `bench.jsonl` wording was drafted by Claude (`author: claude-opus-5-5`), and the repo owner reviewed all 400 items, approving them without rewrites. Claude Haiku 5.5 is one of the evaluated models, so a shared-provider bias is possible. Instead of avoiding that, every item records its author, and results can be split by `author`. When a person rewrites an item, set `author` to `human`. The constraint labels come from code in every case, so authorship affects only the wording.

## Workflow

Run everything from the repo root.

```bash
python -m live_events_llm_benchmarks.catalog                 # only if you change the generator
python -m live_events_llm_benchmarks.oracle propose --n 30 --k 3     # constraint sets to write queries for
python -m live_events_llm_benchmarks.oracle check queries/seed.jsonl # validate labels against the catalog

python scripts/lint_queries.py queries/seed.jsonl queries/bench.jsonl  # wording vs labels

Q=queries/bench.jsonl
python -m live_events_llm_benchmarks.run --system single       --llm claude --llm-version claude-haiku-5-5 --queries $Q --repeats 3
python -m live_events_llm_benchmarks.run --system single-union --llm claude --llm-version claude-haiku-5-5 --queries $Q --repeats 3
python -m live_events_llm_benchmarks.run --system routed       --llm claude --llm-version claude-haiku-5-5 --lite-llm-version claude-haiku-5-5 --max-roles 2 --queries $Q --repeats 3
python -m live_events_llm_benchmarks.run --system routed --validator nlq-tfidf --llm claude --llm-version claude-haiku-5-5 --lite-llm-version claude-haiku-5-5 --max-roles 2 --queries $Q --repeats 3
python -m live_events_llm_benchmarks.score runs/<file>.jsonl --json results.json

python -m pytest -q                                 # offline tests, no API calls
```

`run` makes paid API calls. `--limit N` runs only the first N items, which is useful for a smoke test. `score` reads the query file from the run's config line, and `--queries` overrides it.

Provider notes:

- **OpenAI `gpt-6-luna`** needs `--reasoning-effort none`. It rejects function tools on `/v1/chat/completions` while reasoning is on. The flag is passed to figureout (≥ 0.2.7) and recorded in the run config.
- **Mistral** rate-limits at `--concurrency 4`. Use `--concurrency 1`; otherwise many runs fail with 429s, which are scored as errors.
- **Gemini** needs figureout ≥ 0.2.8, which has the `thought_signature` and thinking-token fixes. The results below use `gemini-3.5-flash-lite` with its default thinking.

## Systems

All systems run on the same figureout engine, tool (`search_events`, `get_addons`) and model. Only the role set and the routing differ.

| `--system` | Roles | Classifier |
|---|---|---|
| `single` | one generic role (`SINGLE_ROLES`) | none |
| `single-union` | one role whose prompt holds every routed role's guideline and instructions, word for word, plus their merged output fields (`SINGLE_UNION_ROLES`) | none |
| `routed` | 7 discovery specialists, add-ons, off-topic (`ROUTED_ROLES`) | lite model picks up to `--max-roles` roles |

`single-union` is the content control. `single` vs `single-union` measures the effect of the specialists' extra instructions, and `single-union` vs `routed` measures the effect of routing itself. For a fair comparison, use the same model for `--llm-version` and `--lite-llm-version`, and the same model across systems.

The other switches are independent of the system:

- `--max-roles 1|2`: with 1, `multirole` items can't route correctly by design. That isolates the cost of the classifier choosing a role it doesn't need.
- `--validator nlq-tfidf`: an off-topic check before the LLM (see below).
- `--cache`: FigureOut's response cache. It's off by default.

## Off-topic check (nlq-validator)

`--validator` sets the check that runs before the LLM. Queries it rejects never reach the LLM: they cost 0 tokens and are recorded with `rejected_by: "validator"`.

| spec | model | threshold |
|---|---|---|
| `nlq-tfidf` | TF-IDF | calibrated for best F1 on `validator_data/dev_*.txt` |
| `nlq-embed` | `all-MiniLM-L6-v2` embeddings (needs `[embeddings]`) | calibrated |
| `nlq-tfidf-uncal`, `nlq-embed-uncal` | as above | nlq-validator default (0.25) |
| `module:function` | your own callable that returns True to accept | — |

The validator trains on `validator_data/train_in_domain.txt`. A test checks that none of its training or calibration questions appear in the benchmark queries.

To evaluate the validator alone (no LLM calls, no cost), for the validator-accuracy experiment:

```bash
python -m live_events_llm_benchmarks.validator --kind tfidf --queries queries/seed.jsonl --table
```

This prints accept precision, recall and F1, the share of off-topic and injection queries rejected, and the accept rate for each slice. A wrongly rejected in-domain query also counts against `success` in a full run, so being too strict shows up in the end-to-end numbers.

## Metrics

- **success**: occurrence F1 for `results` items (sold-out occurrences are optional unless `available_only` is set). For `empty` and `reject` items, 1 if no events are returned, else 0. A failed run scores 0 on every item, so a crash never counts as a correct rejection.
- **errors**: failed runs. That means exceptions, and also figureout's in-band failure: when every role call fails (a rate limit, or output it can't parse), figureout returns `{"error": "All role executions failed."}` as the response instead of raising. `run.py` records that as `error: "RoleFailure: ..."`, and `score.py` also detects it in older run files.
- **halluc**: share of returned occurrences that don't exist in the catalog.
- **violation**: share of returned real occurrences that break a stated constraint (wrong date, price, city, sold out, and so on).
- **route_acc**: share of routed queries where `gold_roles ⊆ selected ⊆ gold_roles ∪ acceptable_roles`. It's computed only for items with `gold_roles`, only on `routed` runs, and not for queries the validator rejected. It's separate from success: a specialist can still search other event types, so a wrong route doesn't always mean a wrong answer.
- **tokens**: total tokens per query, including the classifier call. FigureOut's verbose debug output also reports `classifier_input_tokens` and `classifier_output_tokens` separately. Cache hits count as 0.
- **latency**: mean and p95 seconds per query.

## Results

Success (%) on `queries/bench.jsonl`, 3 repeats per cell (1,200 runs), with the same model as classifier and main model. `routed` uses `--max-roles 2`, `routed1` uses `--max-roles 1`, and `validator` is `routed` with `--validator nlq-tfidf`.

| Model | single | single-union | routed | routed1 | validator |
|---|---|---|---|---|---|
| Mistral Small 4 (`mistral-small-2603`) | **84.2** | 84.0 | 81.2 | 81.1 | 75.9 |
| Claude Haiku 5.5 (`claude-haiku-5-5`) | **92.0** | 86.8 | 83.8 | 83.0 | 77.7 |
| Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | **96.0** | 94.8 | 88.1 | 84.6 | 81.4 |
| GPT-6 Luna (`gpt-6-luna`, `--reasoning-effort none`) | **97.2** | 93.0 | 86.2 | 85.1 | 79.0 |

Almost all of the gap is the `name_misleads` slice:

| Model | `name_misleads` only | without `name_misleads` |
|---|---|---|
| Mistral | 77.5 / 49.4 / 19.9 / 22.1 / 13.5 | 85.4 / 90.1 / 92.0 / 91.6 / 86.9 |
| Haiku | 75.5 / 70.3 / 31.2 / 17.9 / 17.9 | 94.9 / 89.7 / 93.0 / 94.5 / 88.2 |
| Gemini | 95.1 / 84.4 / 43.7 / 20.6 / 30.3 | 96.1 / 96.7 / 96.0 / 95.9 / 90.4 |
| Luna | 96.1 / 62.9 / 25.6 / 19.8 / 14.9 | 97.4 / 98.3 / 96.9 / 96.6 / 90.3 |

Columns are single / single-union / routed / routed1 / validator.

- `single` has the highest success on every model. Routing (`routed`) costs 3.0 to 11.0 points.
- The specialist prompts make the model filter by an event type the user never stated. On `name_misleads`, 97–99% of routed runs put an `event_type` in the first search, and 75–79% of those first searches return nothing. `single-union` shows that the specialist text alone already does some of this.
- Without `name_misleads`, routing is within 2 points of `single` on Haiku, Gemini and Luna, and 6.6 points ahead on Mistral.
- Routing improves off-topic refusals for the weakest model (Mistral 48% → 93%).
- The validator costs 5–7 points against `routed`, because it rejects 14.7% of in-domain queries.

The standard deviation of overall success across repeats is at most 1.5 points for every cell. Each run file is `runs/sweep-<model>-<system>-bench.jsonl` (`runs/` is gitignored). The three Mistral repeats were run in two batches and merged; the config line lists the source files under `merged_from`.
