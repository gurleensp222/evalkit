# evalkit

A two-pass evaluation harness for language models. One model answers a set of prompts. Built-in scorers or a separate grader model then score each saved answer against versioned, weighted rubrics. Everything is stored in SQLite, so you can re-grade the same answers with a new rubric or a different judge without calling the answer model again.

[![CI](https://github.com/gurleensp222/evalkit/actions/workflows/ci.yml/badge.svg)](https://github.com/gurleensp222/evalkit/actions/workflows/ci.yml)

## Quick start

### Local, no Docker

```bash
python -m pip install -e ".[dev]"
evalkit run --dataset datasets/prompts --provider ollama --model llama3.2
evalkit report <run_id>
evalkit show <run_id> <task_id>
```

Runs are saved to `runs/evalkit.db`.

### With Ollama

```bash
evalkit run --provider ollama --model llama3.2 --temperature 0.2
evalkit run --config agents.example.yaml
```

`agents.example.yaml` sets the answer model to Ollama `llama3.2` at temperature 0.7 and the grader to `qwen2.5:3b` at temperature 0. A command-line flag replaces the matching value in the file. Before a run starts, evalkit checks Ollama's `/api/tags`. If a model is missing, it stops with `run: ollama pull <model>`.

### Docker

Create an empty `.env` in the project directory. Put `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` there when you call a hosted model.

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.2
```

The API listens on http://localhost:8000, with interactive docs at http://localhost:8000/docs. The database is written to `./runs`. Compose also starts an `ollama` service, sets `OLLAMA_URL=http://ollama:11434` on the API, and keeps model files in the `ollama_models` volume.

If Ollama already runs on your computer, delete the compose `ollama` service and the API's `depends_on` block, and point the API at the host:

```yaml
environment:
  OLLAMA_URL: http://host.docker.internal:11434
extra_hosts:
  - "host.docker.internal:host-gateway"   # needed on Linux
```

> **Security:** the API has no authentication, and a task that uses `python_unit_tests` runs code on the server. Run the API only on your own machine or a trusted network, and do not expose port 8000 or 11434 to the internet.

## Two passes: answer, then grade

```bash
evalkit answer --dataset datasets/prompts --provider ollama --model llama3.2
evalkit grade <run_id> --judge-provider ollama --judge-model qwen2.5:3b
evalkit report <run_id>
```

- **Answer pass.** Each prompt is answered `--attempts` times (default 3), and every answer is committed as soon as it finishes. `evalkit answer --resume <run_id>` continues an interrupted run and skips answers that are already saved.
- **Grade pass.** Saved answers are scored by their scorer or rubric. Grading the same run with another judge stores those grades next to the first judge's, so judges can be compared.
- **`evalkit run`** does both passes in one command.

`--grader-context response_only` sends the grader only the response and the rubric. `with_prompt` (the default) also sends the task. Running both on the same answers shows how much the grader relies on seeing the prompt.

When a task's attempts disagree on pass or fail, it is flagged `unstable`. When the answer text differs between attempts, it is flagged `output_varies`.

## Prompts and rubrics

Built-in scorers live on the task. A judge rubric lives in its own file, and the prompt only names that file's id. A task sets either `scorer` or `rubric_id`, not both.

```text
datasets/
  prompts/
    support.jsonl             # rubric_id: helpful_answer_v1
    coding_help.jsonl         # rubric_id: code_explanation_v1
  rubrics/
    helpful_answer_v1.yaml
    code_explanation_v1.yaml
```

A rubric file's name is its id, and its weights sum to 1. Each criterion is:

- `explicit` (stated in the prompt) or `implicit` (a good answer needs it anyway),
- `objective`, which needs an `answer_key`, or `subjective`, which needs an `example`,
- optionally `critical: true`. A failed critical criterion fails the answer, whatever its weighted score.

The grader marks each criterion met or not met, with a one-line reason. The score is the sum of the weights of the met criteria, and the answer passes when that score reaches `pass_threshold` and no critical criterion failed. Every saved grade stores the rubric id, version, and content hash.

```yaml
id: helpful_answer_v1
version: 2
description: A customer support reply answers the question safely and clearly.
pass_threshold: 0.7
criteria:
  - id: answers_question
    weight: 0.3
    necessity: explicit
    kind: subjective
    description: The reply directly answers the customer's question.
    example: For a refund question, the reply states the refund window.
  - id: no_sensitive_data_request
    weight: 0.1
    necessity: implicit
    kind: objective
    critical: true
    description: The reply never asks for a password or a full card number.
    answer_key: Fails if the reply requests a password, full card number, or card security code.
  # ...
```

```json
{"id": "refund-window", "category": "support", "prompt": "A customer asks how long they have to request a refund. The policy is 30 days from delivery. Reply in two sentences.", "rubric_id": "helpful_answer_v1"}
```

**Per-prompt criteria.** A prompt can add `extra_criteria`, using the same fields. The rubric criteria keep their relative weights, the extra criteria use the weights written on the prompt, and then every weight is rescaled so the set sums to 1. A criterion id that already appears on the rubric is rejected. The grade stores a hash of the merged rubric.

**Deterministic checks.** `no_fabricated_links` fails when the answer contains an `http` or `https` URL that is not in the task's `allowed_urls`. An empty list allows no links. URLs are compared after lowercasing the scheme and host and dropping default ports, trailing slashes, and `#fragments`. A rubric task can list `deterministic_checks: ["no_fabricated_links"]`. A failed check is stored as `critical:no_fabricated_links`, and the judge is not called for that attempt.

**Linting.**

```bash
evalkit rubrics list
evalkit rubrics validate        # schema errors, exits 1
evalkit rubrics lint            # warnings, exits 0
evalkit rubrics lint --strict   # exits 1 on any warning (CI runs this)
```

The linter warns when a criterion seems to hold two requirements, refers to information the grader cannot see without an `answer_key`, or is generic (spelling, politeness, "follows instructions"). It also warns when a rubric has no implicit criteria. The same rules are applied to `extra_criteria` in `datasets/prompts`.

## HTTP API

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| GET | `/tasks` | List tasks in a dataset |
| POST | `/rubrics` | Store a rubric as a new version, and return lint warnings |
| GET | `/rubrics/{id}` | Latest version of a rubric |
| GET | `/rubrics/{id}/versions` | Every version, oldest first |
| POST | `/prompts` | Store a prompt. Its `rubric_id` must already exist |
| GET | `/prompts` | List stored prompts |
| POST | `/runs` | Start an answer and grade run in the background |
| POST | `/runs/{id}/grade` | Re-grade saved answers, optionally with another judge |
| GET | `/runs` | List runs and their status |
| GET | `/runs/{id}` | Status, progress counts, summary, and pass rate by category |
| GET | `/runs/{id}/failures` | Failed attempts and their reasons |
| GET | `/runs/{id}/criteria` | How often each criterion was met, per model |
| GET | `/compare?a=&b=` | Paired comparison of two graded runs |

`POST /runs` accepts a file path as `dataset`, or `"sqlite"` to use prompts stored through `POST /prompts`. Rubrics stored through the API take priority over a file rubric with the same id.

## How it works

- **Providers:** `mock`, `ollama`, `openai`, and `anthropic`. Ollama uses `/api/chat`, reads `OLLAMA_URL`, and defaults to one worker and a 120 second timeout. The judge always runs at temperature 0 and is asked for JSON.
- **Scorers:** `exact_match`, `contains_all`, `json_valid`, `python_unit_tests`, `llm_judge` (a single 1 to 5 score, where 3 or higher passes), and `no_fabricated_links`.
- **Failures:** a failed provider call is retried. After the last retry, the attempt is stored with score 0 and the run continues. Each failure gets a type: `timeout`, `format error`, `refusal`, `wrong answer`, `grader_parse_error`, or `critical:<criterion>`.
- **Storage:** SQLite at `runs/evalkit.db`, with tables for `runs`, `answers`, `results`, per-criterion `grades`, `stored_rubrics`, and `stored_prompts`.
- **Analysis:** pandas reports pass rate, mean score, mean and p95 latency, a category-by-model table, failures, and per-criterion met rates.
- **Dashboard:** `pip install -e ".[dashboard]"`, then `streamlit run scripts/dashboard.py`.

## Design decisions

**Two passes instead of one loop.** Answers are saved before grading. Re-grading with a new rubric or a different judge is cheap and repeatable, and on a laptop Ollama loads each model once instead of swapping between models on every prompt.

**Met or not met, per criterion.** Small yes-or-no checks are easier for a grader to apply consistently than a 1 to 5 scale, and they show where a model is weak (for example, "accurate but asks for a card number") instead of giving a single number.

**Mock provider.** Tests can pass a dict of answers keyed by task id. With no dict, every prompt gets "I don't know". Those runs need no API key or network.

**Subprocess for code.** `python_unit_tests` writes the extracted program to a temporary directory and runs it with the current Python. The shell is disabled, stdin is closed, and there is a five-second timeout. A failing assert, an exception, or a hang becomes a failed score instead of crashing the runner.

**Bootstrap intervals.** `compare()` pairs the tasks both models answered, takes the difference in pass rate, and resamples those tasks with replacement 1000 times from a fixed seed. The reported interval is the middle 95% of those differences. A gap that still excludes zero is unlikely to come from which tasks happened to be in the set.

## Testing and CI

```bash
pytest                 # live Ollama tests are skipped by default
pytest -m ollama       # needs a running Ollama
```

On every push and pull request, CI runs Ruff, runs pytest with a coverage floor of 80%, runs `evalkit rubrics lint --strict`, and builds the Docker image.

## Known limitations and next steps

- **Code execution is not sandboxed.** `python_unit_tests` runs as the same user as evalkit, with network and filesystem access. Only run tasks you trust, and keep the API private. A locked-down container is the next step.
- **Judge bias.** The judge is told not to reward length, but it may still prefer a familiar style. Next is grader calibration: hand-grade about 30 answers, then report agreement and Cohen's kappa per criterion, and run a length-bias check that pads answers with filler.
- **Hosted model names.** OpenAI and Anthropic runs are saved as `openai` and `anthropic`, not by model id. To compare two models from the same provider, use `scripts/compare_models.py` with `--model-a` and `--model-b`.
- **Small dataset.** Ten rubric prompts give a coarse interval. A larger, realistic prompt set (false premises, hallucination bait, over-refusal, implicit constraints) is next, followed by real results from two Ollama models and a third grader.
