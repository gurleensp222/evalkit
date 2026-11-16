# evalkit

Score model outputs on a JSONL task set, save the run, and compare pass rates.

[![CI](https://github.com/gurleensp222/evalkit/actions/workflows/ci.yml/badge.svg)](https://github.com/gurleensp222/evalkit/actions/workflows/ci.yml)

## Results

Local mock run of `datasets/sample_tasks.jsonl` (run `4824ed0c60ef432486202289b9900a7a`). Four of eight tasks passed. Pass rate 0.50, mean score 0.50.

| Category | Passed | Pass rate |
| --- | --- | --- |
| math | 1/1 | 1.00 |
| code | 1/2 | 0.50 |
| format | 1/2 | 0.50 |
| knowledge | 1/3 | 0.33 |

| Task | Category | Result | Reason |
| --- | --- | --- | --- |
| two-plus-two | math | pass | exact match |
| add-function | code | pass | all asserts passed |
| add-function-arguments | code | fail | assertion failed |
| name-and-city | format | pass | valid JSON with required keys |
| keys-a-and-b | format | fail | missing key `b` |
| primary-colors | knowledge | pass | all expected strings found |
| closest-planets | knowledge | fail | missing mercury, venus, earth |
| capital-of-france | knowledge | fail | expected Paris, got Lyon |

A second mock run of the same file (`0c905dadf98844598c0ad95e53fdf18f`) matched task for task. `compare()` reported a difference of 0.00 with interval [0.00, 0.00].

## Quick start

Create an empty `.env` in the project directory. Compose reads that file on startup. Put `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` there when you call a live model.

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.2
```

The API listens on http://localhost:8000. Interactive docs are at http://localhost:8000/docs, and saved runs are written to `./runs`. Compose also starts an `ollama` service and sets `OLLAMA_URL=http://ollama:11434` on the API. Model files stay in the `ollama_models` volume.

When Ollama is already running on the host, delete the compose `ollama` service and the API `depends_on` block, and point the API at the host instead:

```yaml
environment:
  OLLAMA_URL: http://host.docker.internal:11434
```

`host.docker.internal` is the machine running Docker Desktop. The API container then uses the models you already pulled on the host.

For a local mock run without Docker:

```bash
python -m pip install -e ".[dev]"
evalkit run --provider mock
evalkit report <run_id>
evalkit show <run_id> <task_id>
```

An Ollama run records the model id, not the word `ollama`:

```bash
evalkit run --provider ollama --model llama3.2 --temperature 0.2
```

`evalkit run` prints a run id, a pass count, and one line per task. CI runs this same command and fails if the sample pass rate drops below 0.50.

## How it works

Each line of a dataset is a task: an id, a category, a prompt, a scorer name, and the `expected` value that scorer checks. A provider turns the prompt into text. Built-in providers are `mock`, `ollama`, `openai`, and `anthropic`.

The runner loads the file, calls the provider from a pool of four workers, retries a failed call, and applies a timeout to each attempt. One failed task is stored with score 0 and the run continues. The named scorer maps the output to a score from 0 to 1. Scorers are `exact_match`, `contains_all`, `json_valid`, `python_unit_tests`, and `llm_judge`.

Storage writes one JSON object per line to `runs/<run_id>.jsonl`. Analysis loads those files and reports pass rate, mean score, latency, a category-by-model table, the failures, and a paired comparison of two models.

## Design decisions

**Mock provider.** Tests, CI, and the sample dataset call canned answers keyed by task id. Those runs are deterministic and do not need an API key or a network. The sample file is written so four canned answers pass and four fail.

**Subprocess for code.** `python_unit_tests` writes the extracted program to a temporary directory and runs it with the current Python, with the shell disabled, stdin closed, and a five-second timeout. A failing assert, a raised exception, or a hang becomes a failed score instead of taking down the runner.

**Bootstrap intervals.** `compare()` pairs tasks both models answered, takes the difference in pass rate, and resamples those tasks with replacement 1000 times from a fixed seed. The reported interval is the middle 95% of those differences. A gap that still excludes zero is unlikely to be an accident of which tasks were in the set.

## Known limitations and next steps

The code scorer is a separate process with a timeout. It still runs as the same user, with a normal Python install, so it is the wrong place to run untrusted model output that should be denied network or filesystem access. A container with those limits removed would be the next boundary.

`llm_judge` sends the prompt, the answer, and a rubric to a judge model and tells it not to reward length. The judge can still prefer a familiar style. A raw score of 3, 4, or 5 passes. `evalkit run` does not take a judge; use `scripts/compare_models.py --judge` for a dataset that uses that scorer.

The table above is the mock provider, not two live models. With API keys set:

```bash
python scripts/compare_models.py --a openai --b anthropic
```

Eight binary tasks make a coarse interval. Two identical runs collapse it to a point. A larger task set is what would make the interval informative. The same provider used twice also needs `--model-a` and `--model-b`, or both runs are stored under one model name and `compare()` cannot pair them.
