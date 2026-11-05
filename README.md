# Formula-1-Tyre-Strategy
S403011 Machine Learning Project: Formula 1 Tyre Strategy

The data comes from the Kaggle competition
[s-403011-ml-project-formula-1-tyre-strategy](https://www.kaggle.com/competitions/s-403011-ml-project-formula-1-tyre-strategy).
It is **not** stored in this repo: `data.py` downloads it from Kaggle with your own API token
and caches it on your machine.

## Setup

### 1. Install uv

We use [uv](https://docs.astral.sh/uv/) to manage Python and the packages, so everyone
gets exactly the same versions (they are pinned in `uv.lock`).

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

(Or `brew install uv` / `pip install uv`.)

### 2. Install the dependencies

From the repo folder:

```bash
uv sync
```

This creates a `.venv/` folder with the right Python packages. You don't need to activate it:
prefix commands with `uv run` (e.g. `uv run python data.py`). If you prefer, activate it
with `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows), and
select `.venv` as the interpreter in VS Code / Jupyter.

To add a new package, use `uv add <package>` (not `pip install`) so it ends up in
`pyproject.toml` and `uv.lock` for everyone. Commit both files.

### 3. Get access to the Kaggle data

1. Create a [Kaggle](https://www.kaggle.com) account (or log in).
2. Open the [competition page](https://www.kaggle.com/competitions/s-403011-ml-project-formula-1-tyre-strategy)
   and **join the competition / accept the rules**. Without this the download fails with a
   403 / permission error.
3. Go to **Settings → API** ([kaggle.com/settings](https://www.kaggle.com/settings)) and click
   **Generate New Token**. Copy the token.

### 4. Add your API token

Copy the template and paste your token in it:

```bash
cp .env.example .env
```

`.env` should look like:

```
KAGGLE_API_TOKEN=your_kaggle_api_token_here
```

`.env` is in `.gitignore`. **Never commit it** — the token is personal.

### 5. Check that it works

```bash
uv run python data.py
```

The first run downloads the files (cached afterwards, under `~/.cache/kagglehub/`) and prints
the shape of each dataset.

## Loading the data in your code

```python
from data import load_data

train, test, sample_submission = load_data()
```

Or load a single dataset:

```python
from data import load_train, load_test, load_sample_submission

train = load_train()
```

| Function | File | Returns |
| --- | --- | --- |
| `load_train()` | `train_laps.parquet` | training laps `DataFrame` |
| `load_test()` | `test_laps.parquet` | test laps `DataFrame` |
| `load_sample_submission()` | `sample_submission.csv` | submission format `DataFrame` |
| `get_data_dir()` | — | `Path` of the folder with the raw files |

## Troubleshooting

- **401 / unauthenticated**: `.env` is missing, is not in the repo root, or the token is wrong.
  Generate a new token and try again.
- **403 / forbidden**: you haven't joined the competition / accepted the rules on Kaggle.
- **`ModuleNotFoundError`**: run `uv sync`, and make sure you run with `uv run` or with
  `.venv` activated / selected as the interpreter.
- **Not using uv?** In a virtual environment, run
  `pip install kagglehub pandas pyarrow python-dotenv` (the list in `pyproject.toml`).
  The versions won't be pinned like they are with `uv.lock`.
