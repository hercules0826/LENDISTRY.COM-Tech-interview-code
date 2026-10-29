# LENDISTRY Technical Exercises

This repository contains two independent Python exercises. Each lives in its own
directory, was developed on its own branch, and is submitted as its own pull request.

| Exercise | Directory | Summary |
| --- | --- | --- |
| 1 | `exercise-1-vector-classification/` | Decide whether a query vector belongs to Group A or Group B, in pure Python with no ML or similarity libraries. |
| 2 | `exercise-2-ai-assistant/` | A terminal chat assistant that lets an LLM decide when to call a `calculate_total` tool. Built without agent frameworks. |

Each exercise directory has its own `README.md` covering the approach, assumptions,
and how to run it.

## Python setup

You need **Python 3.10 or newer**. The exercises were developed on Python 3.11.

```bash
# 1. Create and activate a virtual environment (from the repository root)
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 2. Install the dependencies for the exercise you want to run
pip install -r exercise-1-vector-classification/requirements.txt
pip install -r exercise-2-ai-assistant/requirements.txt

# 3. Run the tests
python -m pytest exercise-1-vector-classification
python -m pytest exercise-2-ai-assistant
```

## Credentials

Exercise 2 calls an LLM API. It reads the API key from an environment variable,
so no credential is ever stored in the repository. `.env` files are git-ignored.
See `exercise-2-ai-assistant/README.md` for details.

## Workflow

- `main` holds only this README and shared config.
- Each exercise is built on a branch created from `main`
  (`exercise-1-vector-classification`, `exercise-2-ai-assistant`).
- Each branch has one pull request, which stays open for review.
