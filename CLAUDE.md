# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM evaluation pipeline implementing Eugene Yan's "Product Evals in Three Simple Steps" methodology.

**Inspiration:** https://eugeneyan.com/writing/product-evals/

**Repository:** https://github.com/nibzard/evals

### Core Principles (Eugene Yan)
- Binary pass/fail labels (not numeric scales)
- Aim for 50-100 failures in 200+ samples for meaningful signal
- One evaluator per dimension, not a "God Evaluator"
- Use smaller models to generate organic failures
- Target Cohen's Kappa 0.4-0.6 (substantial agreement)
- Prioritize recall on failure detection (catching defects is critical)

### Key Results
- GPT-5-mini vs GPT-5.1: **Cohen's Kappa 0.615** (substantial agreement)
- Fail Recall: **82.4%** (good at catching failures)
- Human inter-rater reliability is often only kappa 0.2-0.3

## Commands

**Always use Astral UV, never venv directly.**

```bash
# Evaluation demos (three providers compared)
uv run eval_demo_gpt5.py        # GPT-5.1 + Responses API + function calling (recommended)
uv run eval_demo_gpt4.py        # GPT-4o-mini + Chat Completions
uv run eval_demo_gemini.py      # Gemini 2.0 Flash Lite (may hit rate limits)

# Verify evaluator against ground truth labels
uv run verify_evaluator.py                              # Full 200 samples
uv run verify_evaluator.py --sample 50                  # Quick test with 50 samples
uv run verify_evaluator.py --model gpt-5-nano-2025-08-07  # Test different model
uv run verify_evaluator.py --category edge_cases        # Filter by category
uv run verify_evaluator.py --output results/custom.jsonl  # Custom output file

# Visualize evaluation runs
uv run plot_runs.py                        # Show plots interactively
uv run plot_runs.py --output results/plots # Save to directory
uv run plot_runs.py --last 5               # Only last 5 runs

# Data generation pipeline
uv run scripts/generate_hard_questions.py     # Generate questions + answers (GPT-5.1 + GPT-3.5)
uv run scripts/label_responses.py             # Label responses (GPT-5.1 function calling)
uv run scripts/generate_answers.py            # Generate answers (Gemini)

# Dependencies
uv add package-name
```

## File Structure

```
├── eval_demo_gemini.py       # Gemini provider demo
├── eval_demo_gpt4.py         # GPT-4o-mini provider demo
├── eval_demo_gpt5.py         # GPT-5.1 provider demo (best practices)
├── verify_evaluator.py       # Verify smaller model vs ground truth
├── plot_runs.py              # Visualize run comparisons (matplotlib)
├── scripts/
│   ├── generate_answers.py   # Gemini answer generation
│   ├── generate_hard_questions.py  # Question generation (GPT-5.1)
│   └── label_responses.py    # Labeling with function calling (GPT-5.1)
├── data/
│   └── questions_version_2.csv  # v2: 35.5% fail rate (good signal)
├── results/
│   └── runs.jsonl            # Evaluation run logs (gitignored)
├── EVAL_PROVIDERS.md         # Detailed provider comparison
├── LESSONS_LEARNED.md        # API patterns and gotchas
└── README.md                 # User documentation
```

## Datasets

| Dataset | Questions | Failures | Fail Rate | Answer Model |
|---------|-----------|----------|-----------|--------------|
| `data/questions.csv` | 200 | 7 | 3.5% | Gemini 2.0 Flash Lite |
| `data/questions_version_2.csv` | 200 | 71 | 35.5% | GPT-3.5-turbo-0125 |

### v2 Question Categories

| Category | Fail Rate | Description |
|----------|-----------|-------------|
| precise_calculations | 60% | Math, code output, conversions |
| temporal_changing | 60% | Current events, changing facts |
| multi_part | 37% | Questions requiring multiple pieces |
| ambiguous_debatable | 26% | Contested facts, multiple valid answers |
| edge_cases | 23% | Nuanced topics, exceptions |
| trick_misconceptions | 14% | Common myths |

## Models & APIs

| Provider | Model | API | Use Case |
|----------|-------|-----|----------|
| OpenAI | gpt-5.1-2025-11-13 | Responses API | Labeling, question generation |
| OpenAI | gpt-5-mini-2025-08-07 | Responses API | Evaluator (good balance) |
| OpenAI | gpt-5-nano-2025-08-07 | Responses API | Evaluator (cheapest) |
| OpenAI | gpt-4o-mini | Chat Completions | Demo comparison |
| OpenAI | gpt-3.5-turbo-0125 | Chat Completions | Weak answer model |
| Google | gemini-2.0-flash-lite | generate_content | Answer generation |

## API Patterns

### GPT-5.1 Responses API (Recommended)

```python
from openai import OpenAI
client = OpenAI()

# Key differences from Chat Completions:
result = client.responses.create(
    model="gpt-5.1-2025-11-13",
    input=[...],                    # NOT "messages"
    max_completion_tokens=100,      # NOT "max_tokens"
    tools=[EVAL_TOOL],
    tool_choice={"type": "function", "name": "submit_evaluation"},
)

# Output access
for item in result.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
```

### Function Calling with Strict Schema

```python
EVAL_TOOL = {
    "type": "function",
    "name": "submit_evaluation",
    "strict": True,  # Enforces schema compliance
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],  # Only these values allowed
            }
        },
        "required": ["verdict"],
        "additionalProperties": False  # Required for strict mode
    }
}
```

### Chat Completions (GPT-4o-mini, GPT-3.5)

```python
result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100,
)
text = result.choices[0].message.content
```

## Key Technical Insights

### 1. Function Calling > Text Parsing
Text parsing is fragile ("Pass", "PASS", "I think it passes"). Use `strict: true` + `enum` for guaranteed values.

### 2. Position Bias Detection
Run comparisons twice with swapped order:
- Same answer both times → genuine preference
- Different answers → position bias (return "tie")

### 3. Rate Limits
- Gemini free tier: RPD (requests per day) is the bottleneck, not RPM
- Use exponential backoff on 429 errors

### 4. Evaluation Metrics
- **Cohen's Kappa**: 0.4-0.6 = substantial, >0.6 = excellent
- **Fail Recall**: Priority metric (catching defects)
- **Confidence Intervals**: `accuracy ± 1.96 * sqrt(acc * (1-acc) / n)`

## Environment

Required in `.env`:
```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
```

## Documentation

- `EVAL_PROVIDERS.md` - Detailed comparison of Gemini vs GPT-4o-mini vs GPT-5.1
- `LESSONS_LEARNED.md` - API patterns, rate limits, parameter differences
- `README.md` - User-facing documentation with quick start

## Common Tasks

### Adding a new evaluator dimension
1. Create function in `eval_demo_*.py` following `evaluate_helpfulness()` pattern
2. Use function calling with strict schema for reliable output
3. Add to `run_eval_parallel()` call

### Testing evaluator reliability
```bash
uv run verify_evaluator.py --sample 50
```
Look for:
- Cohen's Kappa >= 0.4 (substantial agreement)
- Fail Recall >= 80% (catching defects)

### Generating more failures
Use weaker answer model (GPT-3.5) or harder question categories (precise_calculations, temporal_changing).
