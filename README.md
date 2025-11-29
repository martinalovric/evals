# LLM Evaluation Pipeline

A practical implementation of [Eugene Yan's "Product Evals in Three Simple Steps"](https://eugeneyan.com/writing/product-evals/) methodology for building reliable LLM evaluators.

## Key Results

| Metric | Value | Target |
|--------|-------|--------|
| **Cohen's Kappa** | 0.615 | 0.4-0.6 (substantial) |
| **Fail Recall** | 82.4% | High (catch defects) |
| **Accuracy** | 82.0% | - |

GPT-5-mini achieves **substantial agreement** with GPT-5.1 ground truth labels, exceeding typical human inter-rater reliability (kappa 0.2-0.3).

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/nibzard/evals.git
cd evals

# Add your API keys to .env
echo "GOOGLE_API_KEY=your-key" >> .env
echo "OPENAI_API_KEY=your-key" >> .env

# Run evaluation demos
uv run eval_demo_gpt5.py      # GPT-5.1 with function calling (recommended)
uv run eval_demo_gpt4.py      # GPT-4o-mini
uv run eval_demo_gemini.py    # Gemini (requires quota)

# Verify evaluator against ground truth
uv run verify_evaluator.py --sample 50
```

## Project Structure

```
├── eval_demo_gemini.py       # Gemini 2.0 Flash Lite demo
├── eval_demo_gpt4.py         # GPT-4o-mini (Chat Completions)
├── eval_demo_gpt5.py         # GPT-5.1 (Responses API + function calling)
├── verify_evaluator.py       # Verify evaluator vs ground truth labels
├── plot_runs.py              # Visualize run comparisons
├── scripts/
│   ├── generate_answers.py   # Generate answers with Gemini
│   ├── generate_hard_questions.py  # Generate challenging questions
│   └── label_responses.py    # Label with GPT-5.1 function calling
├── data/
│   └── questions_version_2.csv  # 200 questions, 35.5% fail rate
├── docs/
│   └── plots/                # Generated visualizations (tracked)
├── results/
│   └── runs.jsonl            # Evaluation run logs (gitignored)
├── EVAL_PROVIDERS.md         # Provider comparison report
├── LESSONS_LEARNED.md        # API patterns and gotchas
└── CLAUDE.md                 # Instructions for Claude Code
```

## Provider Comparison

| Provider | Model | API | Output Parsing | Position Bias |
|----------|-------|-----|----------------|---------------|
| Gemini | gemini-2.0-flash-lite | generate_content | Text matching (fragile) | Variable |
| OpenAI | gpt-4o-mini | Chat Completions | Text matching (fragile) | High ("tie") |
| OpenAI | gpt-5.1 | Responses API | Function calling (reliable) | Low (consistent) |

**Recommendation:** Use GPT-5.1 with function calling for production evaluators. The `strict: true` schema with enums guarantees valid output.

## Eugene Yan's Methodology

### 1. Label Data
- **Binary labels only** - pass/fail, not numeric scales
- **Target 50-100 failures** in 200+ samples for meaningful signal
- **Organic failures** - use weaker models, not synthetic defects

### 2. Build Evaluators
- **One evaluator per dimension** - not a "God Evaluator"
- **75/25 train/test split** - tune prompts on train, measure on test
- **Mitigate position bias** - run comparisons twice with swapped order

### 3. Measure Agreement
- **Cohen's Kappa** - accounts for chance agreement
  - 0.4-0.6 = substantial (target)
  - 0.6-0.8 = substantial to excellent
  - Human baseline often only 0.2-0.3
- **Prioritize recall on failures** - catching defects is critical

## Datasets

### v1: `data/questions.csv`
- 200 questions, 7 failures (3.5% fail rate)
- Answer model: Gemini 2.0 Flash Lite
- Too few failures for meaningful signal

### v2: `data/questions_version_2.csv`
- 200 questions, 71 failures (35.5% fail rate)
- Answer model: GPT-3.5-turbo-0125 (weaker = more failures)
- Categories designed for higher failure rates:

| Category | Fail Rate | Description |
|----------|-----------|-------------|
| precise_calculations | 60% | Math, code output, conversions |
| temporal_changing | 60% | Current events, changing facts |
| multi_part | 37% | Questions requiring multiple pieces |
| ambiguous_debatable | 26% | Contested facts |
| edge_cases | 23% | Nuanced topics, exceptions |
| trick_misconceptions | 14% | Common myths |

## Key Learnings

### 1. Function Calling > Text Parsing
```python
# Fragile - model might say "Pass", "PASS", "I think it passes"
return "pass" if "pass" in result.text.lower() else "fail"

# Reliable - enum guarantees exact values
EVAL_TOOL = {
    "strict": True,
    "parameters": {
        "properties": {
            "verdict": {"enum": ["pass", "fail"]}
        }
    }
}
```

### 2. Position Bias is Real
Run comparisons twice with swapped order. If answers differ, the model has position bias.

### 3. Smaller Models Can Evaluate
GPT-5-mini achieves kappa 0.615 against GPT-5.1 labels - substantial agreement at lower cost.

### 4. Math is Hard
`precise_calculations` category shows worst agreement (57% accuracy, kappa 0.09). Smaller models struggle with math.

## Commands Reference

```bash
# Evaluation demos
uv run eval_demo_gpt5.py              # Recommended
uv run eval_demo_gpt4.py
uv run eval_demo_gemini.py

# Verify evaluator against ground truth
uv run verify_evaluator.py                              # Full dataset
uv run verify_evaluator.py --sample 50                  # Sample 50 items
uv run verify_evaluator.py --model gpt-5-nano-2025-08-07  # Test different model
uv run verify_evaluator.py --category precise_calculations  # Filter by category
uv run verify_evaluator.py --output results/custom.jsonl    # Custom output file

# Visualize evaluation runs
uv run plot_runs.py                   # Save to docs/plots/ (default)
uv run plot_runs.py --last 5          # Only last 5 runs
uv run plot_runs.py --output custom/  # Custom output directory

# Data generation pipeline
uv run scripts/generate_hard_questions.py  # Generate questions + answers
uv run scripts/label_responses.py          # Label with GPT-5.1
```

## References

- [Eugene Yan: Product Evals in Three Simple Steps](https://eugeneyan.com/writing/product-evals/)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)

## License

MIT
