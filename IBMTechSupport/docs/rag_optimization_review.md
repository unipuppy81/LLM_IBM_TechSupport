# RAG Optimization - 5 Experiments

Dataset: 135 questions (90 answerable, 45 impossible)

## Baseline
- Token F1: 16.52%
- Retrieval Hit@5: 66.67% (60/90)
- False refusals: 45/90
- Correct refusals: 29/45

## Experiment 1: Judge ON / OFF
- ON F1: 16.52%
- OFF F1: 28.79%
- OFF correct refusals: 1/45
- Decision: Keep Judge

## Experiment 2: Judge Prompt and Model
- Balanced: 67 false refusals
- Strict: 55 false refusals
- 3B Strict: F1 14.11%, 50 false refusals
- Decision: Keep 1.5B Baseline

## Experiment 3: Neighbor Chunk Expansion
- Radius 0: F1 16.52%
- Radius 1: F1 15.40%
- Radius 2: F1 14.52%
- Decision: Keep Radius 0

## Experiment 4: Title-based Contextual Retrieval
- Baseline Hit@5: 66.67%
- Title BM25: 66.67%
- Title BM25 + Dense: 62.22%
- Decision: Keep existing retrieval

## Experiment 5: Generation Prompts
- Baseline F1: 16.52%
- Concise F1: 16.16%
- Evidence Check F1: 16.16%
- Decision: Keep existing generation prompt

## Conclusion
No tested configuration improved overall baseline performance.

Next steps:
1. Audit retrieval failures (30 questions)
2. Validate evidence sufficiency in retrieved chunks
3. Analyze Judge refusals and false positives
4. Compare Judge P(YES) with Reranker scores
5. Fine-tune Reranker and evaluate extractive answers
