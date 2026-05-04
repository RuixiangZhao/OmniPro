"""OmniProact-Bench metrics package.

Subpackages:
    probe/   — probe-mode metrics (temporal, content, aggregator)
    online/  — online-mode metrics (scorer)

Top-level:
    llm_judge.LLMJudge — unified LLM-as-a-judge for free-text content

Backwards-compat re-exports are provided in the modules that were moved;
new code should import from the new locations.
"""
