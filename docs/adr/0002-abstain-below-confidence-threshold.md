# Abstain (empty answer) when confidence is below a threshold

Supersedes [ADR 0001](0001-always-return-best-candidate.md).

The published League scoring (Match 3 brief) treats an empty answer as correct on an unanswerable
question and states that guessing is scored as wrong. ADR 0001 assumed empty and wrong answers
both score zero; that assumption no longer holds, so a low-confidence guess now costs points on
every unanswerable question it is asked.

The agent therefore returns `""` (and no citations) whenever its chosen candidate's confidence is
below `M4_MIN_CONFIDENCE` (default 0.5). Values the LLM produced that do not appear verbatim in
any passage drop below that threshold, and when the LLM reads the passages and explicitly says
the answer is not there, deterministic fallback answers lose 0.25 confidence. Setting
`M4_MIN_CONFIDENCE=0` restores the ADR 0001 behaviour should the Match 4 brief reward any
non-empty answer. Re-tune the threshold on the eval set (answerable + unanswerable cases).
