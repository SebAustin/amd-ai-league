# Always return the best candidate answer instead of abstaining

The Match 4 agent returns its best candidate Answer whenever it has read at least one page, even
at low confidence, and reports that confidence alongside it. Leading competing entries do the
opposite ("proof or silence": return an empty answer unless verified). Under exact-match grading an
empty answer and a wrong answer both score zero, so abstaining can never gain points and often
loses them. Revisit if the Match 4 harness turns out to penalise wrong answers more than empty ones,
or to reward calibrated refusals — the `confidence` field already allows thresholding later.
