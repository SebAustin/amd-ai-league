# Glossary — AMD AI League

**League**
The AMD AI League on lablab.ai (Sep 1 – Dec 1, 2026). A single individual points leaderboard; top 3 by total Points win League prizes (requires ≥5 Academy Courses).
_Avoid_: hackathon, challenge, event

**Match**
One of six timed building competitions inside the League, each with its own task, prizes and Points value. A Match is *live* (prizes + Points) or *closed* (Points only).
_Avoid_: mini-challenge (older name used on submission pages), round

**Points**
The single score that ranks the League leaderboard; earned from Onboarding, Academy Courses, Matches, Referrals and community activity. Shown on the leaderboard as XP.
_Avoid_: score (ambiguous with Match grading)

**Onboarding**
The one-time League setup steps (AMD sign-up, GPU credits, first Course, profile, Discords) worth +300 Points.

**Course**
An AMD AI Academy course. Course counts unlock milestone Points (5/10/20/25) and ≥5 are required to claim a League prize.
_Avoid_: lesson, module

**Referral**
A real friend who joins through the participant's link; qualifies once they finish one Course.
_Avoid_: invite (the act, not the person)

**Submission**
The entry a participant files for a Match: a lablab project page plus the container Image reference.
_Avoid_: project, entry

**Image**
The Docker image a Submission points to, built on the mandated ROCm PyTorch base, with model weights baked in. It is what the Harness runs.

**Harness**
The organisers' grader that runs the Image against hidden test cases and scores its output.
_Avoid_: evaluator, judge (judges are humans)

**Request**
One Match 4 test case: a natural-language question plus a starting website (live URL or saved copy) that the agent must answer.
_Avoid_: query, task, prompt

**Answer**
The single specific value an agent returns for a Request — not a summary or list of links.

**Citation**
The page (URL, or path inside a saved copy) the Answer was taken from.
_Avoid_: source, reference
