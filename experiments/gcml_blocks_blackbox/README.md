# GCML Blocks black-box interactive pilot

Model: `gpt-5.6-luna`, reasoning effort `medium`. Each case ran in an isolated `projectless` thread and received only the English task statement plus English environment observations. The model had no repository or research-context access.

Judging contract: official GCML 10x10 tiling semantics, reusable shapes, zero-based anchors, one action per turn, and an all-zero grid within at most 8 executed actions. Any illegal action immediately fails and terminates the case. The complete conversation history and reasoning summaries are in `logs/case_*.md`.

## Eight pass@8 cases (including prior diagnostic)

| Case | Source sample | Result | Steps / failure |
|---|---|---:|---|
| B0 | prior official row 18002 diagnostic | FAIL | illegal `(4,5,6)` after 7 legal actions |
| B1 | official row 18000 | PASS | 8 actions |
| B2 | official row 18001 | FAIL | illegal action `(1,5,3)` at step 1 |
| B3 | official row 18002 | PASS | 8 actions |
| B4 | official row 18000 | PASS | 8 actions |
| B5 | official row 18001 | FAIL | illegal action `(2,8,8)` at step 6 |
| B6 | official row 18002 | FAIL | illegal action `(0,4,2)` at step 4 |
| B7 | official row 18002 | PASS | 8 actions |

Including the prior interactive diagnostic as B0 (FAIL), the denominator is eight independent interactions: `pass@8 = 4/8 = 0.5000`.

## Prior interactive diagnostic

`logs/previous_diagnostic.md` preserves the earlier Blocks interaction. It is included in the eight-case pass@8 denominator as B0: after seven legal actions the model proposed illegal `(4,5,6)`, so it is an immediate failure under the project rule.
