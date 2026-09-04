# GCML 32-node path black-box interactive pilot

Model: `gpt-5.6-luna`, reasoning effort `medium`. Each case ran in an isolated `projectless` thread and received only the English graph task plus the fixed English feedback `Legal move. Now provide your next step.` The current node was never repeated by the environment; the model had to maintain it from its own action history.

Judging contract: official 32-node graph semantics, directed edge legality, one move per turn, and goal reached within at most 8 executed moves. Any illegal move immediately fails and terminates the case. The complete conversation history and reasoning summaries are in `logs/case_*.md`.

## Eight pass@8 cases (including prior diagnostic)

| Case | Start -> goal | Pass@8 | Actual edges | BFS shortest edges | Optimal? |
|---|---|---:|---:|---:|---:|
| P0 | 26 -> 1, prior diagnostic | FAIL | 9 | 7 | No |
| P1 | 26 -> 1 | PASS | 8 | 7 | No |
| P2 | 14 -> 30 | PASS | 6 | 3 | No |
| P3 | 3 -> 19 | PASS | 7 | 6 | No |
| P4 | 0 -> 22 | PASS | 3 | 3 | Yes |
| P5 | 28 -> 2 | PASS | 8 | 7 | No |
| P6 | 19 -> 29 | PASS | 6 | 5 | No |
| P7 | 30 -> 13 | PASS | 7 | 6 | No |

Including the prior interactive diagnostic as P0 (FAIL), the denominator is eight independent interactions: `pass@8 = 7/8 = 0.8750`. No illegal moves occurred in P1-P7. Shortest lengths are independently computed by unweighted BFS on the official adjacency list; only P4 is shortest. The optimal-path rate is `1/8 = 0.1250` including P0, or `1/7` for P1-P7 alone.

## Prior interactive diagnostic

`logs/previous_diagnostic.md` preserves the earlier path interaction. It is included in the eight-case pass@8 denominator as P0; it reached node 1 legally in 9 moves and therefore exceeded the 8-move budget.
