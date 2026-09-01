# State-aware LLM reasoning

This is a new research workspace for testing a narrow question:

> Can an explicit, learned state model improve an LLM's ability to maintain the current state, predict action-conditioned changes, and plan under unfamiliar task rules?

The wording is intentionally provisional. We have not yet established that current LLM failures come from a missing state representation, that the required component should be a small neural network, or that the resulting system warrants the term *world model*.

## Current direction

The initial testbed will use small, exactly simulatable tasks such as tic-tac-toe, reduced-board Gomoku, and controlled rule variants. These environments let us separate:

- state reconstruction from a history or observation;
- legal and terminal-state judgment;
- next-state prediction under an action;
- planning and action selection;
- memorized competence on standard rules from generalization to new rules.

The primary comparison will keep the evaluated LLM tool-free. A deterministic program may act as the referee and scorer; programmatic search used to choose the model's move is reported separately as a tool-assisted control.

## Repository map

- `AGENTS.md`: collaboration, reasoning, experiment, and Git rules
- `docs/RESEARCH_BRIEF.md`: working definitions, hypotheses, confounds, and falsification criteria
- `docs/PROJECT_STATUS_AND_TODO.md`: the only current status page and ordered next actions
- `experiments/`: durable studies once their contracts are concrete

Large datasets, logs, checkpoints, and generated runs are intentionally excluded from Git.
