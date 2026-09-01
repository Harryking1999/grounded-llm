# Project status and ordered TODO

Updated: 2026-09-02

This is the only current status page. It records stable decisions and ordered work, not transient process IDs, ports, scratch observations, or raw logs.

## Current objective

Determine whether an explicit learned state model can improve an LLM's state estimation, action-conditioned prediction, and planning, especially under unfamiliar rules or representations.

The immediate objective is not to train a model. It is to make the claim and benchmark discriminative enough that a positive result cannot be explained by ordinary game memorization, clearer prompting, deterministic parsing, or tool-assisted search.

## Established project decisions

- This directory is an independent Git repository rather than a subproject in the OPD history.
- The prior Feishu document is treated as motivation around cognitive maps, predictive representations, and planning; it does not establish novelty for this project.
- “State model” is the neutral component name until its architecture and evidence justify a stronger label.
- Standard tic-tac-toe and Gomoku are calibration tasks, not sufficient novelty evidence.
- The primary LLM condition uses no external solver or code for move selection.
- Deterministic code may maintain the true environment and score legal moves/wins as a referee.
- State estimation, transition prediction, rule application, and planning will be measured separately.
- First-principles inquiry means pursuing the essential causal account of the failure. Statistical regularities and benchmark patterns are evidence, not substitutes for mechanism.
- Harness complexity and validation must remain proportional to the question being tested.

## Current evidence boundary

- We have an initial qualitative observation that LLM game reasoning can look strong on familiar five-in-a-row terminology and play.
- We do not yet have a controlled baseline, model/version record, fixed prompts, randomized state set, or quantitative result.
- We therefore cannot yet conclude that LLMs lack a state representation, that Gomoku is difficult, or that a learned auxiliary network will improve performance.
- The modified-pattern idea is not evaluable until its rule is written as a deterministic predicate.

## Ordered TODO

1. Identify the phenomenon at its causal core: what exactly fails when the model loses the situation, what rival mechanisms could produce the same behavior, and whether “missing state” is a cause or only a description. Introduce state/action formalism only where it helps distinguish these explanations.
2. Formalize a small family of rule predicates, including the proposed “four then right turn” pattern, with unambiguous examples and boundary cases.
3. Build the minimal deterministic environment/referee and tests. Do not implement a solver in the primary agent path.
4. Create a frozen baseline set spanning valid/invalid states, equivalent histories, next-state queries, terminal judgment, and move choice; include standard and held-out rule variants.
5. Evaluate one or more LLM baselines without tools and classify failures by stage rather than relying on win rate alone.
6. Use the failure decomposition to choose the smallest mechanism: better state presentation, recurrent scratchpad, learned state estimator, learned transition model, or planner.
7. Only then implement the learned state model and compare it against matched deterministic-state, extra-context, extra-compute, and tool-assisted controls.
8. In parallel with steps 1–5, map the closest prior work by concrete mechanism and evaluation: cognitive maps, latent/world models, recurrent memory/state tracking, neural algorithmic reasoning, model-based RL, and LLM planning.
9. Promote a result to a paper claim only after held-out rule or representation transfer, matched controls, and at least one non-game task support the same mechanism.

## Next discussion checkpoint

Before code is added, settle the answers to these three questions:

1. Is the proposed module expected to infer the current state, learn the transition dynamics, or both?
2. Does the LLM receive the module's state as text/tokens, or are we considering an internal neural interface?
3. Is the first paper claim about better diagnosis of LLM state failures, a new state-model architecture, or a generalization result enabled by that architecture?
