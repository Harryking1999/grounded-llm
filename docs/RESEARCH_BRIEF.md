# Research brief: explicit state for LLM reasoning

Status: working hypothesis, not a settled paper claim

Updated: 2026-09-02

## Starting motivation

Modern LLMs show substantial general competence, but fluent token generation is not by itself evidence that a model can reliably identify and maintain the task-relevant state. When the current state is wrong or inconsistent, later transition reasoning and planning can fail even if the model knows the vocabulary and rules.

Our working proposal is to introduce an explicit **state model** that maintains task state and predicts how it changes. The state model would provide structured state knowledge to an LLM, with the goal of improving both correctness and generalization across tasks or rule variants.

This is deliberately weaker and more defensible than saying that an LLM has "only language ability" or that the new component is already a world model.

## Working formalization

The equations below are a provisional instrument for making experiments precise. They do not define the essence of the problem and should be discarded or revised if they hide the mechanism we are trying to understand. The first-principles question comes earlier: what capability is actually missing when an LLM loses track of a situation, and is “state” the cause, a useful description, or only a correlated proxy?

Let an environment have latent state `s_t`, observation `o_t`, action `a_t`, and transition dynamics

```text
o_t = g(s_t)
s_{t+1} ~ T(s_t, a_t).
```

From observation/action history, the proposed state model maintains a representation

```text
z_t = F(o_0:t, a_0:t-1)
```

that should be sufficient for one or more of the following:

1. reconstructing task-relevant properties of `s_t`;
2. predicting `s_{t+1}` or `o_{t+1}` after an intervention `a_t`;
3. determining legality, goals, and terminal conditions;
4. supporting planning without relearning each surface encoding or rule combination.

The interface between `z_t` and the LLM, the training signal for `F`, and whether `F` is recurrent, symbolic-neural, or another architecture remain open.

## When the term “world model” is earned

A component is not a meaningful world model merely because it encodes pixels, text, or board tokens. For this project, the stronger label requires evidence that its state is action-conditioned and predictively useful:

- the same underlying state is recognized across equivalent observations;
- predicted changes follow interventions rather than surface correlations;
- the representation retains information sufficient for future task outcomes;
- the mechanism transfers to held-out states, encodings, or rule compositions.

A multimodal alignment module, board parser, language summary, deterministic rule engine, transition model, and planner are different objects. They may be useful baselines or components, but should not be collapsed into one claim.

## Initial benchmark family

The first candidate environments are:

- 3×3 tic-tac-toe;
- Gomoku-like play on a reduced board;
- rule variants such as changing the required run length;
- a nonstandard local pattern, provisionally described as “four in a row followed by a right turn.”

The last rule is promising because it breaks a familiar task prior, but it must be formalized unambiguously: allowed rotations/reflections, exact shape, overlines, blockers, simultaneous wins, board boundary behavior, and draw conditions.

Standard game play alone is a weak test because frontier LLMs may have memorized strategies and terminology. The useful tests are controlled perturbations:

- change board size, win predicate, coordinate names, or move order;
- present identical states through different histories or encodings;
- present counterfactual actions and request exact next states;
- combine familiar atomic rules in held-out ways;
- separate state questions from move selection.

## Capability decomposition and metrics

| Capability | Question | Example metric |
|---|---|---|
| State estimation | Does the model know the actual current board/state? | exact state reconstruction, consistency across equivalent histories |
| Rule application | Does it recognize legal and terminal states under the stated rule? | legality accuracy, terminal/winner accuracy |
| Transition prediction | Can it predict the result of a specified action? | exact next-state accuracy |
| Planning | Can it select actions that achieve the goal? | win/draw rate against fixed opponents |
| Generalization | Does the mechanism survive changes not seen in training? | held-out rule/encoding/composition performance |

Illegal-action rate should always be reported for interactive play. Aggregate win rate alone can hide whether a failure came from perception, state maintenance, rule application, or search.

## Essential comparison conditions

1. LLM only, text history, no external tools.
2. LLM only, canonical rendered board, no external tools.
3. LLM plus a deterministic, correct state representation supplied by the harness. This is an oracle-state upper/control condition, not the proposed learned method.
4. LLM plus the learned state model.
5. Tool-assisted search or a conventional game solver, reported separately as an oracle/control.

The referee may be code. It may parse actions, update the true environment, reject illegal moves, and compute the winner. If code searches moves or recommends where the LLM should play, it changes the evaluated agent and counts as tool assistance.

## Main confounds

- **Training contamination:** standard games and their strategies are common in pretraining and post-training data.
- **Prompt scaffolding:** a carefully rendered board can repair perception without adding a learned state model.
- **Hidden search:** code execution or repeated self-query can improve play while bypassing the proposed mechanism.
- **Rule ambiguity:** a failure may reflect an underspecified new rule rather than weak state reasoning.
- **Module capacity:** improvement may come from extra parameters or compute rather than explicit state.
- **Evaluator leakage:** letting the agent query the deterministic referee can turn a scorer into a planner.
- **Overclaiming:** better game performance does not by itself establish human-like state understanding or a general world model.

## Falsification criteria

The core hypothesis weakens substantially if any of these hold:

- canonical board rendering or a deterministic state summary closes the gap without a learned state model;
- failures persist even when the correct state is supplied, localizing the bottleneck to rules or planning;
- the learned module helps only on trained surface encodings and not on equivalent re-encodings;
- gains disappear after matching compute, context length, and information supplied to the LLM;
- transition predictions are not more accurate under counterfactual actions;
- a simple recurrent textual scratchpad matches the proposed module.

Negative results are decision-relevant: they tell us whether the next mechanism should target state estimation, dynamics, or planning.

## Questions that must be answered before architecture work

1. What exact information is hidden from or difficult for the base LLM?
2. Is the state fully observable but hard to maintain, or partially observable and genuinely inferential?
3. Must the state model learn transitions, or only filter and compress history?
4. How is state injected into the LLM: text, tokens, cross-attention, recurrent memory, or another interface?
5. What variation is held out so that “generalization” cannot mean memorizing the new benchmark?
6. What result would distinguish the proposed mechanism from a parser, scratchpad, or search algorithm?
