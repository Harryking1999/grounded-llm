# Project agent instructions

## Project scope

This repository is a new research project about whether an explicit state model can improve an LLM's state estimation, transition prediction, reasoning, and planning. It is independent from the parent OPD project. Reuse general experiment discipline from OPD, but do not import OPD-specific runs, launchers, terminology, remote state, or lifecycle gates.

## Authoritative project sources

- Current status and ordered TODO: `docs/PROJECT_STATUS_AND_TODO.md`
- Current research claim, definitions, and falsification criteria: `docs/RESEARCH_BRIEF.md`
- Experiment-specific contracts: `experiments/<study>/configs/` once a study exists

Keep exactly one current status page. Update an authoritative file instead of creating dated or version-suffixed alternatives. Do not duplicate machine-readable parameters in prose after formal configs exist.

## Collaboration and authority

- Follow the user's stated scope and priorities. Do not silently broaden the project or turn a discussion into an implementation, remote run, purchase, publication, push, or deployment.
- Proceed autonomously with read-only investigation and small, reversible local steps that directly support an authorized task. Ask before a costly experiment, a material research fork, external communication, destructive cleanup, or a change that commits the project to a substantially different claim.
- Treat user instructions as decisions, but not as evidence that a scientific premise is true. Surface contradictions, hidden assumptions, negative results, and simpler explanations directly.
- Maintain genuine interest and give a clear recommendation. Confidence must come from a concrete mechanism or result, not from confident wording.
- When the user's intent and the current evidence conflict, explain the evidence and tradeoff rather than agreeing reflexively or acting around the user.

## First-principles inquiry

“First principles” here means investigating the essence and causal structure of the problem, not beginning from a standard formalism or decomposing it into a fixed checklist. Ask what phenomenon actually needs explaining, what must be true for it to occur, and which assumptions come only from familiar terminology, benchmarks, or current methods.

Treat statistical regularities, benchmark correlations, scaling trends, and recurring empirical patterns as **second-principles evidence**. They are useful clues and constraints, but they do not by themselves explain the mechanism. Do not replace “why does this happen?” with “this pattern often appears,” and do not mistake prediction from a correlation for understanding of the underlying cause.

Maintain this angle throughout the project:

- seek the smallest causal account that explains the phenomenon;
- ask what remains invariant when wording, representation, task, or implementation changes;
- distinguish a mechanism from a proxy that merely tracks it;
- question inherited definitions such as “state,” “reasoning,” “intelligence,” and “world model” when they obscure rather than clarify;
- use existing theories and mathematical formalisms as tools, not as premises that force the answer;
- let surprising or negative experiments revise the question itself, not only the proposed solution.

Formalization should follow and sharpen the inquiry rather than substitute for it. When a state-transition description is useful, identify:

1. environment state `s_t`;
2. observation `o_t` and what information it omits;
3. action `a_t`;
4. transition rule `p(s_{t+1} | s_t, a_t)`;
5. task objective or reward;
6. what the LLM receives, stores, predicts, and controls.

Keep these capabilities separate unless evidence connects them:

- parsing/perception of an observation;
- estimating and maintaining the current state;
- predicting action-conditioned transitions;
- evaluating goals or terminal conditions;
- search/planning over future states;
- choosing and expressing an action.

Do not assume that token processing implies the absence of latent state. Do not call an observation encoder a world model merely because it compresses pixels or tokens. A world-model claim requires evidence that the learned representation supports future prediction or intervention across meaningful changes in observations, actions, or rules. These criteria are working tools for testing the idea, not the definition of first-principles thinking.

## Research loop

For each important claim:

1. state the claim in testable language;
2. describe the proposed mechanism;
3. identify the strongest trivial or memorization-based alternative;
4. find the closest prior work and inspect its concrete information flow, loss, and evaluation rather than relying on broad labels;
5. design the lowest-cost experiment that separates the explanations;
6. record observation, inference, uncertainty, and next decision separately.

Standard games may be heavily represented in training data. Prefer controlled rule changes, equivalent state re-encodings, counterfactual transitions, and held-out compositions when testing generalization. A result on ordinary tic-tac-toe or Gomoku alone is not evidence for a new state mechanism.

## Experiment harness

- Put each durable study under `experiments/<study>/` only after it has a concrete hypothesis or evaluator.
- Use `README.md` for the study's hypothesis, baselines, metrics, and current conclusion; `configs/` for formal machine-readable runs; `src/` for reusable implementation; and `tests/` for targeted correctness checks.
- Store generated runs, logs, model artifacts, and large datasets outside Git under ignored paths such as `runs/`, `logs/`, `models/`, or `data/raw/`. Commit only source, compact formal configs, and decision-relevant summaries.
- Keep the environment/referee separate from the evaluated agent. Code that validates legal moves or scores wins is allowed as harness infrastructure; code used by the agent to search or choose a move is a distinct tool-assisted condition and must not be mixed with the primary no-tool result.
- Start with the closest simple baseline. Add a learned component only after the baseline failure is localized to state estimation, transition prediction, or planning.
- Use matched prompts, states, rules, and budgets across comparisons. Report illegal-action rate, state/transition accuracy, task success, and presentation consistency when relevant; do not hide null or negative results.
- A quick smoke may be run with a command and a short note. Formal or costly runs require one committed config and one clear output directory, not a stack of generic preflight gates.

## Proportional checks

- Inspect enough to form one plausible hypothesis, make the smallest reversible change, and run the cheapest check that can falsify it.
- Every additional check must address a concrete failure risk. Do not stack syntax, manifest, hash, consistency, and lifecycle gates that prove the same thing.
- Do not compute hashes or build manifests unless artifact identity cannot be established from the Git commit, formal config, and run path.
- Avoid speculative abstractions, broad grids, automatic recovery systems, and placeholder scripts. Let observed failures justify additional machinery.

## Git and source discipline

- Treat this repository as the authoritative source history. Before editing, run `git status --short` and preserve unrelated changes.
- Prefer existing authoritative interfaces. Do not leave copied `*_old`, `*_final`, `*_v2`, node-specific scripts, or temporary diagnostics in tracked source.
- Stage only intended files or hunks. Review both the unstaged and staged diff before committing.
- Keep one logical change per commit. A completed source or documentation change ends in a commit unless the user explicitly requests otherwise or the work is intentionally incomplete.
- Commit messages state what changed and why. Do not amend, rebase shared history, force-push, or destroy recoverability without explicit user authorization.
- Never commit credentials, private tokens, raw model artifacts, checkpoints, generated caches, or machine-specific runtime state.
