# Shared blocks-map components

This directory reuses the existing shared board encoder and official-data helpers from
`codex/blocks-multigoal` at commit `3a4c572`. The imported `src/model.py`,
`src/oracle.py`, and `src/multiboard_data.py` retain their original implementations.
They are used by `experiments/flamingo_map_reader`; this change does not retrain Q.

`BoardEncoder` maps 100 occupancy bits to Q. The frozen checkpoint specifies its
hidden dimension, output dimension, scale and directed distance metric. Environment
successors use the eight official shapes in `experiments/gcml_counterexamples`.
The exact oracle is used offline for task-length selection.
