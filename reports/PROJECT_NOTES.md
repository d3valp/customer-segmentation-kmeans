
# Project notes for reviewers

This repository is designed as a portfolio project, not as a production system.

## Why the data is synthetic

The project ships with a realistic synthetic sample so anyone can run it immediately. This avoids privacy problems and removes dependencies on external data providers. The tradeoff is that the model results demonstrate the workflow rather than real-world predictive performance.

## What makes the project reviewable

- raw and processed data are separated;
- pipeline steps are runnable as modules;
- artifacts are generated into `models/` and `reports/figures/`;
- tests cover core cleaning/feature logic;
- limitations and next steps are stated explicitly.
