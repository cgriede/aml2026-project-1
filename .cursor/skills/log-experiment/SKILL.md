---
name: log-experiment
description: >-
  Logs pasted `python lib` cross-validation runs into this repo's experiment
  note and runs/iterations.md. Use when the user says /log-experiment, "log
  this experiment", or pastes python lib output and asks to record it.
disable-model-invocation: true
---

# Log experiment

Record a pasted `python lib` sweep. Do not rerun it. Do not edit `lib/` or `GLOBAL.py`.

## Ask first

Ask before writing when either is unclear:

- Which note the runs continue. Default is the experiment note already used for this sweep.
- Which printed line is the knob (`contamination`, `nu`, or another setting).

## Write

1. Read `runs/iterations.md` and the target note (`experiments/NNN/EXP.md`).
2. From each pasted run, take the knob, rows dropped, train `n` from the shape line, the `|r|>=0.95` column list, Ridge CV mean ± std, XGB CV mean ± std, and `n_estimators`.
3. The comparison arm is the unfiltered row already in that note. Do not invent a baseline.
4. Append one section to the note: one table, same columns as the tables already there, then two or three sentences. Round means to three decimals.
5. Append the next row in `runs/iterations.md`. Link the note. XGB is the goal. Ridge is reported beside it.

## Keep rule

An arm replaces the unfiltered run only when its mean XGB R² is at least 0.01 higher and that gap is larger than the challenger's fold standard error (`std / sqrt(5)`). Folds are re-split after row drops, so they are not paired. If no arm clears the bar, the keep cell is **no, no row drop**, and the accepted-pipeline paragraph stays as it is.

## Reply

The table and the keep sentence. No restatement of the logs.
