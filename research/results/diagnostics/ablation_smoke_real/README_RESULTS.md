# Real-data ablation diagnostic

This one-epoch CPU run checked that the real-data pipeline completed and produced the expected statistics files. It is a software diagnostic, not a fair model comparison.

The run used 1,330,816 rows from 69 profiles and a profile-grouped 48/10/11 train/validation/test split. Five models were evaluated over the same 184,699 forecast windows. Neural settings were seed 0, batch size 512, AdamW learning rate 3e-4, weight decay 1e-5, gradient clipping 1.0, one epoch, and a 5% random training-window fraction. History was 128 samples and horizon was 20 steps.

## Diagnostic metrics

| Model | Pooled test MAE ( degC) |
|---|---:|
| Ridge direct | 0.11188 |
| Global adaptive | 0.12707 |
| `oc_full` | 0.12748 |
| Persistence | 0.13501 |
| Linear extrapolation | 0.21586 |

The `oc_full` checkpoint selected epoch 1 with validation MAE 0.11904  degC. For `oc_full` versus persistence, the variant-minus-reference delta was +0.00754  degC with 95% profile-bootstrap CI -0.00022 to 0.01512  degC; Wilcoxon *p* = 0.0674 and Holm-adjusted *p* = 0.1611 over 11 test profiles. This result is inconclusive.

The DM tests are supporting diagnostics only. They compare serially related window errors, so the profile-level analysis is the relevant summary here. The default bandwidth was floor(sqrt(n)) for thinned series with 102-1,817 origins; all bandwidths remained under the sample-size cap.

## Files and status

`stats_paper/` contains tables, LaTeX snippets, statistical outputs, and `forest_plot.png`. `run_log.jsonl` records the run settings and validation history. Per-window errors are under `errors/` and excluded from Git because of their size.

This run has been superseded as the project's main comparison by the nested GPU evaluation in `../../final/nested_cv_gpu/`. Its numbers are retained as a record of the pipeline check and should not be cited as publication evidence.