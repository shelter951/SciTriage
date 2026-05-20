# Candidate Manifest Summary

- Candidates: 4

## Families

- `batch_schedule`: 1
- `depth_scaling`: 1
- `interaction`: 1
- `optimizer_lr`: 1

## Candidates

| Variant | Family | Base | Patch Intent | Claim |
|---|---|---|---|---|
| batch_small | batch_schedule | depth4_baseline | Reduce TOTAL_BATCH_SIZE from 2**16 to 2**15. | Reducing total batch size to 2**15 improves val_bpb under the 4090 quick-probe autoresearch contract. |
| depth7_batch_small | interaction | depth7 | Apply smaller total batch size to the accepted depth7 candidate. | Combining depth7 with smaller total batch size further improves val_bpb. |
| depth7 | depth_scaling | depth4_baseline | Increase DEPTH from 4 to 7 under the same 4090 quick-probe contract. | Increasing depth to 7 improves val_bpb under the 4090 quick-probe autoresearch contract. |
| matrix_lr_high | optimizer_lr | depth4_baseline | Increase MATRIX_LR from 0.04 to 0.06. | Increasing MATRIX_LR to 0.06 significantly improves val_bpb under the 4090 quick-probe autoresearch contract. |