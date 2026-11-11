# Results

Test-split metrics (the test split was not used for checkpoint or model selection). Rows sorted by validation ROC-AUC. Deployment model (by validation): **efficientnet_b0**.

| run | model | params_m | best_epoch | epochs_run | best_val_roc_auc | accuracy | accuracy_95ci | macro_f1 | ai_precision | ai_recall | roc_auc | roc_auc_95ci | roc_auc_jpeg70 | roc_auc_jpeg50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | efficientnet_b0.ra_in1k | 4.01 | 23 | 25 | 0.9524 | 0.8433 | 0.803–0.887 | 0.8433 | 0.8552 | 0.8267 | 0.9038 | 0.866–0.940 | 0.8931 | 0.8864 |
| resnet50 | resnet50.a1_in1k | 23.51 | 13 | 20 | 0.9355 | 0.8067 | 0.763–0.850 | 0.8067 | 0.8067 | 0.8067 | 0.8975 | 0.859–0.931 | 0.8932 | 0.8908 |
| simple_cnn | simple_cnn | 3.13 | 55 | 60 | 0.8769 | 0.7667 | 0.720–0.813 | 0.7666 | 0.7778 | 0.7467 | 0.8351 | 0.788–0.881 | 0.8337 | 0.8299 |
