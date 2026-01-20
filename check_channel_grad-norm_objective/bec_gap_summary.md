# BEC Gap Summary

- Rows with finite metrics: 360
- Best (delta ≤ 0): 0.0111 — model: cnn-9, epoch: 10, outage: 0.1, chan_penalty: 0.1, kl_penalty: 0.01
- Median delta: -0.0200
- Mean delta: -0.3143

## Top-5 Closest (smallest |delta|)

| Delta | Abs Delta | Baseline | Channel | Outage | Chan Penalty | KL Penalty | Epoch | Bound Valid | JSON Path |
|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0.0000 | 0.0000 | 0.0129 | 0.0129 | 0.1 | 1.0 | 0.1 | 50 | True | results/posterior/cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch50_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan1.0-bec-outage0.1-kl0.1_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |
| 0.0000 | 0.0000 | 0.0263 | 0.0263 | 0.1 | 0.1 | 0.01 | 20 | True | results/posterior/fcn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan0.1-bec-outage0.1-kl0.01_perc-pri0.5_epoch-pri20_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |
| 0.0000 | 0.0000 | 0.0179 | 0.0179 | 0.5 | 1.0 | 0.01 | 20 | True | results/posterior/cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan1.0-bec-outage0.5-kl0.01_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.5_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |
| 0.0000 | 0.0000 | 0.0182 | 0.0182 | 0.5 | 1.0 | 0.01 | 10 | True | results/posterior/cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch10_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan1.0-bec-outage0.5-kl0.01_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.5_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |
| 0.0000 | 0.0000 | 0.0183 | 0.0183 | 0.5 | 0.1 | 0.1 | 50 | True | results/posterior/cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch50_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan0.1-bec-outage0.5-kl0.1_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.5_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |

## Worst-5 (largest negative delta)

## Worst-5 (largest negative delta)

| Delta | Baseline | Channel | Outage | Chan Penalty | KL Penalty | Epoch | Bound Valid | JSON Path |
|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---|
| -0.8888 | 0.0132 | 0.9020 | 0.1 | 1.0 | 0.1 | 20 | False | results/posterior/cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-channel_norm-chan1.0-bec-outage0.1-kl0.1_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |
| -0.8888 | 0.0132 | 0.9020 | 0.1 | 1.0 | 1.0 | 20 | False | results/posterior/cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-channel_norm-chan1.0-bec-outage0.1-kl1.0_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |
| -0.8891 | 0.0129 | 0.9020 | 0.1 | 1.0 | 1.0 | 50 | False | results/posterior/cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch50_bs250_lr0.001_mon0.95_dp0.2_objective-channel_norm-chan1.0-bec-outage0.1-kl1.0_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |
| -0.8891 | 0.0129 | 0.9020 | 0.1 | 1.0 | 0.1 | 50 | False | results/posterior/cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch50_bs250_lr0.001_mon0.95_dp0.2_objective-channel_norm-chan1.0-bec-outage0.1-kl0.1_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |
| -0.8891 | 0.0129 | 0.9020 | 0.1 | 1.0 | 0.01 | 50 | False | results/posterior/cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch50_bs250_lr0.001_mon0.95_dp0.2_objective-channel_norm-chan1.0-bec-outage0.1-kl0.01_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json |
