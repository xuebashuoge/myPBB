# Channel vs Vanilla Objective Analysis Report

## Executive Summary

- **Total configurations analyzed**: 100
- **Configurations where channel outperforms vanilla**: 24
- **Configurations with valid bounds**: 56
- **Successful configurations (both criteria)**: 3

## Key Findings

### Average Improvements

- **Average loss reduction**: 6.80%
- **Average error reduction**: 9.13%
- **Maximum loss reduction**: 9.49%
- **Maximum error reduction**: 11.80%

### Performance by Channel Type

| channel_type   |   loss_reduction_pct |   error_reduction_pct |
|:---------------|---------------------:|----------------------:|
| rayleigh-zf    |              6.80238 |               9.13185 |

### Performance by Model and Dataset

| model_dataset   |   loss_reduction_pct |   error_reduction_pct |
|:----------------|---------------------:|----------------------:|
| cnn-9_cifar10   |              6.80238 |               9.13185 |

## Detailed Results

See the following files for detailed results:

- `all_comparisons.csv`: All channel vs vanilla comparisons
- `successful_improvements.csv`: Configurations meeting success criteria
- `summary_improvements.png`: Overview of improvements
- `channel_spec_effects.png`: Effect of channel specifications
- `parameter_effects.png`: Effect of various parameters
- `bound_validation.png`: Validation of theoretical bounds
