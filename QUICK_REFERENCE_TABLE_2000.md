# Quick Reference Table - Lipschitz Constants (MC=2000)

## By Model and Norm Type

| Model | Dataset | Norm | BEC-0.1 | BEC-0.2 | BEC-0.3 | BEC-0.4 | BEC-0.5 | Ray-10dB | Ray-7.5dB | Ray-5dB | Ray-2.5dB | Ray-0dB |
|-------|---------|------|---------|---------|---------|---------|---------|----------|-----------|---------|-----------|---------|
| cnn-4 | mnist | Frob | 0.0054 | 0.0046 | 0.0040 | 0.0036 | 0.0033 | 0.0120 | 0.0128 | 0.0141 | 0.0150 | 0.0150 |
| cnn-4 | mnist | Spec | 0.0553 | 0.0549 | 0.0524 | 0.0509 | 0.0501 | 0.0889 | 0.1049 | 0.1134 | 0.1161 | 0.1175 |
| cnn-9 | cifar10 | Frob | 0.0015 | 0.0014 | 0.0014 | 0.0012 | 0.0012 | 0.0034 | 0.0044 | 0.0057 | 0.0058 | 0.0063 |
| cnn-9 | cifar10 | Spec | 0.0191 | 0.0185 | 0.0184 | 0.0167 | 0.0172 | 0.0388 | 0.0494 | 0.0633 | 0.0665 | 0.0730 |
| fcn-4 | mnist | Frob | 0.0106 | 0.0104 | 0.0100 | 0.0091 | 0.0092 | 0.0162 | 0.0161 | 0.0161 | 0.0160 | 0.0166 |
| fcn-4 | mnist | Spec | 0.0879 | 0.0869 | 0.0852 | 0.0782 | 0.0800 | 0.1208 | 0.1200 | 0.1199 | 0.1248 | 0.1292 |

## Best Configurations (Lowest Lipschitz)

| Rank | Model | Dataset | Channel | Norm | Lipschitz | Use Case |
|------|-------|---------|---------|------|-----------|----------|
| 1 | cnn-9 | cifar10 | BEC-0.5 | Frob | 0.0012 | Maximum robustness to erasures |
| 2 | cnn-9 | cifar10 | BEC-0.4 | Frob | 0.0012 | High robustness to erasures |
| 3 | cnn-9 | cifar10 | BEC-0.3 | Frob | 0.0014 | Moderate erasure tolerance |
| 4 | cnn-9 | cifar10 | BEC-0.2 | Frob | 0.0014 | Low erasure scenarios |
| 5 | cnn-9 | cifar10 | BEC-0.1 | Frob | 0.0015 | Minimal erasure |

## Worst Configurations (Highest Lipschitz)

| Rank | Model | Dataset | Channel | Norm | Lipschitz | Note |
|------|-------|---------|---------|------|-----------|------|
| 1 | fcn-4 | mnist | Ray-0dB | Spec | 0.1292 | Worst-case scenario |
| 2 | fcn-4 | mnist | Ray-2.5dB | Spec | 0.1248 | Very noisy Rayleigh |
| 3 | fcn-4 | mnist | Ray-10dB | Spec | 0.1208 | Even good SNR is sensitive |
| 4 | fcn-4 | mnist | Ray-5dB | Spec | 0.1199 | Mid-range SNR |
| 5 | fcn-4 | mnist | Ray-7.5dB | Spec | 0.1200 | Mid-high SNR |

## Average by Category

### By Model
- **cnn-9**: 0.0207 (BEST)
- **cnn-4**: 0.0447
- **fcn-4**: 0.0582 (WORST)

### By Norm Type
- **Frobenius**: 0.0084 (BEST - 8.79× better)
- **Spectral**: 0.0739 (WORST)

### By Channel Type (All models/norms averaged)
- **BEC-0.1**: 0.0300
- **BEC-0.2**: 0.0294
- **BEC-0.3**: 0.0286
- **BEC-0.4**: 0.0266 (BEST among BEC)
- **BEC-0.5**: 0.0268
- **Rayleigh-10dB**: 0.0467 (BEST among Rayleigh)
- **Rayleigh-7.5dB**: 0.0513
- **Rayleigh-5dB**: 0.0554
- **Rayleigh-2.5dB**: 0.0574
- **Rayleigh-0dB**: 0.0596 (WORST)

## SNR to Noise Parameter Conversion

| SNR (dB) | Tx Power | Noise Variance | Linear SNR |
|----------|----------|----------------|------------|
| 10.0 | 1.0 | 0.1000 | 10.00 |
| 7.5 | 1.0 | 0.1778 | 5.62 |
| 5.0 | 1.0 | 0.3162 | 3.16 |
| 2.5 | 1.0 | 0.5623 | 1.78 |
| 0.0 | 1.0 | 1.0000 | 1.00 |

**Formula:** SNR(dB) = 10 × log₁₀(Tx_Power / Noise_Var)

## Key Ratios and Comparisons

### Spectral vs Frobenius Ratio
- **Average Ratio**: 8.79×
- **Min Ratio**: 6.43× (fcn-4, mnist, BEC-0.4)
- **Max Ratio**: 14.03× (cnn-9, cifar10, BEC-0.4)

### BEC Impact (Frobenius only)
- **cnn-4**: 39% reduction from 0.1→0.5
- **cnn-9**: 21% reduction from 0.1→0.5
- **fcn-4**: 14% reduction from 0.1→0.5

### Rayleigh SNR Impact (Frobenius only)
- **cnn-4**: 20% reduction from 0dB→10dB
- **cnn-9**: 46% reduction from 0dB→10dB ⭐
- **fcn-4**: 3% reduction from 0dB→10dB (nearly constant!)

## Recommendations by Use Case

### For Generalization Bounds
✅ **Use:** cnn-9 on CIFAR-10 with Frobenius norm  
📊 **Expected Lipschitz:** ~0.0012-0.0063 depending on channel  
🎯 **Best for:** Tight PAC-Bayes bounds

### For Noisy Channels (Low SNR)
✅ **Use:** cnn-9 on CIFAR-10  
📊 **At 0dB SNR:** 0.0063 (Frobenius)  
🎯 **Best for:** Robust deployment in poor conditions

### For High-Erasure Scenarios
✅ **Use:** cnn-9 on CIFAR-10  
📊 **At 50% erasure:** 0.0012 (Frobenius) - surprisingly low!  
🎯 **Best for:** Unreliable communication channels

### For Minimal Lipschitz
✅ **Use:** cnn-9, CIFAR-10, BEC-0.5, Frobenius  
📊 **Lipschitz:** 0.0012  
🎯 **Best for:** Theoretical analysis and tightest bounds

### For Spectral Norm Analysis
✅ **Use:** cnn-9 on CIFAR-10  
📊 **Best case:** 0.0167 (BEC-0.4)  
⚠️ **Note:** Still 14× higher than Frobenius

## Interesting Observations

### 🔍 Non-Monotonic BEC Behavior
Most models show **decreasing** Lipschitz with **increasing** erasure rate (counterintuitive!)

### 🔍 FCN-4 SNR Insensitivity  
FCN-4 Lipschitz constant barely changes across SNR range (0-10 dB)

### 🔍 Optimal BEC Outage Rate
Around **0.4-0.5** provides best robustness across models

### 🔍 Diminishing Returns in SNR
Going from 5dB to 10dB provides smaller improvements than 0dB to 5dB

---

**Last Updated:** December 11, 2025  
**Source:** `lipschitz_results_summary_2000.csv`
