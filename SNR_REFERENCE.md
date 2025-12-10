# SNR Reference for Rayleigh Channel Experiments

## Signal-to-Noise Ratio (SNR) Mapping

For all Rayleigh fading experiments with **tx_power = 1.0**:

| Noise Variance (σ²) | SNR (linear) | SNR (dB) | Description |
|---------------------|--------------|----------|-------------|
| 0.1 | 10.0 | **10.0 dB** | High quality channel |
| 0.1778 | 5.624 | **7.5 dB** | Good channel |
| 0.3162 | 3.162 | **5.0 dB** | Medium channel |
| 0.5623 | 1.778 | **2.5 dB** | Poor channel |
| 1.0 | 1.0 | **0.0 dB** | Very poor channel |

## Calculation

SNR (dB) = 10 × log₁₀(Pₜₓ / σ²ₙₒᵢₛₑ)

Where:
- Pₜₓ = transmit power = 1.0
- σ²ₙₒᵢₛₑ = noise variance

## Channel Quality Interpretation

### By SNR Level

- **≥ 10 dB**: Excellent - Minimal impact on network robustness
- **7.5 dB**: Good - Slight degradation, still robust
- **5 dB**: Moderate - Noticeable degradation
- **2.5 dB**: Poor - Significant degradation
- **0 dB**: Critical - Severe degradation, saturation effects

### By Network Behavior (Average Lipschitz)

| SNR | CNN-4 | CNN-9 | FCN-4 | Trend |
|-----|-------|-------|-------|-------|
| 10 dB | 0.048 | 0.021 | 0.064 | Best performance |
| 7.5 dB | 0.054 | 0.024 | 0.066 | +11% degradation |
| 5 dB | 0.060 | 0.028 | 0.067 | +23% degradation |
| 2.5 dB | 0.063 | 0.035 | 0.070 | +35% degradation |
| 0 dB | 0.066 | 0.027 | 0.073 | Saturation/non-monotonic |

## Key Observations

1. **Logarithmic Relationship**: Lipschitz constant changes approximately linearly with SNR in dB scale (logarithmic in linear scale)

2. **Non-monotonic at 0 dB**: CNN-9 shows decrease at 0 dB SNR, suggesting saturation or measurement effects

3. **Model Sensitivity**:
   - **CNN-9**: Most robust, only 30% increase from 10dB → 0dB
   - **FCN-4**: Least sensitive to SNR changes (flat response)
   - **CNN-4**: Most predictable logarithmic response

4. **SNR Operating Regions**:
   - **High SNR** (> 7.5 dB): Information-rich regime
   - **Medium SNR** (2.5-7.5 dB): Adaptation zone
   - **Low SNR** (< 2.5 dB): Saturation regime

## Practical Guidelines

### System Design
- **Target SNR ≥ 7.5 dB** for reliable operation
- **Budget for 2.5 dB margin** for robust deployment
- **Avoid operation below 2.5 dB** due to unpredictable behavior

### Training
- **Train at multiple SNR levels**: 10, 5, and 2.5 dB
- **Focus on medium SNR** (5 dB) for robustness
- **Test at 0 dB** for worst-case validation

### Certification
- **Use 5 dB as stress test** for general robustness
- **Use 0 dB for worst-case** but interpret carefully due to saturation
- **Verify across all five SNR points** to catch non-monotonic behaviors

---

## Comparison with BEC Channels

| Channel Type | Metric | Range | Typical Impact |
|--------------|--------|-------|----------------|
| BEC | Outage probability | 0.1 - 0.5 | Non-monotonic (peak at 0.2) |
| Rayleigh | SNR (dB) | 0 - 10 | Logarithmic decrease with SNR |

**Key Difference**: BEC shows paradoxical behavior (worse at 20% than 50%), while Rayleigh generally improves with higher SNR (with saturation at extremes).

---

## References

All experimental data uses:
- **Tx power**: 1.0 (normalized)
- **Fading model**: Rayleigh
- **MC samples**: 500
- **Seed**: 7 (for reproducibility)
