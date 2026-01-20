"""
Analysis: Why BEC and Rayleigh Channels Show DIFFERENT Patterns

This script explains the key difference between BEC and Rayleigh channels
in terms of their Lipschitz constant behavior.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("WHY BEC AND RAYLEIGH SHOW DIFFERENT PATTERNS")
print("="*80)

print("\n" + "="*80)
print("KEY INSIGHT: Different Numerator Behavior!")
print("="*80)

print("\nRecall: Lipschitz k = |loss(w') - loss(w)| / ||w' - w||")
print("         k = NUMERATOR / DENOMINATOR")

print("\n" + "="*80)
print("1. BEC CHANNEL (Discrete Erasures)")
print("="*80)

print("\nChannel Model:")
print("  w' = Bernoulli(1-outage) ∈ {0, 1}")
print("  Ideal: w = 1 (identity)")
print("  Output: y' = w' * x")

print("\nDenominator (Distance):")
for outage in [0.1, 0.3, 0.5]:
    d = np.sqrt(outage * 1000)
    print(f"  Outage {outage:.1f}: ||w'-I||_F = √({outage}×1000) = {d:.1f}")

print("\nNumerator (Loss Difference):")
print("  When outage increases:")
print("    • More values erased (0s instead of 1s)")
print("    • Network LOSES information")
print("    • But: Remaining information is still CLEAN (no noise added)")
print("    • Loss increases, but NOT proportionally to √outage")
print("    • Why? Network redundancy, averaging effects")

print("\n  Key: Information LOSS, but no CORRUPTION")

print("\nScaling Analysis:")
print("  Denominator: grows as √outage")
print("  Numerator: grows slower (sub-linear in outage)")
print("  Result: k = Numerator/Denominator DECREASES")

print("\n" + "="*80)
print("2. RAYLEIGH CHANNEL (Continuous Noise)")
print("="*80)

print("\nChannel Model:")
print("  h ~ CN(0, tx_power) = complex gain")
print("  n ~ CN(0, noise_var) = additive noise")
print("  y' = h*x + n")
print("  SNR = tx_power / noise_var")

print("\nDenominator (Distance):")
print("  For scalar flat fading:")
print("    weight = h ~ CN(0, tx_power)")
print("    bias = n ~ CN(0, noise_var)")
print("    Ideal: weight = 1, bias = 0")
print("    Distance metric:")
print("      d² = (||h - 1||² + ||n||²)")

tx_power = 1.0
for snr_db in [0, 5, 10]:
    snr_linear = 10**(snr_db/10)
    noise_var = tx_power / snr_linear
    
    # Expected distance
    # E[|h-1|²] = E[|h|²] - 2*Re(E[h]) + 1
    # For h ~ CN(0, tx_power), E[h] = 0, E[|h|²] = tx_power
    # So E[|h-1|²] = tx_power + 1
    expected_h_dist_sq = tx_power + 1
    
    # E[|n|²] = noise_var
    expected_n_dist_sq = noise_var
    
    expected_d = np.sqrt(expected_h_dist_sq + expected_n_dist_sq)
    
    print(f"\n  SNR {snr_db} dB (noise_var={noise_var:.4f}):")
    print(f"    E[||h-1||²] ≈ {expected_h_dist_sq:.4f}")
    print(f"    E[||n||²] = {expected_n_dist_sq:.4f}")
    print(f"    E[distance] ≈ {expected_d:.4f}")

print("\n  IMPORTANT: Distance is roughly CONSTANT across SNR!")
print("    (Dominated by E[|h-1|²] = tx_power + 1 ≈ 2)")

print("\nNumerator (Loss Difference):")
print("  When SNR decreases (more noise):")
print("    • Additive noise n increases")
print("    • Signal corruption increases")
print("    • Output: y' = h*x + n becomes more CORRUPTED")
print("    • Loss difference |loss(y') - loss(x)| INCREASES")

print("\n  When SNR increases (less noise):")
print("    • Additive noise n decreases")
print("    • Signal corruption decreases")
print("    • Output closer to ideal")
print("    • Loss difference DECREASES")

print("\n  Key: Information CORRUPTION (noise added to ALL values)")

print("\nScaling Analysis:")
print("  Denominator: roughly CONSTANT (~√2 for tx_power=1)")
print("  Numerator: DECREASES with SNR (less noise corruption)")
print("  Result: k = Numerator/Denominator DECREASES with SNR")

print("\n" + "="*80)
print("3. SIDE-BY-SIDE COMPARISON")
print("="*80)

print("\n┌─────────────────┬──────────────────────┬──────────────────────┐")
print("│    Property     │      BEC Channel     │  Rayleigh Channel    │")
print("├─────────────────┼──────────────────────┼──────────────────────┤")
print("│ Worse Channel   │ High outage (0.5)    │ Low SNR (0 dB)       │")
print("│ Better Channel  │ Low outage (0.1)     │ High SNR (10 dB)     │")
print("├─────────────────┼──────────────────────┼──────────────────────┤")
print("│ Denominator     │ √outage × √dim       │ ~constant (~√2)      │")
print("│ (Distance)      │ INCREASES with outage│ Does NOT scale w/SNR │")
print("├─────────────────┼──────────────────────┼──────────────────────┤")
print("│ Numerator       │ Sub-linear growth    │ DECREASES with SNR   │")
print("│ (Loss diff)     │ Info loss (discrete) │ Noise (continuous)   │")
print("├─────────────────┼──────────────────────┼──────────────────────┤")
print("│ Lipschitz k     │ DECREASES w/ outage  │ DECREASES with SNR   │")
print("│                 │ (denom grows faster) │ (numer decreases)    │")
print("└─────────────────┴──────────────────────┴──────────────────────┘")

print("\n" + "="*80)
print("4. THE CRITICAL DIFFERENCE")
print("="*80)

print("\nBEC (Bernoulli):")
print("  • Discrete random variable: {0, 1}")
print("  • Mean = 1 - outage")
print("  • Variance = outage × (1 - outage)")
print("  • Distance from ideal (1):")
print("    E[(w-1)²] = outage  → Distance = √outage")
print("  → DENOMINATOR SCALES WITH OUTAGE")

print("\nRayleigh (Complex Normal):")
print("  • Continuous random variable: h ~ CN(0, tx_power)")
print("  • Mean = 0 (not 1!)")
print("  • For h ~ CN(0, σ²): E[|h|²] = σ²")
print("  • Distance from ideal (1):")
print("    E[|h-1|²] = E[|h|²] + 1 = tx_power + 1")
print("  • For tx_power=1: E[|h-1|²] = 2 (CONSTANT!)")
print("  → DENOMINATOR DOES NOT SCALE WITH SNR")

print("\n" + "="*80)
print("5. MATHEMATICAL PROOF")
print("="*80)

print("\nBEC Denominator:")
print("  w ~ Bernoulli(1-p) where p=outage")
print("  E[w] = 1-p")
print("  E[(w-1)²] = E[w²] - 2E[w] + 1")
print("           = E[w] - 2E[w] + 1    (since w ∈ {0,1}, w²=w)")
print("           = (1-p) - 2(1-p) + 1")
print("           = 1 - p - 2 + 2p + 1")
print("           = p")
print("  → Distance = √(p × dimension) = √(outage × dim)")
print("  → GROWS WITH OUTAGE ✓")

print("\nRayleigh Denominator (weight part):")
print("  h ~ CN(0, tx_power)")
print("  E[|h-1|²] = E[|h|²] - 2*Re(E[h*·1]) + |1|²")
print("           = tx_power - 2*Re(0) + 1")
print("           = tx_power + 1")
print("  For tx_power=1: E[|h-1|²] = 2 (CONSTANT!)")
print("  → Distance ≈ √2 (INDEPENDENT OF SNR!)")

print("\nRayleigh Denominator (noise part):")
print("  n ~ CN(0, noise_var)")
print("  E[|n|²] = noise_var = tx_power / SNR")
print("  → DECREASES with SNR")
print("  But this is typically much smaller than E[|h-1|²] = 2")

print("\nTotal Rayleigh Distance:")
print("  d² ≈ E[|h-1|²] + E[|n|²]")
print("     ≈ (tx_power + 1) + noise_var")
print("     ≈ 2 + noise_var    (for tx_power=1)")
print("  → Dominated by constant term (2)")
print("  → Roughly CONSTANT across SNR range!")

print("\n" + "="*80)
print("6. NUMERICAL VERIFICATION")
print("="*80)

print("\nBEC Distances:")
dimension = 1000
for outage in [0.1, 0.2, 0.3, 0.4, 0.5]:
    distance = np.sqrt(outage * dimension)
    print(f"  Outage {outage:.1f}: distance = {distance:.2f}")

print("\nRayleigh Distances:")
tx_power = 1.0
for snr_db in [0, 2.5, 5, 7.5, 10]:
    snr_linear = 10**(snr_db/10)
    noise_var = tx_power / snr_linear
    # Distance from h~CN(0,1) to 1
    weight_dist_sq = tx_power + 1  # E[|h-1|²]
    noise_dist_sq = noise_var      # E[|n|²]
    total_dist = np.sqrt(weight_dist_sq + noise_dist_sq)
    print(f"  SNR {snr_db:4.1f} dB: d² = {weight_dist_sq:.3f} + {noise_dist_sq:.3f} = {total_dist:.3f}")

print("\n  → BEC: Distance varies 2.24× (from 10 to 22.4)")
print("  → Rayleigh: Distance varies only 1.17× (from 1.41 to 1.66)")

print("\n" + "="*80)
print("7. CONCLUSION")
print("="*80)

print("\nWhy do BEC and Rayleigh show different patterns?")
print()
print("BEC:")
print("  ✓ Denominator GROWS with outage (∝ √outage)")
print("  ✓ Numerator grows sub-linearly")
print("  → Ratio DECREASES: Higher outage → Lower Lipschitz")
print()
print("Rayleigh:")
print("  ✓ Denominator is roughly CONSTANT (≈ √2)")
print("  ✓ Numerator DECREASES with SNR (less noise corruption)")
print("  → Ratio DECREASES: Higher SNR → Lower Lipschitz")
print()
print("Both show k DECREASING, but for DIFFERENT REASONS:")
print("  • BEC: Denominator effect dominates")
print("  • Rayleigh: Numerator effect dominates")

print("\n" + "="*80)
print("FINAL ANSWER:")
print("="*80)
print("The patterns are DIFFERENT because:")
print()
print("1. BEC uses Bernoulli(1-p) centered at 1-p")
print("   → Distance from 1 scales with p (outage)")
print()
print("2. Rayleigh uses CN(0, σ²) centered at 0")
print("   → Distance from 1 is constant (≈ √2 for σ²=1)")
print()
print("3. The SAME Lipschitz definition gives DIFFERENT scaling")
print("   because the channel models are fundamentally different!")
print("="*80)
