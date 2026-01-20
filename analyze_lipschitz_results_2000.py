import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Path to the lipschitz results folder
lipschitz_path = Path("results/lipschitz")

# List to store all results
results = []

# Iterate through all folders
for folder in lipschitz_path.iterdir():
    if folder.is_dir() and "mcsamples2000" in folder.name:
        # Parse folder name
        parts = folder.name.split('_')
        
        # Extract information based on naming convention
        model_name = parts[0]  # e.g., cnn-4, fcn-4, cnn-9
        dataset = parts[1]  # e.g., mnist, cifar10
        prior_type = parts[2]  # e.g., rand
        prior_dist = parts[3]  # e.g., gaussian
        sigma = parts[4]  # e.g., sig0.03
        channel_type = parts[5]  # e.g., bec-outage0.1, rayleigh-tx1.0-noise0.1
        channel_layer = parts[6]  # e.g., chan-layer2
        mc_samples = parts[7]  # e.g., mcsamples2000
        norm_type = parts[8]  # e.g., norm-frob
        seed = parts[9] if len(parts) > 9 else "N/A"  # e.g., seed7
        
        # Calculate SNR for Rayleigh channels
        channel_type_display = channel_type
        snr_db = None
        if "rayleigh" in channel_type:
            # Parse tx power and noise variance
            # Format: rayleigh-tx1.0-noise0.1
            parts_ch = channel_type.split('-')
            tx_power = float(parts_ch[1].replace('tx', ''))
            noise_var = float(parts_ch[2].replace('noise', ''))
            snr_db = 10 * np.log10(tx_power / noise_var)
            channel_type_display = f"rayleigh-snr{snr_db:.1f}dB"
        
        # Read the JSON file
        json_file = folder / "lipschitz_results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                lipschitz_constant = data.get("lipschitz_constant", None)
                
                # Store the result
                results.append({
                    "Model": model_name,
                    "Dataset": dataset,
                    "Prior Type": prior_type,
                    "Prior Dist": prior_dist,
                    "Sigma": sigma,
                    "Channel Type": channel_type,
                    "Channel Type (SNR)": channel_type_display,
                    "SNR (dB)": snr_db,
                    "Channel Layer": channel_layer,
                    "MC Samples": mc_samples,
                    "Norm Type": norm_type,
                    "Seed": seed,
                    "Lipschitz Constant": lipschitz_constant
                })

# Create DataFrame
df = pd.DataFrame(results)

# Sort by model, dataset, channel type, norm type for better readability
df = df.sort_values(by=["Model", "Dataset", "Channel Type", "Norm Type"])

# Display the full table
print("=" * 150)
print("LIPSCHITZ CONSTANT RESULTS (MC Samples = 2000)")
print("=" * 150)
# Display with SNR for better readability
display_df = df[["Model", "Dataset", "Channel Type (SNR)", "SNR (dB)", "Channel Layer", "Norm Type", "Lipschitz Constant"]]
print(display_df.to_string(index=False))
print("\n")

# Analysis and insights
print("=" * 150)
print("ANALYSIS AND INSIGHTS")
print("=" * 150)

# 1. Compare norm types
print("\n1. COMPARISON BY NORM TYPE:")
print("-" * 80)
norm_comparison = df.groupby(["Model", "Dataset", "Channel Type (SNR)", "Norm Type"])["Lipschitz Constant"].mean().unstack(fill_value=None)
print(norm_comparison)
print("\nInsight: Comparing Frobenius norm vs Spectral norm for the same configurations.")

# 2. Compare channel types
print("\n2. COMPARISON BY CHANNEL TYPE (WITH SNR):")
print("-" * 80)
channel_comparison = df.groupby(["Model", "Dataset", "Channel Type (SNR)"])["Lipschitz Constant"].mean()
print(channel_comparison)
print("\nInsight: How different channel types affect the Lipschitz constant.")

# 2b. SNR-based analysis for Rayleigh
print("\n2b. RAYLEIGH CHANNEL: SNR ANALYSIS:")
print("-" * 80)
rayleigh_df = df[df["Channel Type (SNR)"].str.contains("rayleigh", na=False)]
if not rayleigh_df.empty:
    snr_analysis = rayleigh_df.groupby(["Model", "Dataset", "SNR (dB)", "Norm Type"])["Lipschitz Constant"].mean()
    print(snr_analysis)
    print("\nInsight: Lipschitz constant vs SNR (dB) for Rayleigh fading channels.")

# 3. Compare models
print("\n3. COMPARISON BY MODEL:")
print("-" * 80)
model_comparison = df.groupby(["Model", "Dataset"])["Lipschitz Constant"].agg(['mean', 'min', 'max', 'std'])
print(model_comparison)
print("\nInsight: Different model architectures and their Lipschitz constants.")

# 4. Channel layer comparison
print("\n4. COMPARISON BY CHANNEL LAYER:")
print("-" * 80)
layer_comparison = df.groupby(["Model", "Channel Layer", "Channel Type"])["Lipschitz Constant"].mean()
print(layer_comparison)
print("\nInsight: Effect of applying channel noise at different layers.")

# 5. Detailed analysis by channel parameters
print("\n5. DETAILED CHANNEL ANALYSIS:")
print("-" * 80)

# BEC channels
bec_results = df[df["Channel Type"].str.contains("bec")]
if not bec_results.empty:
    print("\nBinary Erasure Channel (BEC) Results:")
    print("Outage Rate Impact:")
    for (model, dataset), group in bec_results.groupby(["Model", "Dataset"]):
        print(f"\n  {model} on {dataset}:")
        bec_pivot = group.pivot_table(values='Lipschitz Constant', 
                                       index='Channel Type', 
                                       columns='Norm Type', 
                                       aggfunc='mean')
        print(bec_pivot)
    print("\nInsight: Higher outage rates (0.5) vs lower (0.1) impact on Lipschitz constant.")

# Rayleigh channels
rayleigh_results = df[df["Channel Type"].str.contains("rayleigh")]
if not rayleigh_results.empty:
    print("\n\nRayleigh Fading Channel Results (by SNR):")
    print("SNR Impact:")
    for (model, dataset), group in rayleigh_results.groupby(["Model", "Dataset"]):
        print(f"\n  {model} on {dataset}:")
        rayleigh_pivot = group.pivot_table(values='Lipschitz Constant',
                                            index='SNR (dB)',
                                            columns='Norm Type',
                                            aggfunc='mean')
        print(rayleigh_pivot.sort_index(ascending=False))  # High SNR to low SNR
    print("\nInsight: Higher SNR (better channel) → Lower Lipschitz constant (more robust).")

# 6. Key findings
print("\n" + "=" * 150)
print("KEY FINDINGS:")
print("=" * 150)

findings = []

# Finding 1: Norm type effect
frob_mean = df[df["Norm Type"] == "norm-frob"]["Lipschitz Constant"].mean()
spec_mean = df[df["Norm Type"] == "norm-spec"]["Lipschitz Constant"].mean()
findings.append(f"1. Norm Type: Frobenius norm average = {frob_mean:.6f}, Spectral norm average = {spec_mean:.6f}")
if frob_mean < spec_mean:
    findings.append(f"   → Frobenius norm gives {((spec_mean/frob_mean - 1) * 100):.2f}% lower Lipschitz constants on average")
else:
    findings.append(f"   → Spectral norm gives {((frob_mean/spec_mean - 1) * 100):.2f}% lower Lipschitz constants on average")

# Finding 2: Model architecture
for model in df["Model"].unique():
    model_mean = df[df["Model"] == model]["Lipschitz Constant"].mean()
    findings.append(f"2. Model {model}: Average Lipschitz constant = {model_mean:.6f}")

# Finding 3: Channel severity for BEC
if not bec_results.empty:
    outage_rates = sorted(bec_results["Channel Type"].unique())
    findings.append("\n3. BEC Channel Impact (Outage Rate):")
    for outage in outage_rates:
        outage_mean = df[df["Channel Type"] == outage]["Lipschitz Constant"].mean()
        findings.append(f"   {outage}: {outage_mean:.6f}")

# Finding 4: Channel severity for Rayleigh (SNR)
if not rayleigh_results.empty:
    snr_values = sorted(rayleigh_results["SNR (dB)"].dropna().unique(), reverse=True)
    findings.append("\n4. Rayleigh Channel Impact (SNR):")
    for snr in snr_values:
        snr_mean = df[df["SNR (dB)"] == snr]["Lipschitz Constant"].mean()
        findings.append(f"   SNR = {snr:.1f} dB: {snr_mean:.6f}")

# Finding 5: Dataset effect
findings.append("\n5. Dataset Effect:")
for dataset in df["Dataset"].unique():
    dataset_mean = df[df["Dataset"] == dataset]["Lipschitz Constant"].mean()
    findings.append(f"   {dataset}: {dataset_mean:.6f}")

# Finding 6: Trend analysis
findings.append("\n6. Trend Analysis:")

# BEC trend
if not bec_results.empty:
    bec_trend = bec_results.groupby("Channel Type")["Lipschitz Constant"].mean().sort_index()
    if len(bec_trend) > 1:
        outage_vals = [float(ct.split('outage')[1]) for ct in bec_trend.index]
        lip_vals = bec_trend.values
        # Check if increasing
        if all(lip_vals[i] <= lip_vals[i+1] for i in range(len(lip_vals)-1)):
            findings.append("   BEC: Lipschitz constant INCREASES with outage rate (monotonic)")
        elif all(lip_vals[i] >= lip_vals[i+1] for i in range(len(lip_vals)-1)):
            findings.append("   BEC: Lipschitz constant DECREASES with outage rate (monotonic)")
        else:
            findings.append("   BEC: Lipschitz constant shows NON-MONOTONIC behavior with outage rate")

# Rayleigh trend
if not rayleigh_results.empty:
    rayleigh_trend = rayleigh_results.groupby("SNR (dB)")["Lipschitz Constant"].mean().sort_index()
    if len(rayleigh_trend) > 1:
        snr_vals = rayleigh_trend.index.values
        lip_vals = rayleigh_trend.values
        # Check if decreasing (higher SNR should give lower Lip)
        if all(lip_vals[i] >= lip_vals[i+1] for i in range(len(lip_vals)-1)):
            findings.append("   Rayleigh: Lipschitz constant DECREASES with SNR (monotonic - expected)")
        elif all(lip_vals[i] <= lip_vals[i+1] for i in range(len(lip_vals)-1)):
            findings.append("   Rayleigh: Lipschitz constant INCREASES with SNR (monotonic - unexpected!)")
        else:
            findings.append("   Rayleigh: Lipschitz constant shows NON-MONOTONIC behavior with SNR")

for finding in findings:
    print(finding)

# Statistical summary
print("\n" + "=" * 150)
print("STATISTICAL SUMMARY:")
print("=" * 150)
print(f"Total number of configurations: {len(df)}")
print(f"Models tested: {df['Model'].unique()}")
print(f"Datasets: {df['Dataset'].unique()}")
print(f"Channel types: {df['Channel Type'].nunique()}")
print(f"Norm types: {df['Norm Type'].unique()}")
print(f"\nLipschitz Constant Statistics:")
print(f"  Mean: {df['Lipschitz Constant'].mean():.6f}")
print(f"  Median: {df['Lipschitz Constant'].median():.6f}")
print(f"  Std Dev: {df['Lipschitz Constant'].std():.6f}")
print(f"  Min: {df['Lipschitz Constant'].min():.6f}")
print(f"  Max: {df['Lipschitz Constant'].max():.6f}")

# Save results to CSV
output_file = "lipschitz_results_summary_2000.csv"
df.to_csv(output_file, index=False)
print(f"\n\nResults saved to: {output_file}")
