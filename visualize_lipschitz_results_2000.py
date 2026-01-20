import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Read the CSV file
df = pd.read_csv("lipschitz_results_summary_2000.csv")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Lipschitz Constant Analysis - MC Samples 2000', fontsize=16, fontweight='bold')

# 1. Norm Type Comparison
ax1 = axes[0, 0]
norm_data = df.groupby(['Norm Type'])['Lipschitz Constant'].agg(['mean', 'std'])
x_pos = np.arange(len(norm_data))
ax1.bar(x_pos, norm_data['mean'], yerr=norm_data['std'], capsize=5, 
        color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['Frobenius', 'Spectral'])
ax1.set_ylabel('Average Lipschitz Constant')
ax1.set_title('Norm Type Comparison')
ax1.grid(axis='y', alpha=0.3)
# Add text annotations
for i, (mean, std) in enumerate(zip(norm_data['mean'], norm_data['std'])):
    ax1.text(i, mean + std + 0.005, f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Model Architecture Comparison
ax2 = axes[0, 1]
model_data = df.groupby(['Model'])['Lipschitz Constant'].agg(['mean', 'std'])
x_pos = np.arange(len(model_data))
colors = ['#2ecc71', '#f39c12', '#9b59b6']
ax2.bar(x_pos, model_data['mean'], yerr=model_data['std'], capsize=5,
        color=colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_data.index)
ax2.set_ylabel('Average Lipschitz Constant')
ax2.set_title('Model Architecture Comparison')
ax2.grid(axis='y', alpha=0.3)
# Add text annotations
for i, (mean, std) in enumerate(zip(model_data['mean'], model_data['std'])):
    ax2.text(i, mean + std + 0.005, f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Channel Type Comparison
ax3 = axes[0, 2]
channel_data = df.groupby(['Channel Type (SNR)'])['Lipschitz Constant'].mean().sort_values()
channel_data.plot(kind='barh', ax=ax3, color='steelblue', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Average Lipschitz Constant')
ax3.set_title('Channel Type Comparison')
ax3.grid(axis='x', alpha=0.3)
# Adjust label size
ax3.tick_params(axis='y', labelsize=7)

# 4. Detailed comparison by Model and Norm
ax4 = axes[1, 0]
pivot_data = df.pivot_table(values='Lipschitz Constant', 
                             index=['Model', 'Dataset'], 
                             columns='Norm Type', 
                             aggfunc='mean')
x = np.arange(len(pivot_data))
width = 0.35
ax4.bar(x - width/2, pivot_data['norm-frob'], width, label='Frobenius',
        color='#3498db', alpha=0.7, edgecolor='black')
ax4.bar(x + width/2, pivot_data['norm-spec'], width, label='Spectral',
        color='#e74c3c', alpha=0.7, edgecolor='black')
ax4.set_ylabel('Lipschitz Constant')
ax4.set_title('Model/Dataset vs Norm Type')
ax4.set_xticks(x)
ax4.set_xticklabels([f"{m}\n{d}" for m, d in pivot_data.index], rotation=0, fontsize=9)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. BEC vs Rayleigh Channel Impact
ax5 = axes[1, 1]
# Separate BEC and Rayleigh
bec_data = df[df['Channel Type'].str.contains('bec')].groupby('Channel Type')['Lipschitz Constant'].mean()
# For Rayleigh, sort by SNR value (high to low)
rayleigh_df = df[df['Channel Type'].str.contains('rayleigh')]
rayleigh_grouped = rayleigh_df.groupby(['SNR (dB)', 'Channel Type (SNR)'])['Lipschitz Constant'].mean()
rayleigh_data = rayleigh_grouped.reset_index().sort_values('SNR (dB)', ascending=False).set_index('Channel Type (SNR)')['Lipschitz Constant']

# Combine and plot
all_channels = pd.concat([bec_data, rayleigh_data])
colors_channel = ['#27ae60' if 'bec' in ch else '#e67e22' for ch in all_channels.index]
all_channels.plot(kind='bar', ax=ax5, color=colors_channel, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Average Lipschitz Constant')
ax5.set_title('BEC (Green) vs Rayleigh (Orange) Channels')
ax5.set_xlabel('Channel Configuration')
ax5.tick_params(axis='x', rotation=45, labelsize=7)
ax5.grid(axis='y', alpha=0.3)

# 6. Heatmap of Lipschitz Constants
ax6 = axes[1, 2]
# Create a pivot table for heatmap
heatmap_data = df.pivot_table(values='Lipschitz Constant',
                               index=['Model', 'Dataset'],
                               columns=['Channel Type (SNR)', 'Norm Type'],
                               aggfunc='mean')
# Simplify column names for readability
heatmap_data.columns = [f"{ct.split('-')[0][:3]}-{nt.split('-')[1][:4]}" 
                        for ct, nt in heatmap_data.columns]
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax6,
            cbar_kws={'label': 'Lipschitz Constant'}, linewidths=0.5, annot_kws={'fontsize': 6})
ax6.set_title('Lipschitz Constant Heatmap')
ax6.set_xlabel('Channel-Norm', fontsize=8)
ax6.set_ylabel('Model-Dataset')
ax6.tick_params(axis='x', labelsize=6, rotation=90)

plt.tight_layout()
plt.savefig('lipschitz_analysis_plots_2000.pdf', dpi=300, bbox_inches='tight')
plt.savefig('lipschitz_analysis_plots_2000.png', dpi=300, bbox_inches='tight')
print("Plots saved as 'lipschitz_analysis_plots_2000.pdf' and 'lipschitz_analysis_plots_2000.png'")
plt.show()

# Additional plot: Noise level effect
fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Noise Level Impact on Lipschitz Constant (MC=2000)', fontsize=14, fontweight='bold')

# BEC outage rate effect
ax7.set_title('BEC: Outage Rate Effect')
for (model, dataset), group in df[df['Channel Type'].str.contains('bec')].groupby(['Model', 'Dataset']):
    for norm in group['Norm Type'].unique():
        subset = group[group['Norm Type'] == norm].sort_values('Channel Type')
        outage_rates = [float(ct.split('outage')[1]) for ct in subset['Channel Type']]
        lipschitz = subset['Lipschitz Constant'].values
        marker = 'o' if norm == 'norm-frob' else 's'
        linestyle = '-' if norm == 'norm-frob' else '--'
        label = f"{model}-{dataset}-{norm.split('-')[1]}"
        ax7.plot(outage_rates, lipschitz, marker=marker, label=label, linewidth=2, 
                markersize=8, linestyle=linestyle, alpha=0.8)
ax7.set_xlabel('Outage Rate', fontsize=12)
ax7.set_ylabel('Lipschitz Constant', fontsize=12)
ax7.legend(fontsize=8, loc='best')
ax7.grid(True, alpha=0.3)
ax7.set_xlim([0.05, 0.55])

# Rayleigh noise level effect (SNR)
ax8.set_title('Rayleigh: SNR Effect')
for (model, dataset), group in df[df['Channel Type'].str.contains('rayleigh')].groupby(['Model', 'Dataset']):
    for norm in group['Norm Type'].unique():
        subset = group[group['Norm Type'] == norm].sort_values('SNR (dB)', ascending=False)
        snr_values = subset['SNR (dB)'].values
        lipschitz = subset['Lipschitz Constant'].values
        marker = 'o' if norm == 'norm-frob' else 's'
        linestyle = '-' if norm == 'norm-frob' else '--'
        label = f"{model}-{dataset}-{norm.split('-')[1]}"
        ax8.plot(snr_values, lipschitz, marker=marker, label=label, linewidth=2, 
                markersize=8, linestyle=linestyle, alpha=0.8)
ax8.set_xlabel('SNR (dB)', fontsize=12)
ax8.set_ylabel('Lipschitz Constant', fontsize=12)
ax8.legend(fontsize=8, loc='best')
ax8.grid(True, alpha=0.3)
ax8.invert_xaxis()  # Higher SNR on left (better channel)

plt.tight_layout()
plt.savefig('lipschitz_noise_impact_2000.pdf', dpi=300, bbox_inches='tight')
plt.savefig('lipschitz_noise_impact_2000.png', dpi=300, bbox_inches='tight')
print("Noise impact plots saved as 'lipschitz_noise_impact_2000.pdf' and 'lipschitz_noise_impact_2000.png'")
plt.show()

# Additional plot 3: Direct SNR comparison for Rayleigh
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle('Rayleigh Channel: SNR Impact by Model (MC=2000)', fontsize=14, fontweight='bold')

rayleigh_df = df[df['Channel Type'].str.contains('rayleigh')]
models = rayleigh_df['Model'].unique()

for idx, model in enumerate(models):
    ax = axes3[idx]
    model_data = rayleigh_df[rayleigh_df['Model'] == model]
    
    for norm in model_data['Norm Type'].unique():
        subset = model_data[model_data['Norm Type'] == norm].sort_values('SNR (dB)', ascending=False)
        snr_values = subset['SNR (dB)'].values
        lipschitz = subset['Lipschitz Constant'].values
        marker = 'o' if norm == 'norm-frob' else 's'
        linestyle = '-' if norm == 'norm-frob' else '--'
        label = f"{norm.split('-')[1]}"
        ax.plot(snr_values, lipschitz, marker=marker, label=label, linewidth=2.5, 
                markersize=10, linestyle=linestyle, alpha=0.8)
    
    dataset = model_data['Dataset'].unique()[0]
    ax.set_title(f'{model} on {dataset}')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('Lipschitz Constant', fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Higher SNR on left

plt.tight_layout()
plt.savefig('lipschitz_snr_comparison_2000.pdf', dpi=300, bbox_inches='tight')
plt.savefig('lipschitz_snr_comparison_2000.png', dpi=300, bbox_inches='tight')
print("SNR comparison plots saved as 'lipschitz_snr_comparison_2000.pdf' and 'lipschitz_snr_comparison_2000.png'")
plt.show()

print("\nAll visualizations completed successfully!")
