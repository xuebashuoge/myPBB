#!/usr/bin/env python3
"""
Create a visual summary table of key results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

# Load data
successful = pd.read_csv('successful_improvements.csv')
all_improvements = pd.read_csv('all_improvements.csv')
all_comparisons = pd.read_csv('all_comparisons.csv')

# Create a comprehensive summary figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('Channel vs Vanilla Objective: Comprehensive Analysis Summary', 
             fontsize=16, fontweight='bold')

# 1. Overall Statistics (top left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')

stats_text = f"""
OVERALL STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Configurations: {len(all_comparisons)}

Channel Types:
  • BEC: {len(all_comparisons[all_comparisons['channel_type'] == 'bec'])}
  • Rayleigh-ZF: {len(all_comparisons[all_comparisons['channel_type'] == 'rayleigh-zf'])}

Results:
  ✓ Goal Satisfied: {len(all_improvements)} (24%)
  ✓ Bounds Valid: {(all_comparisons['bounds_valid']).sum()} (56%)
  ✓ BOTH: {len(successful)} (3%)
"""

ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Success Rate by Channel Type (top middle)
ax2 = fig.add_subplot(gs[0, 1])
channel_stats = []
for channel_type in all_comparisons['channel_type'].unique():
    subset = all_comparisons[all_comparisons['channel_type'] == channel_type]
    channel_stats.append({
        'Channel': channel_type,
        'Total': len(subset),
        'Improved': len(subset[subset['goal_satisfied']]),
        'Valid Bounds': len(subset[subset['bounds_valid']]),
        'Both': len(subset[subset['goal_satisfied'] & subset['bounds_valid']])
    })

channel_df = pd.DataFrame(channel_stats)
x = np.arange(len(channel_df))
width = 0.2

ax2.bar(x - 1.5*width, channel_df['Total'], width, label='Total', alpha=0.8)
ax2.bar(x - 0.5*width, channel_df['Improved'], width, label='Improved', alpha=0.8)
ax2.bar(x + 0.5*width, channel_df['Valid Bounds'], width, label='Valid Bounds', alpha=0.8)
ax2.bar(x + 1.5*width, channel_df['Both'], width, label='Both ✓', alpha=0.8, color='green')

ax2.set_xlabel('Channel Type')
ax2.set_ylabel('Count')
ax2.set_title('Success Rates by Channel Type')
ax2.set_xticks(x)
ax2.set_xticklabels(channel_df['Channel'])
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Top 3 Successful Configurations (top right)
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')

if len(successful) > 0:
    top3_text = "TOP 3 SUCCESSFUL CONFIGS\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    for idx, row in successful.nlargest(3, 'loss_reduction_pct').iterrows():
        top3_text += f"#{idx+1}: {row['model']}/{row['dataset']}\n"
        top3_text += f"  Channel: {row['channel_type']}\n"
        top3_text += f"  SNR: {row['channel_spec']:.1f} dB\n"
        top3_text += f"  Epoch: {row['epoch']}, KL: {row['kl_penalty']}\n"
        top3_text += f"  Loss ↓: {row['loss_reduction_pct']:.2f}%\n"
        top3_text += f"  Error ↓: {row['error_reduction_pct']:.2f}%\n\n"
else:
    top3_text = "No configurations\nmeet both criteria"

ax3.text(0.05, 0.95, top3_text, transform=ax3.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 4. Improvement distribution (middle left)
ax4 = fig.add_subplot(gs[1, 0])
if len(all_improvements) > 0:
    valid_improv = all_improvements[all_improvements['bounds_valid']]
    invalid_improv = all_improvements[~all_improvements['bounds_valid']]
    
    bins = np.linspace(0, all_improvements['loss_reduction_pct'].max()*1.1, 15)
    ax4.hist([valid_improv['loss_reduction_pct'], invalid_improv['loss_reduction_pct']], 
            bins=bins, label=['Valid Bounds', 'Invalid Bounds'], alpha=0.7)
    ax4.set_xlabel('Loss Reduction (%)')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Loss Improvements')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# 5. Model/Dataset breakdown (middle middle)
ax5 = fig.add_subplot(gs[1, 1])
if len(all_improvements) > 0:
    model_dataset_stats = all_improvements.groupby('model_dataset').agg({
        'loss_reduction_pct': 'mean',
        'error_reduction_pct': 'mean',
        'bounds_valid': 'sum'
    }).reset_index()
    
    x = np.arange(len(model_dataset_stats))
    width = 0.35
    
    ax5.bar(x - width/2, model_dataset_stats['loss_reduction_pct'], width, 
           label='Avg Loss Reduction %', alpha=0.8)
    ax5.bar(x + width/2, model_dataset_stats['error_reduction_pct'], width, 
           label='Avg Error Reduction %', alpha=0.8)
    
    ax5.set_xlabel('Model/Dataset')
    ax5.set_ylabel('Reduction (%)')
    ax5.set_title('Average Improvements by Model/Dataset')
    ax5.set_xticks(x)
    ax5.set_xticklabels(model_dataset_stats['model_dataset'], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

# 6. Epoch effects (middle right)
ax6 = fig.add_subplot(gs[1, 2])
if len(all_improvements) > 0:
    epoch_stats = all_improvements.groupby('epoch').agg({
        'loss_reduction_pct': ['mean', 'std', 'count']
    }).reset_index()
    
    epochs = epoch_stats['epoch']
    means = epoch_stats[('loss_reduction_pct', 'mean')]
    stds = epoch_stats[('loss_reduction_pct', 'std')]
    counts = epoch_stats[('loss_reduction_pct', 'count')]
    
    ax6.bar(epochs, means, alpha=0.7, color='skyblue')
    ax6.errorbar(epochs, means, yerr=stds, fmt='none', ecolor='black', capsize=5)
    
    # Add count labels
    for i, (e, m, c) in enumerate(zip(epochs, means, counts)):
        ax6.text(e, m + (stds.iloc[i] if not pd.isna(stds.iloc[i]) else 0) + 0.5, 
                f'n={int(c)}', ha='center', fontsize=8)
    
    ax6.set_xlabel('Training Epochs')
    ax6.set_ylabel('Average Loss Reduction (%)')
    ax6.set_title('Effect of Training Duration')
    ax6.set_xticks(epochs)
    ax6.grid(True, alpha=0.3, axis='y')

# 7. Bound validity check (bottom left)
ax7 = fig.add_subplot(gs[2, 0])
if len(successful) > 0:
    # Create scatter plot showing bound margins
    ax7.scatter(successful['vanilla_bound_ce_lhs'], successful['vanilla_bound_ce_rhs'], 
               s=100, alpha=0.6, label='Vanilla', marker='o')
    ax7.scatter(successful['channel_bound_ce_lhs'], successful['channel_bound_ce_rhs'], 
               s=100, alpha=0.6, label='Channel', marker='s')
    
    # Add diagonal line
    max_val = max(successful['vanilla_bound_ce_rhs'].max(), 
                 successful['channel_bound_ce_rhs'].max())
    ax7.plot([0, max_val], [0, max_val], 'r--', label='LHS = RHS', linewidth=2)
    
    ax7.set_xlabel('Bound LHS (Population Risk)')
    ax7.set_ylabel('Bound RHS (Upper Bound)')
    ax7.set_title('CE Bound Validation (Successful Cases)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

# 8. KL Penalty effects (bottom middle)
ax8 = fig.add_subplot(gs[2, 1])
if len(all_improvements) > 0:
    kl_stats = all_improvements.groupby('kl_penalty').agg({
        'loss_reduction_pct': ['mean', 'count'],
        'bounds_valid': 'sum'
    }).reset_index()
    
    x = np.arange(len(kl_stats))
    width = 0.35
    
    ax8.bar(x - width/2, kl_stats[('loss_reduction_pct', 'mean')], width, 
           label='Avg Loss Reduction %', alpha=0.8)
    ax8.bar(x + width/2, kl_stats[('bounds_valid', 'sum')], width, 
           label='Valid Bounds Count', alpha=0.8, color='green')
    
    ax8.set_xlabel('KL Penalty')
    ax8.set_ylabel('Value')
    ax8.set_title('KL Penalty Effects')
    ax8.set_xticks(x)
    ax8.set_xticklabels(kl_stats['kl_penalty'])
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

# 9. Key Insights (bottom right)
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

insights_text = """
KEY INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ WHAT WORKS:
  • Rayleigh-ZF channels
  • Low SNR (harsh channels)
  • Longer training (50 epochs)
  • Low KL penalty (0.01)
  • Frobenius norm

✗ WHAT DOESN'T:
  • BEC channels (0% success)
  • High KL penalty (bounds fail)
  • Spectral norm

⚠ CONCERNS:
  • 0-1 bounds often invalid
  • Only 3/100 configs succeed
  • Need bound refinement
"""

ax9.text(0.05, 0.95, insights_text, transform=ax9.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.savefig('visual_summary.png', dpi=300, bbox_inches='tight')
print("Saved visual_summary.png")

# Create a detailed table for the README
fig2, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

if len(all_improvements) > 0:
    # Get top 20 improvements
    top20 = all_improvements.nlargest(20, 'loss_reduction_pct')[
        ['model', 'dataset', 'channel_type', 'channel_spec', 'epoch', 'kl_penalty', 'norm',
         'loss_reduction_pct', 'error_reduction_pct', 'bounds_valid',
         'vanilla_loss', 'vanilla_error', 'channel_loss', 'channel_error']
    ].copy()
    
    # Round numeric columns
    top20['channel_spec'] = top20['channel_spec'].round(1)
    top20['loss_reduction_pct'] = top20['loss_reduction_pct'].round(2)
    top20['error_reduction_pct'] = top20['error_reduction_pct'].round(2)
    top20['vanilla_loss'] = top20['vanilla_loss'].round(4)
    top20['vanilla_error'] = top20['vanilla_error'].round(4)
    top20['channel_loss'] = top20['channel_loss'].round(4)
    top20['channel_error'] = top20['channel_error'].round(4)
    
    # Replace bounds_valid with checkmark/cross
    top20['Bounds'] = top20['bounds_valid'].map({True: '✓', False: '✗'})
    top20 = top20.drop('bounds_valid', axis=1)
    
    # Rename columns for display
    top20.columns = ['Model', 'Dataset', 'Channel', 'SNR(dB)', 'Epoch', 'KL', 'Norm',
                     'Loss↓%', 'Error↓%', 'V.Loss', 'V.Error', 'C.Loss', 'C.Error', 'Bounds']
    
    # Create table
    table = ax.table(cellText=top20.values, colLabels=top20.columns,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Color code the bounds column
    for i in range(len(top20)):
        cell = table[(i+1, 13)]  # Bounds column
        if top20.iloc[i]['Bounds'] == '✓':
            cell.set_facecolor('lightgreen')
        else:
            cell.set_facecolor('lightcoral')
    
    # Highlight header
    for j in range(len(top20.columns)):
        table[(0, j)].set_facecolor('lightblue')
        table[(0, j)].set_text_props(weight='bold')
    
    plt.title('Top 20 Configurations by Loss Reduction\n(V.=Vanilla, C.=Channel)', 
             fontsize=14, fontweight='bold', pad=20)

plt.savefig('detailed_table.png', dpi=300, bbox_inches='tight')
print("Saved detailed_table.png")

print("\nAll visualizations complete!")
