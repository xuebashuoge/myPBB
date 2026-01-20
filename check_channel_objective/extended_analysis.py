#!/usr/bin/env python3
"""
Extended analysis to understand all configurations where channel objective helps,
regardless of bound validity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('all_comparisons.csv')

print("="*80)
print("EXTENDED ANALYSIS: All Cases Where Channel Objective Helps")
print("="*80)

# Filter for goal satisfied (regardless of bounds)
goal_satisfied = df[df['goal_satisfied']].copy()
goal_satisfied['model_dataset'] = goal_satisfied['model'] + '_' + goal_satisfied['dataset']

print(f"\nTotal configurations where channel reduces both loss and error: {len(goal_satisfied)}")
print(f"Of these, {(goal_satisfied['bounds_valid']).sum()} have valid bounds")
print(f"And {(~goal_satisfied['bounds_valid']).sum()} have invalid bounds")

print("\n" + "="*80)
print("BREAKDOWN BY CHANNEL TYPE")
print("="*80)

for channel_type in df['channel_type'].unique():
    subset = goal_satisfied[goal_satisfied['channel_type'] == channel_type]
    print(f"\n{channel_type.upper()}:")
    print(f"  Total improvements: {len(subset)}")
    print(f"  With valid bounds: {subset['bounds_valid'].sum()}")
    print(f"  Avg loss reduction: {subset['loss_reduction_pct'].mean():.2f}%")
    print(f"  Avg error reduction: {subset['error_reduction_pct'].mean():.2f}%")
    
    if len(subset) > 0:
        print(f"\n  By Model/Dataset:")
        for model_dataset in subset['model_dataset'].unique():
            subset2 = subset[subset['model_dataset'] == model_dataset]
            print(f"    {model_dataset}: {len(subset2)} configs, "
                  f"avg loss reduction: {subset2['loss_reduction_pct'].mean():.2f}%, "
                  f"avg error reduction: {subset2['error_reduction_pct'].mean():.2f}%")

print("\n" + "="*80)
print("WHY BOUNDS ARE INVALID")
print("="*80)

invalid_bounds = goal_satisfied[~goal_satisfied['bounds_valid']].copy()
print(f"\nAnalyzing {len(invalid_bounds)} configurations with goal satisfied but invalid bounds:")

# Check which bound is invalid
invalid_bounds['ce_bound_invalid'] = invalid_bounds['channel_bound_ce_lhs'] >= invalid_bounds['channel_bound_ce_rhs']
invalid_bounds['01_bound_invalid'] = invalid_bounds['channel_bound_01_lhs'] >= invalid_bounds['channel_bound_01_rhs']

print(f"\nCE bound invalid: {invalid_bounds['ce_bound_invalid'].sum()}")
print(f"0-1 bound invalid: {invalid_bounds['01_bound_invalid'].sum()}")
print(f"Both invalid: {(invalid_bounds['ce_bound_invalid'] & invalid_bounds['01_bound_invalid']).sum()}")

print("\nLooking at bound margins:")
invalid_bounds['ce_margin'] = invalid_bounds['channel_bound_ce_rhs'] - invalid_bounds['channel_bound_ce_lhs']
invalid_bounds['01_margin'] = invalid_bounds['channel_bound_01_rhs'] - invalid_bounds['channel_bound_01_lhs']

print(f"\nCE bound margin stats:")
print(f"  Mean: {invalid_bounds['ce_margin'].mean():.6f}")
print(f"  Median: {invalid_bounds['ce_margin'].median():.6f}")
print(f"  Min: {invalid_bounds['ce_margin'].min():.6f}")
print(f"  Max: {invalid_bounds['ce_margin'].max():.6f}")

print(f"\n0-1 bound margin stats:")
print(f"  Mean: {invalid_bounds['01_margin'].mean():.6f}")
print(f"  Median: {invalid_bounds['01_margin'].median():.6f}")
print(f"  Min: {invalid_bounds['01_margin'].min():.6f}")
print(f"  Max: {invalid_bounds['01_margin'].max():.6f}")

print("\n" + "="*80)
print("TOP IMPROVEMENTS (REGARDLESS OF BOUND VALIDITY)")
print("="*80)

print("\nTop 20 by Loss Reduction:")
top20_loss = goal_satisfied.nlargest(20, 'loss_reduction_pct')[
    ['model', 'dataset', 'channel_type', 'channel_spec', 'epoch', 'kl_penalty', 'norm',
     'loss_reduction_pct', 'error_reduction_pct', 'bounds_valid']
]
print(top20_loss.to_string(index=False))

print("\nTop 20 by Error Reduction:")
top20_error = goal_satisfied.nlargest(20, 'error_reduction_pct')[
    ['model', 'dataset', 'channel_type', 'channel_spec', 'epoch', 'kl_penalty', 'norm',
     'loss_reduction_pct', 'error_reduction_pct', 'bounds_valid']
]
print(top20_error.to_string(index=False))

# Save extended results
goal_satisfied.to_csv('all_improvements.csv', index=False)
invalid_bounds.to_csv('improvements_with_invalid_bounds.csv', index=False)

print(f"\n\nSaved extended results to:")
print(f"  - all_improvements.csv ({len(goal_satisfied)} configurations)")
print(f"  - improvements_with_invalid_bounds.csv ({len(invalid_bounds)} configurations)")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Loss reduction distribution with bound validity
ax = axes[0, 0]
valid = goal_satisfied[goal_satisfied['bounds_valid']]
invalid = goal_satisfied[~goal_satisfied['bounds_valid']]
ax.hist([valid['loss_reduction_pct'], invalid['loss_reduction_pct']], 
        label=['Valid bounds', 'Invalid bounds'], alpha=0.7, bins=15)
ax.set_xlabel('Loss Reduction (%)')
ax.set_ylabel('Count')
ax.set_title('Loss Reduction: Valid vs Invalid Bounds')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Error reduction distribution with bound validity
ax = axes[0, 1]
ax.hist([valid['error_reduction_pct'], invalid['error_reduction_pct']], 
        label=['Valid bounds', 'Invalid bounds'], alpha=0.7, bins=15)
ax.set_xlabel('Error Reduction (%)')
ax.set_ylabel('Count')
ax.set_title('Error Reduction: Valid vs Invalid Bounds')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Scatter plot of loss vs error reduction
ax = axes[1, 0]
ax.scatter(valid['loss_reduction_pct'], valid['error_reduction_pct'], 
          alpha=0.6, label='Valid bounds', s=100)
ax.scatter(invalid['loss_reduction_pct'], invalid['error_reduction_pct'], 
          alpha=0.6, label='Invalid bounds', s=100, marker='x')
ax.set_xlabel('Loss Reduction (%)')
ax.set_ylabel('Error Reduction (%)')
ax.set_title('Loss vs Error Reduction')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Channel type breakdown
ax = axes[1, 1]
breakdown = goal_satisfied.groupby(['channel_type', 'bounds_valid']).size().unstack(fill_value=0)
breakdown.plot(kind='bar', ax=ax)
ax.set_xlabel('Channel Type')
ax.set_ylabel('Count')
ax.set_title('Improvements by Channel Type and Bound Validity')
ax.legend(['Invalid Bounds', 'Valid Bounds'])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('extended_analysis.png', dpi=300, bbox_inches='tight')
print(f"  - extended_analysis.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
