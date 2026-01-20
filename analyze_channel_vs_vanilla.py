#!/usr/bin/env python3
"""
Comprehensive analysis of channel-aware vs vanilla training results.

This script analyzes all experimental results to find configurations where:
1. Channel-aware training reduces population risk compared to vanilla
2. The derived bounds are valid (LHS < RHS)
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def parse_folder_name(folder_name):
    """Extract experiment parameters from folder name."""
    params = {}
    
    # Model type
    if folder_name.startswith('cnn-4'):
        params['model'] = 'CNN-4'
    elif folder_name.startswith('cnn-9'):
        params['model'] = 'CNN-9'
    elif folder_name.startswith('fcn-4'):
        params['model'] = 'FCN-4'
    else:
        params['model'] = 'unknown'
    
    # Dataset
    if 'mnist' in folder_name:
        params['dataset'] = 'MNIST'
    elif 'cifar10' in folder_name:
        params['dataset'] = 'CIFAR10'
    else:
        params['dataset'] = 'unknown'
    
    # Prior type
    if 'learnt_gaussian' in folder_name:
        params['prior'] = 'learnt'
    elif 'rand_gaussian' in folder_name:
        params['prior'] = 'random'
    else:
        params['prior'] = 'unknown'
    
    # Extract epoch
    epoch_match = re.search(r'epoch(\d+)_bs', folder_name)
    if epoch_match:
        params['epoch'] = int(epoch_match.group(1))
    
    # Objective type
    if 'objective-vanilla' in folder_name:
        params['objective'] = 'vanilla'
        params['channel_type'] = None
        params['channel_spec'] = None
        params['kl_penalty'] = None
    elif 'objective-channel' in folder_name:
        params['objective'] = 'channel'
        
        # Extract channel type and specification
        if 'bec-outage' in folder_name:
            params['channel_type'] = 'BEC'
            outage_match = re.search(r'bec-outage([\d.]+)', folder_name)
            if outage_match:
                params['channel_spec'] = float(outage_match.group(1))
                params['channel_spec_name'] = f'outage={params["channel_spec"]}'
        elif 'rayleigh-zf-tx' in folder_name:
            params['channel_type'] = 'Rayleigh-ZF'
            tx_match = re.search(r'tx([\d.]+)-noise([\d.]+)', folder_name)
            if tx_match:
                tx_power = float(tx_match.group(1))
                noise_var = float(tx_match.group(2))
                # Convert to SNR (dB)
                snr_linear = tx_power / noise_var
                snr_db = 10 * np.log10(snr_linear)
                params['channel_spec'] = snr_db
                params['channel_spec_name'] = f'SNR={snr_db:.1f}dB'
        
        # Extract KL penalty
        kl_match = re.search(r'-kl([\d.]+)', folder_name)
        if kl_match:
            params['kl_penalty'] = float(kl_match.group(1))
    
    return params


def parse_json_filename(json_filename):
    """Extract channel information from JSON filename."""
    info = {}
    
    if 'bec-outage' in json_filename:
        info['channel_type'] = 'BEC'
        outage_match = re.search(r'bec-outage([\d.]+)', json_filename)
        if outage_match:
            info['channel_spec'] = float(outage_match.group(1))
            info['channel_spec_name'] = f'outage={info["channel_spec"]}'
    elif 'rayleigh-zf-tx' in json_filename:
        info['channel_type'] = 'Rayleigh-ZF'
        tx_match = re.search(r'tx([\d.]+)-noise([\d.]+)', json_filename)
        if tx_match:
            tx_power = float(tx_match.group(1))
            noise_var = float(tx_match.group(2))
            # Convert to SNR (dB)
            snr_linear = tx_power / noise_var
            snr_db = 10 * np.log10(snr_linear)
            info['channel_spec'] = snr_db
            info['channel_spec_name'] = f'SNR={snr_db:.1f}dB'
    elif 'rayleigh-tx' in json_filename and 'rayleigh-zf' not in json_filename:
        # Skip non-ZF rayleigh
        return None
    
    # Extract norm type
    if 'norm-spec' in json_filename:
        info['norm'] = 'spectral'
    elif 'norm-frob' in json_filename:
        info['norm'] = 'frobenius'
    
    return info


def load_all_results(results_dir):
    """Load all experimental results."""
    all_results = []
    
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Parse folder parameters
        params = parse_folder_name(folder)
        if params['model'] == 'unknown':
            continue
        
        # Check bounds folder
        bounds_dir = os.path.join(folder_path, 'bounds')
        if not os.path.exists(bounds_dir):
            continue
        
        # Load all JSON files in bounds folder
        for json_file in os.listdir(bounds_dir):
            if not json_file.endswith('.json'):
                continue
            
            # Parse JSON filename
            json_info = parse_json_filename(json_file)
            if json_info is None:
                continue
            
            # Load JSON data
            json_path = os.path.join(bounds_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except:
                print(f"Failed to load {json_path}")
                continue
            
            # Combine all information
            result = {**params, **json_info, **data}
            result['folder'] = folder
            result['json_file'] = json_file
            
            all_results.append(result)
    
    return pd.DataFrame(all_results)


def find_successful_configurations(df):
    """Find configurations where channel-aware training helps."""
    
    # Group by model, dataset, epoch, channel_type, channel_spec, norm
    # Compare vanilla vs channel objectives
    
    successful_configs = []
    
    # Get all unique configurations (excluding objective and kl_penalty)
    config_cols = ['model', 'dataset', 'prior', 'epoch', 'channel_type', 'channel_spec', 'norm']
    
    for config, group in df.groupby(config_cols):
        # Get vanilla result
        vanilla = group[group['objective'] == 'vanilla']
        if len(vanilla) == 0:
            continue
        
        # Get channel results (may have multiple KL values)
        channel = group[group['objective'] == 'channel']
        if len(channel) == 0:
            continue
        
        vanilla_row = vanilla.iloc[0]
        
        # Check each channel configuration
        for _, channel_row in channel.iterrows():
            # Check if bounds are valid
            vanilla_ce_valid = vanilla_row['bound_ce_lhs'] < vanilla_row['bound_ce_rhs']
            vanilla_01_valid = vanilla_row['bound_01_lhs'] < vanilla_row['bound_01_rhs']
            channel_ce_valid = channel_row['bound_ce_lhs'] < channel_row['bound_ce_rhs']
            channel_01_valid = channel_row['bound_01_lhs'] < channel_row['bound_01_rhs']
            
            # Calculate improvements
            ce_improvement = vanilla_row['stochastic_loss'] - channel_row['stochastic_loss']
            error_improvement = vanilla_row['stochastic_01_error'] - channel_row['stochastic_01_error']
            
            ce_improvement_pct = (ce_improvement / vanilla_row['stochastic_loss']) * 100 if vanilla_row['stochastic_loss'] > 0 else 0
            error_improvement_pct = (error_improvement / vanilla_row['stochastic_01_error']) * 100 if vanilla_row['stochastic_01_error'] > 0 else 0
            
            # Check if channel training helps
            goal_satisfied = (ce_improvement > 0 or error_improvement > 0)
            
            successful_configs.append({
                'model': config[0],
                'dataset': config[1],
                'prior': config[2],
                'epoch': config[3],
                'channel_type': config[4],
                'channel_spec_name': vanilla_row['channel_spec_name'],
                'channel_spec': config[5],
                'norm': config[6],
                'kl_penalty': channel_row['kl_penalty'],
                
                'vanilla_ce_loss': vanilla_row['stochastic_loss'],
                'vanilla_01_error': vanilla_row['stochastic_01_error'],
                'channel_ce_loss': channel_row['stochastic_loss'],
                'channel_01_error': channel_row['stochastic_01_error'],
                
                'ce_improvement': ce_improvement,
                'error_improvement': error_improvement,
                'ce_improvement_pct': ce_improvement_pct,
                'error_improvement_pct': error_improvement_pct,
                
                'vanilla_ce_bound_valid': vanilla_ce_valid,
                'vanilla_01_bound_valid': vanilla_01_valid,
                'channel_ce_bound_valid': channel_ce_valid,
                'channel_01_bound_valid': channel_01_valid,
                'all_bounds_valid': vanilla_ce_valid and vanilla_01_valid and channel_ce_valid and channel_01_valid,
                
                'goal_satisfied': goal_satisfied,
                'goal_satisfied_with_valid_bounds': goal_satisfied and vanilla_ce_valid and vanilla_01_valid and channel_ce_valid and channel_01_valid,
                
                'vanilla_folder': vanilla_row['folder'],
                'channel_folder': channel_row['folder'],
            })
    
    return pd.DataFrame(successful_configs)


def create_summary_report(results_df, output_file='channel_vs_vanilla_summary.md'):
    """Create a comprehensive summary report."""
    
    with open(output_file, 'w') as f:
        f.write("# Channel-Aware vs Vanilla Training Analysis\n\n")
        f.write("## Executive Summary\n\n")
        
        total_configs = len(results_df)
        goal_satisfied = results_df['goal_satisfied'].sum()
        goal_satisfied_valid = results_df['goal_satisfied_with_valid_bounds'].sum()
        
        f.write(f"- **Total configurations analyzed**: {total_configs}\n")
        f.write(f"- **Configurations where channel training helps**: {goal_satisfied} ({100*goal_satisfied/total_configs:.1f}%)\n")
        f.write(f"- **Configurations with improvement AND valid bounds**: {goal_satisfied_valid} ({100*goal_satisfied_valid/total_configs:.1f}%)\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Filter for successful cases
        successful = results_df[results_df['goal_satisfied_with_valid_bounds']]
        
        if len(successful) > 0:
            f.write("### Best Improvements\n\n")
            
            # Top by cross-entropy improvement
            top_ce = successful.nlargest(10, 'ce_improvement_pct')
            f.write("#### Top 10 Cross-Entropy Loss Improvements\n\n")
            f.write("| Model | Dataset | Epoch | Channel | Spec | Norm | KL | CE Improve | Error Improve |\n")
            f.write("|-------|---------|-------|---------|------|------|----|-----------|--------------|\n")
            for _, row in top_ce.iterrows():
                f.write(f"| {row['model']} | {row['dataset']} | {row['epoch']} | {row['channel_type']} | "
                       f"{row['channel_spec_name']} | {row['norm']} | {row['kl_penalty']} | "
                       f"{row['ce_improvement_pct']:.2f}% | {row['error_improvement_pct']:.2f}% |\n")
            f.write("\n")
            
            # Top by error improvement
            top_err = successful.nlargest(10, 'error_improvement_pct')
            f.write("#### Top 10 Error Rate Improvements\n\n")
            f.write("| Model | Dataset | Epoch | Channel | Spec | Norm | KL | CE Improve | Error Improve |\n")
            f.write("|-------|---------|-------|---------|------|------|----|-----------|--------------|\n")
            for _, row in top_err.iterrows():
                f.write(f"| {row['model']} | {row['dataset']} | {row['epoch']} | {row['channel_type']} | "
                       f"{row['channel_spec_name']} | {row['norm']} | {row['kl_penalty']} | "
                       f"{row['ce_improvement_pct']:.2f}% | {row['error_improvement_pct']:.2f}% |\n")
            f.write("\n")
            
            # Analysis by factors
            f.write("### Success Rate by Configuration\n\n")
            
            # By model
            f.write("#### By Model\n\n")
            model_stats = successful.groupby('model').agg({
                'ce_improvement_pct': ['count', 'mean', 'std'],
                'error_improvement_pct': ['mean', 'std']
            }).round(2)
            f.write(model_stats.to_markdown())
            f.write("\n\n")
            
            # By dataset
            f.write("#### By Dataset\n\n")
            dataset_stats = successful.groupby('dataset').agg({
                'ce_improvement_pct': ['count', 'mean', 'std'],
                'error_improvement_pct': ['mean', 'std']
            }).round(2)
            f.write(dataset_stats.to_markdown())
            f.write("\n\n")
            
            # By channel type
            f.write("#### By Channel Type\n\n")
            channel_stats = successful.groupby('channel_type').agg({
                'ce_improvement_pct': ['count', 'mean', 'std'],
                'error_improvement_pct': ['mean', 'std']
            }).round(2)
            f.write(channel_stats.to_markdown())
            f.write("\n\n")
            
            # By norm type
            f.write("#### By Norm Type\n\n")
            norm_stats = successful.groupby('norm').agg({
                'ce_improvement_pct': ['count', 'mean', 'std'],
                'error_improvement_pct': ['mean', 'std']
            }).round(2)
            f.write(norm_stats.to_markdown())
            f.write("\n\n")
            
            # By KL penalty
            f.write("#### By KL Penalty\n\n")
            kl_stats = successful.groupby('kl_penalty').agg({
                'ce_improvement_pct': ['count', 'mean', 'std'],
                'error_improvement_pct': ['mean', 'std']
            }).round(2)
            f.write(kl_stats.to_markdown())
            f.write("\n\n")
            
        else:
            f.write("⚠️ **No configurations found where channel training helps with valid bounds.**\n\n")
        
        # Bound validity analysis
        f.write("## Bound Validity Analysis\n\n")
        f.write(f"- Vanilla CE bounds valid: {results_df['vanilla_ce_bound_valid'].sum()} / {total_configs}\n")
        f.write(f"- Vanilla 0-1 bounds valid: {results_df['vanilla_01_bound_valid'].sum()} / {total_configs}\n")
        f.write(f"- Channel CE bounds valid: {results_df['channel_ce_bound_valid'].sum()} / {total_configs}\n")
        f.write(f"- Channel 0-1 bounds valid: {results_df['channel_01_bound_valid'].sum()} / {total_configs}\n")
        f.write(f"- All bounds valid: {results_df['all_bounds_valid'].sum()} / {total_configs}\n\n")
        
        # Detailed results table
        f.write("## Detailed Results\n\n")
        f.write("Full results are available in `channel_vs_vanilla_detailed.csv`\n\n")
    
    print(f"Summary report saved to {output_file}")


def create_visualizations(results_df, output_prefix='channel_vs_vanilla'):
    """Create visualization plots."""
    
    successful = results_df[results_df['goal_satisfied_with_valid_bounds']]
    
    if len(successful) == 0:
        print("No successful configurations to visualize")
        return
    
    # Plot 1: Improvement distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # CE improvement histogram
    axes[0, 0].hist(successful['ce_improvement_pct'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('CE Loss Improvement (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of CE Loss Improvements')
    axes[0, 0].axvline(0, color='red', linestyle='--', label='No improvement')
    axes[0, 0].legend()
    
    # Error improvement histogram
    axes[0, 1].hist(successful['error_improvement_pct'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Error Rate Improvement (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Error Rate Improvements')
    axes[0, 1].axvline(0, color='red', linestyle='--', label='No improvement')
    axes[0, 1].legend()
    
    # Improvement by model
    model_data = successful.groupby('model').agg({
        'ce_improvement_pct': 'mean',
        'error_improvement_pct': 'mean'
    })
    x = np.arange(len(model_data))
    width = 0.35
    axes[1, 0].bar(x - width/2, model_data['ce_improvement_pct'], width, label='CE Loss', alpha=0.7)
    axes[1, 0].bar(x + width/2, model_data['error_improvement_pct'], width, label='Error Rate', alpha=0.7)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Average Improvement (%)')
    axes[1, 0].set_title('Average Improvement by Model')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_data.index)
    axes[1, 0].legend()
    
    # Improvement by KL penalty
    kl_data = successful.groupby('kl_penalty').agg({
        'ce_improvement_pct': 'mean',
        'error_improvement_pct': 'mean'
    })
    x = np.arange(len(kl_data))
    axes[1, 1].bar(x - width/2, kl_data['ce_improvement_pct'], width, label='CE Loss', alpha=0.7)
    axes[1, 1].bar(x + width/2, kl_data['error_improvement_pct'], width, label='Error Rate', alpha=0.7)
    axes[1, 1].set_xlabel('KL Penalty')
    axes[1, 1].set_ylabel('Average Improvement (%)')
    axes[1, 1].set_title('Average Improvement by KL Penalty')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(kl_data.index)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_improvements.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_improvements.pdf', bbox_inches='tight')
    print(f"Saved improvement plots to {output_prefix}_improvements.png/pdf")
    plt.close()
    
    # Plot 2: Scatter plot of vanilla vs channel performance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # CE Loss
    axes[0].scatter(successful['vanilla_ce_loss'], successful['channel_ce_loss'], 
                   c=successful['ce_improvement_pct'], cmap='RdYlGn', s=50, alpha=0.6)
    max_val = max(successful['vanilla_ce_loss'].max(), successful['channel_ce_loss'].max())
    axes[0].plot([0, max_val], [0, max_val], 'k--', label='Equal performance')
    axes[0].set_xlabel('Vanilla CE Loss')
    axes[0].set_ylabel('Channel-Aware CE Loss')
    axes[0].set_title('CE Loss: Vanilla vs Channel-Aware')
    axes[0].legend()
    cbar0 = plt.colorbar(axes[0].collections[0], ax=axes[0])
    cbar0.set_label('Improvement (%)')
    
    # Error Rate
    scatter = axes[1].scatter(successful['vanilla_01_error'], successful['channel_01_error'], 
                             c=successful['error_improvement_pct'], cmap='RdYlGn', s=50, alpha=0.6)
    max_val = max(successful['vanilla_01_error'].max(), successful['channel_01_error'].max())
    axes[1].plot([0, max_val], [0, max_val], 'k--', label='Equal performance')
    axes[1].set_xlabel('Vanilla Error Rate')
    axes[1].set_ylabel('Channel-Aware Error Rate')
    axes[1].set_title('Error Rate: Vanilla vs Channel-Aware')
    axes[1].legend()
    cbar1 = plt.colorbar(scatter, ax=axes[1])
    cbar1.set_label('Improvement (%)')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_scatter.pdf', bbox_inches='tight')
    print(f"Saved scatter plots to {output_prefix}_scatter.png/pdf")
    plt.close()
    
    # Plot 3: Heatmap by channel type and configuration
    if len(successful) > 10:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pivot for CE improvement
        pivot_ce = successful.pivot_table(
            values='ce_improvement_pct',
            index='channel_type',
            columns='channel_spec_name',
            aggfunc='mean'
        )
        sns.heatmap(pivot_ce, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0], center=0)
        axes[0].set_title('Average CE Improvement (%) by Channel Configuration')
        
        # Pivot for error improvement
        pivot_err = successful.pivot_table(
            values='error_improvement_pct',
            index='channel_type',
            columns='channel_spec_name',
            aggfunc='mean'
        )
        sns.heatmap(pivot_err, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1], center=0)
        axes[1].set_title('Average Error Improvement (%) by Channel Configuration')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_prefix}_heatmap.pdf', bbox_inches='tight')
        print(f"Saved heatmap to {output_prefix}_heatmap.png/pdf")
        plt.close()


def main():
    """Main analysis function."""
    results_dir = Path('results/posterior')
    
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return
    
    print("Loading all experimental results...")
    all_results = load_all_results(results_dir)
    print(f"Loaded {len(all_results)} result files")
    
    if len(all_results) == 0:
        print("No results found!")
        return
    
    print("\nAnalyzing configurations...")
    comparison_df = find_successful_configurations(all_results)
    print(f"Analyzed {len(comparison_df)} configuration comparisons")
    
    # Save detailed results
    comparison_df.to_csv('channel_vs_vanilla_detailed.csv', index=False)
    print("Saved detailed results to channel_vs_vanilla_detailed.csv")
    
    # Create summary report
    print("\nGenerating summary report...")
    create_summary_report(comparison_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(comparison_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - channel_vs_vanilla_summary.md (main report)")
    print("  - channel_vs_vanilla_detailed.csv (full data)")
    print("  - channel_vs_vanilla_improvements.png/pdf (improvement distributions)")
    print("  - channel_vs_vanilla_scatter.png/pdf (performance comparison)")
    print("  - channel_vs_vanilla_heatmap.png/pdf (configuration heatmap)")
    print("\nKey findings:")
    
    successful = comparison_df[comparison_df['goal_satisfied_with_valid_bounds']]
    if len(successful) > 0:
        print(f"  ✓ Found {len(successful)} configurations where channel training helps!")
        print(f"  ✓ Average CE improvement: {successful['ce_improvement_pct'].mean():.2f}%")
        print(f"  ✓ Average error improvement: {successful['error_improvement_pct'].mean():.2f}%")
        print(f"\n  Best configuration:")
        best = successful.loc[successful['ce_improvement_pct'].idxmax()]
        print(f"    - Model: {best['model']}, Dataset: {best['dataset']}")
        print(f"    - Channel: {best['channel_type']} ({best['channel_spec_name']})")
        print(f"    - Epoch: {best['epoch']}, KL: {best['kl_penalty']}, Norm: {best['norm']}")
        print(f"    - CE improvement: {best['ce_improvement_pct']:.2f}%")
        print(f"    - Error improvement: {best['error_improvement_pct']:.2f}%")
    else:
        print("  ⚠️ No configurations found where channel training helps with valid bounds")
        invalid_bounds = comparison_df[~comparison_df['all_bounds_valid']]
        print(f"  ℹ️ {len(invalid_bounds)} configurations have invalid bounds")


if __name__ == '__main__':
    main()
