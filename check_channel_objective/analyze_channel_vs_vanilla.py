#!/usr/bin/env python3
"""
Analyze channel objective vs vanilla objective performance.

This script compares the population risk (stochastic_loss/01_error) between
vanilla and channel objectives across different configurations.

Goal: Find configurations where:
1. Vanilla objective has large population risk in wireless environment
2. Channel objective reduces this population risk
3. Verify that derived bounds are valid (lhs < rhs)
"""

import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def parse_folder_name(folder_name):
    """Parse folder name to extract configuration parameters."""
    parts = {}
    
    # Extract model type
    if folder_name.startswith('cnn-4'):
        parts['model'] = 'cnn-4'
    elif folder_name.startswith('cnn-9'):
        parts['model'] = 'cnn-9'
    elif folder_name.startswith('fcn-4'):
        parts['model'] = 'fcn-4'
    else:
        parts['model'] = 'unknown'
    
    # Extract dataset
    if '_mnist_' in folder_name:
        parts['dataset'] = 'mnist'
    elif '_cifar10_' in folder_name:
        parts['dataset'] = 'cifar10'
    else:
        parts['dataset'] = 'unknown'
    
    # Extract prior type
    if '_learnt_' in folder_name:
        parts['prior'] = 'learnt'
    elif '_rand_' in folder_name:
        parts['prior'] = 'rand'
    else:
        parts['prior'] = 'unknown'
    
    # Extract epoch
    epoch_match = re.search(r'epoch(\d+)_bs', folder_name)
    if epoch_match:
        parts['epoch'] = int(epoch_match.group(1))
    
    # Extract objective type
    if 'objective-vanilla' in folder_name:
        parts['objective'] = 'vanilla'
        parts['channel_type'] = None
        parts['channel_spec'] = None
        parts['kl_penalty'] = None
    elif 'objective-channel' in folder_name:
        parts['objective'] = 'channel'
        
        # Extract channel type and specification
        if 'bec-outage' in folder_name:
            parts['channel_type'] = 'bec'
            outage_match = re.search(r'bec-outage([\d.]+)', folder_name)
            if outage_match:
                parts['channel_spec'] = float(outage_match.group(1))
        elif 'rayleigh-zf-tx' in folder_name:
            parts['channel_type'] = 'rayleigh-zf'
            tx_match = re.search(r'tx([\d.]+)-noise([\d.]+)', folder_name)
            if tx_match:
                tx_power = float(tx_match.group(1))
                noise_var = float(tx_match.group(2))
                # Convert to SNR (dB)
                snr_linear = tx_power / noise_var
                snr_db = 10 * np.log10(snr_linear)
                parts['channel_spec'] = snr_db
                parts['tx_power'] = tx_power
                parts['noise_var'] = noise_var
        
        # Extract KL penalty
        kl_match = re.search(r'-kl([\d.]+)', folder_name)
        if kl_match:
            parts['kl_penalty'] = float(kl_match.group(1))
    
    return parts


def parse_json_filename(json_filename):
    """Parse JSON filename to extract channel configuration."""
    parts = {}
    
    # Extract norm type
    if '_norm-frob_' in json_filename:
        parts['norm'] = 'frob'
    elif '_norm-spec_' in json_filename:
        parts['norm'] = 'spec'
    
    # Extract channel type and specification
    if json_filename.startswith('bec-outage'):
        parts['channel_type'] = 'bec'
        outage_match = re.search(r'bec-outage([\d.]+)', json_filename)
        if outage_match:
            parts['channel_spec'] = float(outage_match.group(1))
    elif json_filename.startswith('rayleigh-zf-tx'):
        parts['channel_type'] = 'rayleigh-zf'
        tx_match = re.search(r'tx([\d.]+)-noise([\d.]+)', json_filename)
        if tx_match:
            tx_power = float(tx_match.group(1))
            noise_var = float(tx_match.group(2))
            # Convert to SNR (dB)
            snr_linear = tx_power / noise_var
            snr_db = 10 * np.log10(snr_linear)
            parts['channel_spec'] = snr_db
            parts['tx_power'] = tx_power
            parts['noise_var'] = noise_var
    
    return parts


def load_results(posterior_dir):
    """Load all results from posterior directory."""
    results = []
    
    for folder_name in os.listdir(posterior_dir):
        folder_path = os.path.join(posterior_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        bounds_dir = os.path.join(folder_path, 'bounds')
        if not os.path.exists(bounds_dir):
            continue
        
        folder_config = parse_folder_name(folder_name)
        
        # Load JSON files from bounds directory
        for json_file in os.listdir(bounds_dir):
            if not json_file.endswith('.json'):
                continue
            
            # Skip files with only 'rayleigh' (not rayleigh-zf)
            if json_file.startswith('rayleigh-tx') and 'rayleigh-zf' not in json_file:
                continue
            
            json_path = os.path.join(bounds_dir, json_file)
            json_config = parse_json_filename(json_file)
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Combine configurations
                result = {**folder_config, **json_config, **data}
                result['folder_name'] = folder_name
                result['json_file'] = json_file
                results.append(result)
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
    
    return pd.DataFrame(results)


def find_vanilla_baseline(df, model, dataset, prior, epoch, channel_type, channel_spec, norm):
    """Find vanilla baseline for given configuration."""
    vanilla_df = df[
        (df['objective'] == 'vanilla') &
        (df['model'] == model) &
        (df['dataset'] == dataset) &
        (df['prior'] == prior) &
        (df['epoch'] == epoch) &
        (df['channel_type'] == channel_type) &
        (df['channel_spec'] == channel_spec) &
        (df['norm'] == norm)
    ]
    
    if len(vanilla_df) == 0:
        return None
    elif len(vanilla_df) == 1:
        return vanilla_df.iloc[0]
    else:
        # If multiple, return first one
        return vanilla_df.iloc[0]


def analyze_improvements(df):
    """Analyze where channel objective improves over vanilla."""
    improvements = []
    
    # Get all channel objective results
    channel_df = df[df['objective'] == 'channel'].copy()
    
    for idx, row in channel_df.iterrows():
        # Find corresponding vanilla baseline
        vanilla = find_vanilla_baseline(
            df, 
            row['model'], 
            row['dataset'], 
            row['prior'], 
            row['epoch'],
            row['channel_type'],
            row['channel_spec'],
            row['norm']
        )
        
        if vanilla is None:
            continue
        
        # Calculate improvements
        improvement = {
            'model': row['model'],
            'dataset': row['dataset'],
            'prior': row['prior'],
            'epoch': row['epoch'],
            'channel_type': row['channel_type'],
            'channel_spec': row['channel_spec'],
            'norm': row['norm'],
            'kl_penalty': row['kl_penalty'],
            
            # Vanilla metrics
            'vanilla_loss': vanilla['stochastic_loss'],
            'vanilla_error': vanilla['stochastic_01_error'],
            'vanilla_bound_ce_lhs': vanilla['bound_ce_lhs'],
            'vanilla_bound_ce_rhs': vanilla['bound_ce_rhs'],
            'vanilla_bound_01_lhs': vanilla['bound_01_lhs'],
            'vanilla_bound_01_rhs': vanilla['bound_01_rhs'],
            'vanilla_bound_ce_valid': vanilla['bound_ce_lhs'] < vanilla['bound_ce_rhs'],
            'vanilla_bound_01_valid': vanilla['bound_01_lhs'] < vanilla['bound_01_rhs'],
            
            # Channel metrics
            'channel_loss': row['stochastic_loss'],
            'channel_error': row['stochastic_01_error'],
            'channel_bound_ce_lhs': row['bound_ce_lhs'],
            'channel_bound_ce_rhs': row['bound_ce_rhs'],
            'channel_bound_01_lhs': row['bound_01_lhs'],
            'channel_bound_01_rhs': row['bound_01_rhs'],
            'channel_bound_ce_valid': row['bound_ce_lhs'] < row['bound_ce_rhs'],
            'channel_bound_01_valid': row['bound_01_lhs'] < row['bound_01_rhs'],
            
            # Improvements
            'loss_reduction': vanilla['stochastic_loss'] - row['stochastic_loss'],
            'error_reduction': vanilla['stochastic_01_error'] - row['stochastic_01_error'],
            'loss_reduction_pct': 100 * (vanilla['stochastic_loss'] - row['stochastic_loss']) / vanilla['stochastic_loss'],
            'error_reduction_pct': 100 * (vanilla['stochastic_01_error'] - row['stochastic_01_error']) / vanilla['stochastic_01_error'],
            
            # Check if goal is satisfied
            'goal_satisfied': (
                (vanilla['stochastic_loss'] > row['stochastic_loss']) and
                (vanilla['stochastic_01_error'] > row['stochastic_01_error'])
            ),
            
            # Check if bounds are valid
            'bounds_valid': (
                row['bound_ce_lhs'] < row['bound_ce_rhs'] and
                row['bound_01_lhs'] < row['bound_01_rhs']
            ),
            
            # Additional metrics
            'channel_term': row.get('channel_term', np.nan),
            'kl_final': row.get('kl_final', np.nan),
            'empirical_nll_loss': row.get('empirical_nll_loss', np.nan),
            'empirical_01_error': row.get('empirical_01_error', np.nan),
        }
        
        improvements.append(improvement)
    
    return pd.DataFrame(improvements)


def create_summary_plots(improvements_df, output_dir):
    """Create summary plots of the analysis."""
    
    # Filter for successful improvements
    successful = improvements_df[
        improvements_df['goal_satisfied'] & 
        improvements_df['bounds_valid']
    ].copy()
    
    print(f"\nFound {len(successful)} configurations where goal is satisfied with valid bounds")
    print(f"Out of {len(improvements_df)} total channel configurations analyzed")
    
    # Plot 1: Loss reduction by configuration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1a: Loss reduction by channel type
    ax = axes[0, 0]
    if len(successful) > 0:
        successful_grouped = successful.groupby('channel_type')['loss_reduction_pct'].apply(list)
        for channel_type, values in successful_grouped.items():
            ax.hist(values, alpha=0.6, label=channel_type, bins=20)
        ax.set_xlabel('Loss Reduction (%)')
        ax.set_ylabel('Count')
        ax.set_title('Loss Reduction by Channel Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 1b: Error reduction by channel type
    ax = axes[0, 1]
    if len(successful) > 0:
        successful_grouped = successful.groupby('channel_type')['error_reduction_pct'].apply(list)
        for channel_type, values in successful_grouped.items():
            ax.hist(values, alpha=0.6, label=channel_type, bins=20)
        ax.set_xlabel('Error Reduction (%)')
        ax.set_ylabel('Count')
        ax.set_title('Error Reduction by Channel Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 1c: Loss reduction by model/dataset
    ax = axes[1, 0]
    if len(successful) > 0:
        successful['model_dataset'] = successful['model'] + '_' + successful['dataset']
        pivot = successful.pivot_table(
            values='loss_reduction_pct',
            index='model_dataset',
            columns='channel_type',
            aggfunc='mean'
        )
        pivot.plot(kind='bar', ax=ax)
        ax.set_xlabel('Model_Dataset')
        ax.set_ylabel('Average Loss Reduction (%)')
        ax.set_title('Average Loss Reduction by Model and Channel Type')
        ax.legend(title='Channel Type')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 1d: Error reduction by model/dataset
    ax = axes[1, 1]
    if len(successful) > 0:
        pivot = successful.pivot_table(
            values='error_reduction_pct',
            index='model_dataset',
            columns='channel_type',
            aggfunc='mean'
        )
        pivot.plot(kind='bar', ax=ax)
        ax.set_xlabel('Model_Dataset')
        ax.set_ylabel('Average Error Reduction (%)')
        ax.set_title('Average Error Reduction by Model and Channel Type')
        ax.legend(title='Channel Type')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_improvements.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Channel specification effects
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # BEC outage analysis
    bec_successful = successful[successful['channel_type'] == 'bec']
    if len(bec_successful) > 0:
        ax = axes[0, 0]
        for model_dataset in bec_successful['model_dataset'].unique():
            subset = bec_successful[bec_successful['model_dataset'] == model_dataset]
            subset_sorted = subset.sort_values('channel_spec')
            ax.plot(subset_sorted['channel_spec'], subset_sorted['loss_reduction_pct'], 
                   marker='o', label=model_dataset)
        ax.set_xlabel('BEC Outage Probability')
        ax.set_ylabel('Loss Reduction (%)')
        ax.set_title('BEC: Loss Reduction vs Outage Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        for model_dataset in bec_successful['model_dataset'].unique():
            subset = bec_successful[bec_successful['model_dataset'] == model_dataset]
            subset_sorted = subset.sort_values('channel_spec')
            ax.plot(subset_sorted['channel_spec'], subset_sorted['error_reduction_pct'], 
                   marker='o', label=model_dataset)
        ax.set_xlabel('BEC Outage Probability')
        ax.set_ylabel('Error Reduction (%)')
        ax.set_title('BEC: Error Reduction vs Outage Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Rayleigh-ZF SNR analysis
    rayleigh_successful = successful[successful['channel_type'] == 'rayleigh-zf']
    if len(rayleigh_successful) > 0:
        ax = axes[1, 0]
        for model_dataset in rayleigh_successful['model_dataset'].unique():
            subset = rayleigh_successful[rayleigh_successful['model_dataset'] == model_dataset]
            subset_sorted = subset.sort_values('channel_spec')
            ax.plot(subset_sorted['channel_spec'], subset_sorted['loss_reduction_pct'], 
                   marker='s', label=model_dataset)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Loss Reduction (%)')
        ax.set_title('Rayleigh-ZF: Loss Reduction vs SNR')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        for model_dataset in rayleigh_successful['model_dataset'].unique():
            subset = rayleigh_successful[rayleigh_successful['model_dataset'] == model_dataset]
            subset_sorted = subset.sort_values('channel_spec')
            ax.plot(subset_sorted['channel_spec'], subset_sorted['error_reduction_pct'], 
                   marker='s', label=model_dataset)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Error Reduction (%)')
        ax.set_title('Rayleigh-ZF: Error Reduction vs SNR')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_spec_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: KL penalty effects
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if len(successful) > 0:
        # Loss reduction vs KL penalty
        ax = axes[0, 0]
        for channel_type in successful['channel_type'].unique():
            subset = successful[successful['channel_type'] == channel_type]
            for model_dataset in subset['model_dataset'].unique():
                subset2 = subset[subset['model_dataset'] == model_dataset]
                subset2_sorted = subset2.sort_values('kl_penalty')
                ax.plot(subset2_sorted['kl_penalty'], subset2_sorted['loss_reduction_pct'],
                       marker='o', label=f'{channel_type}_{model_dataset}')
        ax.set_xlabel('KL Penalty')
        ax.set_ylabel('Loss Reduction (%)')
        ax.set_title('Loss Reduction vs KL Penalty')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error reduction vs KL penalty
        ax = axes[0, 1]
        for channel_type in successful['channel_type'].unique():
            subset = successful[successful['channel_type'] == channel_type]
            for model_dataset in subset['model_dataset'].unique():
                subset2 = subset[subset['model_dataset'] == model_dataset]
                subset2_sorted = subset2.sort_values('kl_penalty')
                ax.plot(subset2_sorted['kl_penalty'], subset2_sorted['error_reduction_pct'],
                       marker='o', label=f'{channel_type}_{model_dataset}')
        ax.set_xlabel('KL Penalty')
        ax.set_ylabel('Error Reduction (%)')
        ax.set_title('Error Reduction vs KL Penalty')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Epoch effects
        ax = axes[1, 0]
        epoch_grouped = successful.groupby('epoch').agg({
            'loss_reduction_pct': 'mean',
            'error_reduction_pct': 'mean'
        })
        epoch_grouped.plot(kind='bar', ax=ax)
        ax.set_xlabel('Training Epochs')
        ax.set_ylabel('Average Reduction (%)')
        ax.set_title('Average Improvement by Training Epochs')
        ax.legend(['Loss Reduction', 'Error Reduction'])
        ax.grid(True, alpha=0.3)
        
        # Norm type comparison
        ax = axes[1, 1]
        norm_grouped = successful.groupby('norm').agg({
            'loss_reduction_pct': 'mean',
            'error_reduction_pct': 'mean'
        })
        norm_grouped.plot(kind='bar', ax=ax)
        ax.set_xlabel('Norm Type')
        ax.set_ylabel('Average Reduction (%)')
        ax.set_title('Average Improvement by Norm Type')
        ax.legend(['Loss Reduction', 'Error Reduction'])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Bound validation
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if len(successful) > 0:
        # CE bound: LHS vs RHS for vanilla
        ax = axes[0, 0]
        ax.scatter(successful['vanilla_bound_ce_lhs'], successful['vanilla_bound_ce_rhs'], 
                  alpha=0.6, label='Vanilla')
        max_val = max(successful['vanilla_bound_ce_rhs'].max(), 
                     successful['vanilla_bound_ce_lhs'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='LHS = RHS')
        ax.set_xlabel('Bound CE LHS (Population Risk)')
        ax.set_ylabel('Bound CE RHS (Upper Bound)')
        ax.set_title('Vanilla: CE Bound Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # CE bound: LHS vs RHS for channel
        ax = axes[0, 1]
        ax.scatter(successful['channel_bound_ce_lhs'], successful['channel_bound_ce_rhs'], 
                  alpha=0.6, label='Channel', color='orange')
        max_val = max(successful['channel_bound_ce_rhs'].max(), 
                     successful['channel_bound_ce_lhs'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='LHS = RHS')
        ax.set_xlabel('Bound CE LHS (Population Risk)')
        ax.set_ylabel('Bound CE RHS (Upper Bound)')
        ax.set_title('Channel: CE Bound Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 01 bound: LHS vs RHS for vanilla
        ax = axes[1, 0]
        ax.scatter(successful['vanilla_bound_01_lhs'], successful['vanilla_bound_01_rhs'], 
                  alpha=0.6, label='Vanilla')
        max_val = max(successful['vanilla_bound_01_rhs'].max(), 
                     successful['vanilla_bound_01_lhs'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='LHS = RHS')
        ax.set_xlabel('Bound 01 LHS (Population Error)')
        ax.set_ylabel('Bound 01 RHS (Upper Bound)')
        ax.set_title('Vanilla: 0-1 Bound Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 01 bound: LHS vs RHS for channel
        ax = axes[1, 1]
        ax.scatter(successful['channel_bound_01_lhs'], successful['channel_bound_01_rhs'], 
                  alpha=0.6, label='Channel', color='orange')
        max_val = max(successful['channel_bound_01_rhs'].max(), 
                     successful['channel_bound_01_lhs'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='LHS = RHS')
        ax.set_xlabel('Bound 01 LHS (Population Error)')
        ax.set_ylabel('Bound 01 RHS (Upper Bound)')
        ax.set_title('Channel: 0-1 Bound Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bound_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis function."""
    
    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'results' / 'posterior'
    output_dir = script_dir
    
    print("="*80)
    print("Channel vs Vanilla Objective Analysis")
    print("="*80)
    
    # Load all results
    print("\nLoading results...")
    df = load_results(results_dir)
    print(f"Loaded {len(df)} result files")
    
    # Basic statistics
    print("\n" + "="*80)
    print("Basic Statistics")
    print("="*80)
    print(f"\nObjective types:")
    print(df['objective'].value_counts())
    print(f"\nModels:")
    print(df['model'].value_counts())
    print(f"\nDatasets:")
    print(df['dataset'].value_counts())
    print(f"\nChannel types:")
    print(df[df['objective'] == 'channel']['channel_type'].value_counts())
    print(f"\nNorm types:")
    print(df['norm'].value_counts())
    
    # Analyze improvements
    print("\n" + "="*80)
    print("Analyzing Improvements")
    print("="*80)
    improvements_df = analyze_improvements(df)
    
    # Filter for successful cases
    successful = improvements_df[
        improvements_df['goal_satisfied'] & 
        improvements_df['bounds_valid']
    ]
    
    print(f"\nTotal channel configurations: {len(improvements_df)}")
    print(f"Configurations where goal is satisfied: {len(improvements_df[improvements_df['goal_satisfied']])}")
    print(f"Configurations with valid bounds: {len(improvements_df[improvements_df['bounds_valid']])}")
    print(f"Configurations meeting both criteria: {len(successful)}")
    
    # Save detailed results
    improvements_df.to_csv(output_dir / 'all_comparisons.csv', index=False)
    successful.to_csv(output_dir / 'successful_improvements.csv', index=False)
    
    print(f"\nSaved detailed results to:")
    print(f"  - {output_dir / 'all_comparisons.csv'}")
    print(f"  - {output_dir / 'successful_improvements.csv'}")
    
    # Print summary statistics for successful cases
    if len(successful) > 0:
        print("\n" + "="*80)
        print("Successful Configurations Summary")
        print("="*80)
        
        print("\nBy Channel Type:")
        summary = successful.groupby('channel_type').agg({
            'loss_reduction_pct': ['mean', 'std', 'min', 'max'],
            'error_reduction_pct': ['mean', 'std', 'min', 'max']
        })
        print(summary)
        
        print("\nBy Model and Dataset:")
        successful['model_dataset'] = successful['model'] + '_' + successful['dataset']
        summary = successful.groupby('model_dataset').agg({
            'loss_reduction_pct': ['mean', 'std', 'count'],
            'error_reduction_pct': ['mean', 'std', 'count']
        })
        print(summary)
        
        print("\nBy Epoch:")
        summary = successful.groupby('epoch').agg({
            'loss_reduction_pct': ['mean', 'std', 'count'],
            'error_reduction_pct': ['mean', 'std', 'count']
        })
        print(summary)
        
        print("\nBy KL Penalty:")
        summary = successful.groupby('kl_penalty').agg({
            'loss_reduction_pct': ['mean', 'std', 'count'],
            'error_reduction_pct': ['mean', 'std', 'count']
        })
        print(summary)
        
        print("\nTop 10 Configurations by Loss Reduction:")
        top10_loss = successful.nlargest(10, 'loss_reduction_pct')[
            ['model', 'dataset', 'channel_type', 'channel_spec', 'epoch', 'kl_penalty', 'norm',
             'vanilla_loss', 'channel_loss', 'loss_reduction_pct']
        ]
        print(top10_loss.to_string())
        
        print("\nTop 10 Configurations by Error Reduction:")
        top10_error = successful.nlargest(10, 'error_reduction_pct')[
            ['model', 'dataset', 'channel_type', 'channel_spec', 'epoch', 'kl_penalty', 'norm',
             'vanilla_error', 'channel_error', 'error_reduction_pct']
        ]
        print(top10_error.to_string())
    
    # Create plots
    print("\n" + "="*80)
    print("Creating Plots")
    print("="*80)
    create_summary_plots(improvements_df, output_dir)
    print(f"\nSaved plots to:")
    print(f"  - {output_dir / 'summary_improvements.png'}")
    print(f"  - {output_dir / 'channel_spec_effects.png'}")
    print(f"  - {output_dir / 'parameter_effects.png'}")
    print(f"  - {output_dir / 'bound_validation.png'}")
    
    # Create a summary report
    print("\n" + "="*80)
    print("Creating Summary Report")
    print("="*80)
    
    report_path = output_dir / 'ANALYSIS_REPORT.md'
    with open(report_path, 'w') as f:
        f.write("# Channel vs Vanilla Objective Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total configurations analyzed**: {len(improvements_df)}\n")
        f.write(f"- **Configurations where channel outperforms vanilla**: {len(improvements_df[improvements_df['goal_satisfied']])}\n")
        f.write(f"- **Configurations with valid bounds**: {len(improvements_df[improvements_df['bounds_valid']])}\n")
        f.write(f"- **Successful configurations (both criteria)**: {len(successful)}\n\n")
        
        if len(successful) > 0:
            f.write("## Key Findings\n\n")
            f.write(f"### Average Improvements\n\n")
            f.write(f"- **Average loss reduction**: {successful['loss_reduction_pct'].mean():.2f}%\n")
            f.write(f"- **Average error reduction**: {successful['error_reduction_pct'].mean():.2f}%\n")
            f.write(f"- **Maximum loss reduction**: {successful['loss_reduction_pct'].max():.2f}%\n")
            f.write(f"- **Maximum error reduction**: {successful['error_reduction_pct'].max():.2f}%\n\n")
            
            f.write("### Performance by Channel Type\n\n")
            summary = successful.groupby('channel_type').agg({
                'loss_reduction_pct': 'mean',
                'error_reduction_pct': 'mean'
            })
            f.write(summary.to_markdown())
            f.write("\n\n")
            
            f.write("### Performance by Model and Dataset\n\n")
            summary = successful.groupby('model_dataset').agg({
                'loss_reduction_pct': 'mean',
                'error_reduction_pct': 'mean'
            })
            f.write(summary.to_markdown())
            f.write("\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("See the following files for detailed results:\n\n")
            f.write("- `all_comparisons.csv`: All channel vs vanilla comparisons\n")
            f.write("- `successful_improvements.csv`: Configurations meeting success criteria\n")
            f.write("- `summary_improvements.png`: Overview of improvements\n")
            f.write("- `channel_spec_effects.png`: Effect of channel specifications\n")
            f.write("- `parameter_effects.png`: Effect of various parameters\n")
            f.write("- `bound_validation.png`: Validation of theoretical bounds\n")
    
    print(f"\nSaved summary report to: {report_path}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
