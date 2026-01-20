"""
Analyze and visualize posterior bounds decomposition across different configurations.

This script:
1. Scans all posterior results
2. Parses folder names to extract model, dataset, prior type, and epoch information
3. Loads bound results from JSON files
4. Verifies bound validity (RHS >= LHS)
5. Creates visualizations decomposing bounds into components
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def parse_folder_name(folder_name):
    """
    Parse folder name to extract configuration details.
    
    Example: cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch10_bs250_lr0.001_mon0.95_dp0.2_objective-vanilla_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7
    
    Returns dict with: model, dataset, prior_type, epoch, and other parameters
    """
    parts = folder_name.split('_')
    config = {}
    
    # Extract model type (e.g., cnn-4, fcn-4, cnn-9)
    config['model'] = parts[0]
    
    # Extract dataset (e.g., mnist, cifar10)
    config['dataset'] = parts[1]
    
    # Extract prior type (rand or learnt)
    config['prior_type'] = parts[2]  # rand or learnt
    
    # Extract epoch
    epoch_match = re.search(r'epoch(\d+)', folder_name)
    config['epoch'] = int(epoch_match.group(1)) if epoch_match else None
    
    # Extract other parameters for reference
    config['folder_name'] = folder_name
    
    return config


def parse_json_filename(filename):
    """
    Parse JSON filename to extract channel configuration.
    
    Examples:
    - bec-outage0.1_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json
    - rayleigh-tx1.0-noise0.1_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json
    - rayleigh-zf-tx1.0-noise1.0_chan-layer2_mcsamples500_norm-spec_seed7_bounds.json
    
    Returns dict with: channel_type, channel_spec, norm_type, snr (if applicable)
    """
    config = {}
    
    # Extract norm type
    if 'norm-frob' in filename:
        config['norm_type'] = 'frob'
    elif 'norm-spec' in filename:
        config['norm_type'] = 'spec'
    else:
        config['norm_type'] = 'unknown'
    
    # Extract channel type and specifications
    if filename.startswith('bec-'):
        config['channel_type'] = 'bec'
        outage_match = re.search(r'outage([\d.]+)', filename)
        config['outage'] = float(outage_match.group(1)) if outage_match else None
        config['channel_spec'] = f"outage={config['outage']}"
        config['snr_db'] = None
    elif filename.startswith('rayleigh-zf-'):
        config['channel_type'] = 'rayleigh-zf'
        tx_match = re.search(r'tx([\d.]+)', filename)
        noise_match = re.search(r'noise([\d.]+)', filename)
        config['tx_power'] = float(tx_match.group(1)) if tx_match else None
        config['noise_var'] = float(noise_match.group(1)) if noise_match else None
        if config['tx_power'] is not None and config['noise_var'] is not None:
            snr = config['tx_power'] / config['noise_var']
            config['snr_db'] = 10 * np.log10(snr)
            config['channel_spec'] = f"SNR={config['snr_db']:.1f}dB"
        else:
            config['snr_db'] = None
            config['channel_spec'] = f"tx={config['tx_power']},noise={config['noise_var']}"
    elif filename.startswith('rayleigh-'):
        config['channel_type'] = 'rayleigh'
        tx_match = re.search(r'tx([\d.]+)', filename)
        noise_match = re.search(r'noise([\d.]+)', filename)
        config['tx_power'] = float(tx_match.group(1)) if tx_match else None
        config['noise_var'] = float(noise_match.group(1)) if noise_match else None
        if config['tx_power'] is not None and config['noise_var'] is not None:
            snr = config['tx_power'] / config['noise_var']
            config['snr_db'] = 10 * np.log10(snr)
            config['channel_spec'] = f"SNR={config['snr_db']:.1f}dB"
        else:
            config['snr_db'] = None
            config['channel_spec'] = f"tx={config['tx_power']},noise={config['noise_var']}"
    else:
        config['channel_type'] = 'unknown'
        config['channel_spec'] = 'unknown'
        config['snr_db'] = None
    
    return config


def load_all_results(results_dir):
    """
    Scan all posterior results and load JSON files.
    
    Returns a list of dicts, each containing:
    - config: parsed folder configuration
    - channel_config: parsed channel configuration
    - data: loaded JSON data
    - filepath: path to JSON file
    """
    results = []
    posterior_dir = Path(results_dir) / 'posterior'
    
    if not posterior_dir.exists():
        print(f"Directory {posterior_dir} does not exist!")
        return results
    
    # Iterate through all subdirectories
    for subdir in posterior_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        bounds_dir = subdir / 'bounds'
        if not bounds_dir.exists():
            continue
        
        # Parse folder name
        folder_config = parse_folder_name(subdir.name)
        
        # Load all JSON files in bounds directory
        for json_file in bounds_dir.glob('*_bounds.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Parse JSON filename
                channel_config = parse_json_filename(json_file.name)
                
                # Skip Rayleigh without ZF (keep only BEC and Rayleigh-ZF)
                if channel_config['channel_type'] == 'rayleigh':
                    continue
                
                results.append({
                    'config': folder_config,
                    'channel_config': channel_config,
                    'data': data,
                    'filepath': str(json_file)
                })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return results


def verify_bounds(results):
    """
    Verify that all bounds are valid (RHS >= LHS).
    
    Returns a summary of bound validity.
    """
    print("\n" + "="*80)
    print("BOUND VALIDITY VERIFICATION")
    print("="*80)
    
    ce_violations = []
    zeroone_violations = []
    
    for result in results:
        data = result['data']
        config = result['config']
        channel_config = result['channel_config']
        
        # Check cross-entropy bound
        ce_lhs = data.get('bound_ce_lhs', 0)
        ce_rhs = data.get('bound_ce_rhs', 0)
        if ce_rhs < ce_lhs:
            ce_violations.append({
                'model': config['model'],
                'dataset': config['dataset'],
                'prior': config['prior_type'],
                'epoch': config['epoch'],
                'channel': channel_config['channel_type'],
                'spec': channel_config['channel_spec'],
                'norm': channel_config['norm_type'],
                'lhs': ce_lhs,
                'rhs': ce_rhs,
                'gap': ce_lhs - ce_rhs
            })
        
        # Check 0-1 bound
        zeroone_lhs = data.get('bound_01_lhs', 0)
        zeroone_rhs = data.get('bound_01_rhs', 0)
        if zeroone_rhs < zeroone_lhs:
            zeroone_violations.append({
                'model': config['model'],
                'dataset': config['dataset'],
                'prior': config['prior_type'],
                'epoch': config['epoch'],
                'channel': channel_config['channel_type'],
                'spec': channel_config['channel_spec'],
                'norm': channel_config['norm_type'],
                'lhs': zeroone_lhs,
                'rhs': zeroone_rhs,
                'gap': zeroone_lhs - zeroone_rhs
            })
    
    print(f"\nTotal results analyzed: {len(results)}")
    print(f"Cross-Entropy bound violations: {len(ce_violations)}")
    print(f"0-1 Error bound violations: {len(zeroone_violations)}")
    
    if ce_violations:
        print("\n--- Cross-Entropy Violations ---")
        for v in ce_violations[:10]:  # Show first 10
            print(f"  {v['model']}_{v['dataset']}_{v['prior']}_epoch{v['epoch']}, "
                  f"{v['channel']}_{v['spec']}, norm={v['norm']}: "
                  f"LHS={v['lhs']:.6f} > RHS={v['rhs']:.6f} (gap={v['gap']:.6f})")
    
    if zeroone_violations:
        print("\n--- 0-1 Error Violations ---")
        for v in zeroone_violations[:10]:  # Show first 10
            print(f"  {v['model']}_{v['dataset']}_{v['prior']}_epoch{v['epoch']}, "
                  f"{v['channel']}_{v['spec']}, norm={v['norm']}: "
                  f"LHS={v['lhs']:.6f} > RHS={v['rhs']:.6f} (gap={v['gap']:.6f})")
    
    return {
        'ce_violations': ce_violations,
        'zeroone_violations': zeroone_violations
    }


def compute_bound_decomposition(data):
    """
    Decompose the bound RHS into its components.
    
    Components:
    1. KL term: kl_final / sqrt(n_bound)
    2. Channel term: channel_term
    3. Empirical risk: empirical_nll_loss or empirical_01_error
    
    Returns dict with decomposition for both CE and 0-1 bounds.
    """
    n_bound = data.get('n_bound', 1)
    kl_final = data.get('kl_final', 0)
    
    # KL term (complexity term)
    kl_term = kl_final / np.sqrt(n_bound)
    
    # Channel term
    channel_term = data.get('channel_term', 0)
    
    # Empirical risks
    empirical_ce = data.get('empirical_nll_loss', 0)
    empirical_01 = data.get('empirical_01_error', 0)
    
    return {
        'ce': {
            'kl_term': kl_term,
            'channel_term': channel_term,
            'empirical': empirical_ce,
            'total': kl_term + channel_term + empirical_ce,
            'rhs': data.get('bound_ce_rhs', 0),
            'lhs': data.get('bound_ce_lhs', 0)
        },
        '01': {
            'kl_term': kl_term,
            'channel_term': channel_term,
            'empirical': empirical_01,
            'total': kl_term + channel_term + empirical_01,
            'rhs': data.get('bound_01_rhs', 0),
            'lhs': data.get('bound_01_lhs', 0)
        }
    }


def create_bound_decomposition_plots(results, output_dir):
    """
    Create bound decomposition plots.
    
    For each combination of (epoch, norm_type):
    - Create a figure with 2 rows × 3 columns (6 subfigures)
    - Row 1: Random prior for FCN-4, CNN-4, CNN-9
    - Row 2: Learnt prior for FCN-4, CNN-4, CNN-9
    - Each subfigure shows bars for different channel configurations
    - Each bar shows the decomposition: empirical + channel + KL terms
    - A dashed line shows the LHS (population risk)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group results by (epoch, norm_type)
    grouped = defaultdict(list)
    for result in results:
        config = result['config']
        channel_config = result['channel_config']
        key = (config['epoch'], channel_config['norm_type'])
        grouped[key].append(result)
    
    print(f"\n" + "="*80)
    print(f"CREATING VISUALIZATIONS")
    print("="*80)
    print(f"\nFound {len(grouped)} unique (epoch, norm) combinations to plot")
    
    # Define model order for consistency
    model_order = ['fcn-4', 'cnn-4', 'cnn-9']
    
    # Create plots for each (epoch, norm_type) combination
    for (epoch, norm_type), group_results in sorted(grouped.items()):
        print(f"\nProcessing: epoch{epoch}_norm-{norm_type}")
        
        # Create figure for both CE and 0-1 error
        for loss_type in ['ce', '01']:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Title
            loss_name = 'Cross-Entropy Loss' if loss_type == 'ce' else '0-1 Error'
            fig.suptitle(f'Epoch {epoch} - {norm_type.upper()} Norm - {loss_name} Bound Decomposition', 
                        fontsize=18, fontweight='bold')
            
            # Plot for each model and prior combination
            for row_idx, prior_type in enumerate(['rand', 'learnt']):
                for col_idx, model in enumerate(model_order):
                    ax = axes[row_idx, col_idx]
                    
                    # Filter results for this specific model and prior
                    filtered_results = [
                        r for r in group_results 
                        if r['config']['model'] == model 
                        and r['config']['prior_type'] == prior_type
                    ]
                    
                    # Create title for subplot
                    dataset = filtered_results[0]['config']['dataset'] if filtered_results else 'N/A'
                    subplot_title = f"{model.upper()} ({dataset.upper()})\n{prior_type.capitalize()} Prior"
                    
                    # Plot the decomposition
                    plot_bound_decomposition_bar(ax, filtered_results, loss_type, subplot_title)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure
            filename = f"epoch{epoch}_{norm_type}_{loss_type}_decomposition.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            # plt.savefig(filepath.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")
            plt.close()


def plot_bound_decomposition_bar(ax, results, loss_type, title):
    """
    Plot a single bar chart showing bound decomposition.
    
    Args:
        ax: matplotlib axis
        results: list of result dicts for this configuration
        loss_type: 'ce' or '01'
        title: subplot title
    """
    if not results:
        ax.text(0.5, 0.5, 'No data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Sort results by channel type and spec for consistent ordering
    results = sorted(results, key=lambda r: (
        r['channel_config']['channel_type'],
        r['channel_config'].get('outage', 0),
        r['channel_config'].get('snr_db', 0)
    ))
    
    # Prepare data for plotting
    labels = []
    empirical_vals = []
    channel_vals = []
    kl_vals = []
    lhs_vals = []
    rhs_vals = []
    
    for result in results:
        data = result['data']
        channel_config = result['channel_config']
        
        # Create label
        channel_type = channel_config['channel_type']
        if channel_type == 'bec':
            label = f"BEC\n(p={channel_config.get('outage', 0)})"
        elif 'rayleigh' in channel_type:
            snr_db = channel_config.get('snr_db', 0)
            if 'zf' in channel_type:
                label = f"Rayleigh-ZF\n({snr_db:.1f}dB)"
            else:
                label = f"Rayleigh\n({snr_db:.1f}dB)"
        else:
            label = channel_type
        
        labels.append(label)
        
        # Compute decomposition
        decomp = compute_bound_decomposition(data)
        comp = decomp[loss_type]
        
        empirical_vals.append(comp['empirical'])
        channel_vals.append(comp['channel_term'])
        kl_vals.append(comp['kl_term'])
        lhs_vals.append(comp['lhs'])
        rhs_vals.append(comp['rhs'])
    
    if not labels:
        ax.text(0.5, 0.5, 'No data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Create stacked bar chart
    x = np.arange(len(labels))
    width = 0.6
    
    # Stack: empirical (bottom) + channel + KL (top)
    p1 = ax.bar(x, empirical_vals, width, label='Empirical Risk', 
               color='#3498db', alpha=0.8)
    p2 = ax.bar(x, channel_vals, width, bottom=empirical_vals,
               label='Channel Term', color='#e74c3c', alpha=0.8)
    p3 = ax.bar(x, kl_vals, width, 
               bottom=np.array(empirical_vals) + np.array(channel_vals),
               label='KL Term', color='#2ecc71', alpha=0.8)
    
    # Add dashed line for LHS (population risk)
    ax.plot(x, lhs_vals, 'k--', linewidth=2, marker='o', 
           markersize=6, label='Population Risk (LHS)', zorder=10)
    
    # Formatting
    ax.set_ylabel('Risk Value', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value annotations on bars
    for i, (emp, ch, kl, lhs) in enumerate(zip(empirical_vals, channel_vals, kl_vals, lhs_vals)):
        total = emp + ch + kl
        # Show total RHS value on top
        ax.text(i, total, f'{total:.4f}', ha='center', va='bottom', 
               fontsize=7, fontweight='bold')


def generate_summary_statistics(results, output_dir):
    """
    Generate summary statistics about the bounds.
    """
    output_dir = Path(output_dir)
    
    # Create summary DataFrame
    summary_data = []
    
    for result in results:
        config = result['config']
        channel_config = result['channel_config']
        data = result['data']
        decomp = compute_bound_decomposition(data)
        
        for loss_type in ['ce', '01']:
            comp = decomp[loss_type]
            
            # Calculate tightness (gap between RHS and LHS)
            gap = comp['rhs'] - comp['lhs']
            relative_gap = gap / comp['lhs'] if comp['lhs'] > 0 else np.inf
            
            summary_data.append({
                'model': config['model'],
                'dataset': config['dataset'],
                'prior_type': config['prior_type'],
                'epoch': config['epoch'],
                'channel_type': channel_config['channel_type'],
                'channel_spec': channel_config['channel_spec'],
                'norm_type': channel_config['norm_type'],
                'loss_type': loss_type,
                'lhs': comp['lhs'],
                'rhs': comp['rhs'],
                'empirical': comp['empirical'],
                'channel_term': comp['channel_term'],
                'kl_term': comp['kl_term'],
                'gap': gap,
                'relative_gap': relative_gap,
                'dimension': data.get('dimension', 0),
                'lipschitz': data.get('Lipschitz_constant', 0),
                'kl_final': data.get('kl_final', 0)
            })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = output_dir / 'bound_summary_statistics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSummary statistics saved to: {csv_path}")
    
    # Print some interesting statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\n--- Average Relative Gap (RHS-LHS)/LHS by Configuration ---")
    avg_gap = df.groupby(['model', 'dataset', 'prior_type', 'loss_type'])['relative_gap'].mean()
    print(avg_gap.sort_values())
    
    print("\n--- Component Contribution (Average) ---")
    for loss_type in ['ce', '01']:
        print(f"\n{loss_type.upper()} Loss:")
        subset = df[df['loss_type'] == loss_type]
        total_rhs = subset['rhs'].mean()
        print(f"  Average RHS: {total_rhs:.6f}")
        print(f"  Empirical: {subset['empirical'].mean():.6f} ({100*subset['empirical'].mean()/total_rhs:.1f}%)")
        print(f"  Channel: {subset['channel_term'].mean():.6f} ({100*subset['channel_term'].mean()/total_rhs:.1f}%)")
        print(f"  KL: {subset['kl_term'].mean():.6f} ({100*subset['kl_term'].mean()/total_rhs:.1f}%)")
    
    return df


def main():
    """Main analysis pipeline."""
    # Configuration
    results_dir = '/Users/yangshuo/Git/myPBB/results'
    output_dir = '/Users/yangshuo/Git/myPBB/bound_decomposition'
    
    print("="*80)
    print("POSTERIOR BOUNDS ANALYSIS AND VISUALIZATION")
    print("="*80)
    
    # Load all results
    print("\nLoading results...")
    results = load_all_results(results_dir)
    print(f"Loaded {len(results)} result files")
    
    if not results:
        print("No results found!")
        return
    
    # Verify bounds
    verification = verify_bounds(results)
    
    # Generate summary statistics
    summary_df = generate_summary_statistics(results, output_dir)
    
    # Create visualizations
    create_bound_decomposition_plots(results, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
