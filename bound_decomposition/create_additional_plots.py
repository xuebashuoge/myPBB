"""
Create consolidated additional visualizations with comprehensive information.
Each figure contains rich analysis across multiple configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_epoch_evolution_consolidated(csv_path, output_dir):
    """
    Create consolidated plots showing epoch evolution for all models.
    One figure per loss_type with all models, priors, and both norm types.
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    
    print(f"\nCreating consolidated epoch evolution plots...")
    
    # One figure per loss type (CE and 0-1)
    for loss_type in sorted(df['loss_type'].unique()):
        
        fig, axes = plt.subplots(3, 3, figsize=(22, 16))
        fig.suptitle(f'Epoch Evolution Analysis - {loss_type.upper()} Loss\n(Solid/Dashed lines = Frob/Spec norms)', 
                    fontsize=18, fontweight='bold')
        
        # Filter data
        subset = df[df['loss_type'] == loss_type]
        
        # Get model-dataset combinations
        model_datasets = subset.groupby(['model', 'dataset'])
        
        plot_idx = 0
        for (model, dataset), group in sorted(model_datasets):
            if plot_idx >= 9:
                break
            
            row = plot_idx // 3
            col = plot_idx % 3
            ax = axes[row, col]
            
            # Plot for each combination of prior and norm
            for prior_type, color_base in [('rand', '#e74c3c'), ('learnt', '#2ecc71')]:
                for norm_type, linestyle, marker in [('frob', '-', 'o'), ('spec', '--', 's')]:
                    
                    prior_norm_data = group[(group['prior_type'] == prior_type) & 
                                           (group['norm_type'] == norm_type)]
                    
                    if len(prior_norm_data) == 0:
                        continue
                    
                    # Average over channels for each epoch
                    epoch_avg = prior_norm_data.groupby('epoch').agg({
                        'lhs': 'mean',
                        'rhs': 'mean',
                        'relative_gap': 'mean'
                    })
                    
                    if len(epoch_avg) == 0:
                        continue
                    
                    # Create label
                    label = f'{prior_type.capitalize()}-{norm_type.upper()}'
                    
                    # Plot relative gap
                    ax.plot(epoch_avg.index, epoch_avg['relative_gap'], 
                           marker=marker, linestyle=linestyle, linewidth=2, markersize=7,
                           color=color_base, label=label, alpha=0.8)
            
            ax.set_xlabel('Epoch', fontweight='bold', fontsize=11)
            ax.set_ylabel('Relative Gap (%)', fontweight='bold', fontsize=11)
            ax.set_title(f'{model.upper()} - {dataset.upper()}', fontweight='bold', fontsize=12)
            ax.legend(fontsize=9, loc='best', ncol=2)
            ax.grid(alpha=0.3)
            ax.axhline(y=0, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
            
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, 9):
            axes[idx // 3, idx % 3].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save
        filename = f"epoch_evolution_{loss_type}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        # plt.savefig(output_dir / filename.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()


def plot_prior_comparison_comprehensive(csv_path, output_dir):
    """
    Create comprehensive prior comparison with multiple analyses in one figure.
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    
    print(f"\nCreating comprehensive prior comparison plot...")
    
    # Group by everything except prior_type
    configs = df.groupby(['model', 'dataset', 'norm_type', 'loss_type', 'epoch', 'channel_type', 'channel_spec'])
    
    comparison_data = []
    
    for key, group in configs:
        rand_data = group[group['prior_type'] == 'rand']
        learnt_data = group[group['prior_type'] == 'learnt']
        
        if len(rand_data) > 0 and len(learnt_data) > 0:
            comparison_data.append({
                'model': key[0],
                'dataset': key[1],
                'norm_type': key[2],
                'loss_type': key[3],
                'epoch': key[4],
                'channel_type': key[5],
                'channel_spec': key[6],
                'rand_gap': rand_data['relative_gap'].values[0],
                'learnt_gap': learnt_data['relative_gap'].values[0],
                'rand_lhs': rand_data['lhs'].values[0],
                'learnt_lhs': learnt_data['lhs'].values[0],
                'rand_rhs': rand_data['rhs'].values[0],
                'learnt_rhs': learnt_data['rhs'].values[0],
                'improvement': rand_data['relative_gap'].values[0] - learnt_data['relative_gap'].values[0]
            })
    
    if not comparison_data:
        print("  No matching rand/learnt pairs found")
        return
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create comprehensive figure with 2x3 subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Random vs Learnt Prior - Comprehensive Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Scatter of gaps (by norm type)
    ax = plt.subplot(2, 3, 1)
    for norm_type, color, marker in [('frob', '#3498db', 'o'), ('spec', '#e74c3c', 's')]:
        subset = comp_df[comp_df['norm_type'] == norm_type]
        ax.scatter(subset['rand_gap'], subset['learnt_gap'], 
                  alpha=0.6, s=50, color=color, marker=marker, label=f'{norm_type.upper()} Norm')
    max_val = max(comp_df['rand_gap'].max(), comp_df['learnt_gap'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', label='Equal performance', linewidth=2, alpha=0.5)
    ax.set_xlabel('Random Prior Gap (%)', fontweight='bold')
    ax.set_ylabel('Learnt Prior Gap (%)', fontweight='bold')
    ax.set_title('Gap Comparison\n(below line = learnt better)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 2: Improvement by model/dataset
    ax = plt.subplot(2, 3, 2)
    avg_imp = comp_df.groupby(['model', 'dataset'])['improvement'].mean().sort_values()
    colors_imp = ['green' if x > 0 else 'red' for x in avg_imp.values]
    avg_imp.plot(kind='barh', ax=ax, color=colors_imp, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Average Gap Improvement (%)', fontweight='bold')
    ax.set_title('Improvement by Configuration', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
    
    # Plot 3: Improvement across epochs
    ax = plt.subplot(2, 3, 3)
    for model_dataset, group in comp_df.groupby(['model', 'dataset']):
        epoch_imp = group.groupby('epoch')['improvement'].mean()
        ax.plot(epoch_imp.index, epoch_imp.values, 'o-', linewidth=2, markersize=8,
               label=f'{model_dataset[0].upper()}-{model_dataset[1].upper()}', alpha=0.8)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Average Gap Improvement (%)', fontweight='bold')
    ax.set_title('Improvement vs Training Epoch', fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Plot 4: LHS comparison (population risk)
    ax = plt.subplot(2, 3, 4)
    for model_dataset, group in comp_df.groupby(['model', 'dataset']):
        ax.scatter(group['rand_lhs'], group['learnt_lhs'], 
                  alpha=0.6, s=50, label=f'{model_dataset[0].upper()}-{model_dataset[1].upper()}')
    max_lhs = max(comp_df['rand_lhs'].max(), comp_df['learnt_lhs'].max())
    ax.plot([0, max_lhs], [0, max_lhs], 'k--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Random Prior LHS', fontweight='bold')
    ax.set_ylabel('Learnt Prior LHS', fontweight='bold')
    ax.set_title('Population Risk Comparison\n(below line = learnt has lower risk)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Plot 5: Component comparison (averaged)
    ax = plt.subplot(2, 3, 5)
    
    # Get component data for both priors
    rand_comps = df[df['prior_type'] == 'rand'].groupby(['model', 'dataset']).agg({
        'empirical': 'mean',
        'channel_term': 'mean',
        'kl_term': 'mean'
    })
    learnt_comps = df[df['prior_type'] == 'learnt'].groupby(['model', 'dataset']).agg({
        'empirical': 'mean',
        'channel_term': 'mean',
        'kl_term': 'mean'
    })
    
    x = np.arange(len(rand_comps))
    width = 0.35
    
    # KL term comparison (most important)
    ax.bar(x - width/2, rand_comps['kl_term'], width, label='Random Prior', 
           alpha=0.7, color='#e74c3c', edgecolor='black')
    ax.bar(x + width/2, learnt_comps['kl_term'], width, label='Learnt Prior', 
           alpha=0.7, color='#2ecc71', edgecolor='black')
    
    ax.set_ylabel('KL Term (Average)', fontweight='bold')
    ax.set_title('KL Term Comparison\n(lower = better prior match)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}-{d}' for m, d in rand_comps.index], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 6: Success rate and statistics
    ax = plt.subplot(2, 3, 6)
    ax.axis('off')
    
    # Calculate statistics
    total_cases = len(comp_df)
    learnt_better = (comp_df['improvement'] > 0).sum()
    success_rate = 100 * learnt_better / total_cases
    avg_improvement = comp_df['improvement'].mean()
    median_improvement = comp_df['improvement'].median()
    max_improvement = comp_df['improvement'].max()
    min_improvement = comp_df['improvement'].min()
    
    # Create text summary
    summary_text = f"""
PRIOR COMPARISON STATISTICS
{'='*40}

Total Configurations: {total_cases}

Learnt Prior Better: {learnt_better}/{total_cases}
Success Rate: {success_rate:.1f}%

Gap Improvement (Rand - Learnt):
  Average: {avg_improvement:.2f}%
  Median: {median_improvement:.2f}%
  Max: {max_improvement:.2f}%
  Min: {min_improvement:.2f}%

By Loss Type:
  CE Loss: {comp_df[comp_df['loss_type']=='ce']['improvement'].mean():.2f}%
  0-1 Error: {comp_df[comp_df['loss_type']=='01']['improvement'].mean():.2f}%

By Norm Type:
  Frobenius: {comp_df[comp_df['norm_type']=='frob']['improvement'].mean():.2f}%
  Spectral: {comp_df[comp_df['norm_type']=='spec']['improvement'].mean():.2f}%

Conclusion:
  {'✓ Learnt priors significantly outperform random priors' if success_rate > 80 else '⚠ Mixed results'}
"""
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    filename = "prior_comparison_comprehensive.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    # plt.savefig(output_dir / filename.replace('.png', '.pdf'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    # Print statistics to console
    print(f"\n  Success rate (learnt better): {success_rate:.1f}%")
    print(f"  Average improvement: {avg_improvement:.2f}%")
    print(f"  Median improvement: {median_improvement:.2f}%")


def main():
    csv_path = '/Users/yangshuo/Git/myPBB/bound_decomposition/bound_summary_statistics.csv'
    output_dir = '/Users/yangshuo/Git/myPBB/bound_decomposition'
    
    print("="*80)
    print("CREATING CONSOLIDATED ADDITIONAL VISUALIZATIONS")
    print("="*80)
    
    # Consolidated epoch evolution plots (2 figures: CE and 0-1 loss)
    # Each figure shows all models with both norm types (frob/spec)
    plot_epoch_evolution_consolidated(csv_path, output_dir)
    
    # Comprehensive prior comparison (1 figure with rich information)
    plot_prior_comparison_comprehensive(csv_path, output_dir)
    
    print("\n" + "="*80)
    print("CONSOLIDATED VISUALIZATIONS COMPLETE")
    print("="*80)
    print("\nGenerated 3 comprehensive figures:")
    print("  - 2 epoch evolution plots (1 per loss type, all norms combined)")
    print("  - 1 comprehensive prior comparison plot")


if __name__ == '__main__':
    main()
