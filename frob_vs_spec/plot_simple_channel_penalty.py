"""
Simple Channel Penalty Comparison: Frobenius vs Spectral Norm for BEC

Creates 3 figures (one per model-dataset configuration) showing how channel penalty
varies with outage probability for both Frobenius and Spectral norms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

def compute_frobenius_penalty(K_f, d, p_o):
    """
    Frobenius channel penalty: K_f × √(d × p_o)
    Simplified from: K_f × Σ_{r=1}^d C(d,r) p_o^r (1-p_o)^(d-r) √r
    """
    return K_f * np.sqrt(d * p_o)

def compute_spectral_penalty(K_s, d, p_o):
    """
    Spectral channel penalty: K_s × (1 - (1-p_o)^d)
    """
    return K_s * (1 - (1 - p_o) ** d)

# Configuration data
configs = [
    {
        'name': 'CNN-4 MNIST',
        'dimension': 9216,
        'K_f': 0.004370,  # From bec-outage0.1_chan-layer2_mcsamples500_norm-frob
        'K_s': 0.045504,  # From bec-outage0.1_chan-layer2_mcsamples500_norm-spec
        'pattern': 'cnn-4_mnist*'
    },
    {
        'name': 'FCN-4 MNIST',
        'dimension': 600,
        'K_f': 0.008666,  # From fcn-4 files
        'K_s': 0.071373,
        'pattern': 'fcn-4_mnist*'
    },
    {
        'name': 'CNN-9 CIFAR-10',
        'dimension': 8192,
        'K_f': 0.001518,  # From bec-outage0.1_chan-layer4_mcsamples500_norm-frob (current file)
        'K_s': 0.018739,  # From bec-outage0.1_chan-layer4_mcsamples500_norm-spec
        'pattern': 'cnn-9_cifar10*'
    }
]

# Verify K values by reading from actual files
print("Verifying Lipschitz constants from JSON files...")
for config in configs:
    # Find matching files
    results_dir = Path('results/posterior')
    frob_files = list(results_dir.rglob(f'{config["pattern"]}/bounds/bec-outage0.1*norm-frob*.json'))
    spec_files = list(results_dir.rglob(f'{config["pattern"]}/bounds/bec-outage0.1*norm-spec*.json'))
    
    if frob_files:
        with open(frob_files[0], 'r') as f:
            data = json.load(f)
            config['K_f'] = data['Lipschitz_constant']
            config['dimension'] = int(data['dimension'])
    
    if spec_files:
        with open(spec_files[0], 'r') as f:
            data = json.load(f)
            config['K_s'] = data['Lipschitz_constant']
    
    print(f"  {config['name']}: d={config['dimension']}, K_f={config['K_f']:.6f}, K_s={config['K_s']:.6f}")

print("\nGenerating figures...")

# Outage probability range
outage_range = np.linspace(0.0, 1.0, 500)

# Create individual figures for each configuration
for idx, config in enumerate(configs):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    d = config['dimension']
    K_f = config['K_f']
    K_s = config['K_s']
    
    # Compute penalties
    frob_penalties = [compute_frobenius_penalty(K_f, d, p) if p > 0 else 0 for p in outage_range]
    spec_penalties = [compute_spectral_penalty(K_s, d, p) for p in outage_range]
    
    # Plot both lines
    ax.plot(outage_range, frob_penalties, 'b-', linewidth=3, 
           label=f'Frobenius: $K_f \\times \\sqrt{{d \\times p_o}}$')
    ax.plot(outage_range, spec_penalties, 'r-', linewidth=3,
           label=f'Spectral: $K_s \\times (1 - (1-p_o)^d)$')
    
    # Mark the p_o = 0.1 point (our data point)
    # p_test = 0.1
    # frob_at_01 = compute_frobenius_penalty(K_f, d, p_test)
    # spec_at_01 = compute_spectral_penalty(K_s, d, p_test)
    
    # ax.scatter([p_test], [frob_at_01], s=200, c='blue', edgecolors='black', 
    #           linewidths=2, zorder=5, marker='o', label=f'Frob at $p_o=0.1$: {frob_at_01:.4f}')
    # ax.scatter([p_test], [spec_at_01], s=200, c='red', edgecolors='black',
    #           linewidths=2, zorder=5, marker='s', label=f'Spec at $p_o=0.1$: {spec_at_01:.4f}')
    
    # Formatting
    ax.set_xlabel('Outage Probability ($p_o$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Channel Penalty', fontsize=14, fontweight='bold')
    ax.set_title(f'{config["name"]}\n$d={d}$, $K_f={K_f:.6f}$, $K_s={K_s:.6f}$',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)
    
    # # Add annotation showing ratio at p_o = 0.1
    # ratio = spec_at_01 / frob_at_01
    # improvement = (1 - ratio) * 100
    # ax.text(0.5, max(frob_at_01, spec_at_01) * 1.2,
    #        f'At $p_o=0.1$: Spectral/Frobenius = {ratio:.3f}\n' + 
    #        f'Spectral is {improvement:.1f}% better',
    #        fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
    #        ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    filename_base = config['name'].lower().replace(' ', '_').replace('-', '')
    plt.savefig(f'channel_penalty_{filename_base}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'channel_penalty_{filename_base}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: channel_penalty_{filename_base}.pdf/png")
    plt.close()

# Create a combined figure showing all three
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (ax, config) in enumerate(zip(axes, configs)):
    d = config['dimension']
    K_f = config['K_f']
    K_s = config['K_s']
    
    # Compute penalties
    frob_penalties = [compute_frobenius_penalty(K_f, d, p) if p > 0 else 0 for p in outage_range]
    spec_penalties = [compute_spectral_penalty(K_s, d, p) for p in outage_range]
    
    # Plot both lines
    ax.plot(outage_range, frob_penalties, 'b-', linewidth=2.5, 
           label='Frobenius')
    ax.plot(outage_range, spec_penalties, 'r-', linewidth=2.5,
           label='Spectral')
    
    # # Mark p_o = 0.1
    # p_test = 0.1
    # frob_at_01 = compute_frobenius_penalty(K_f, d, p_test)
    # spec_at_01 = compute_spectral_penalty(K_s, d, p_test)
    
    # ax.scatter([p_test], [frob_at_01], s=150, c='blue', edgecolors='black', 
    #           linewidths=2, zorder=5, marker='o')
    # ax.scatter([p_test], [spec_at_01], s=150, c='red', edgecolors='black',
    #           linewidths=2, zorder=5, marker='s')
    
    # Formatting
    ax.set_xlabel('Outage Probability ($p_o$)', fontsize=12, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Channel Penalty', fontsize=12, fontweight='bold')
    ax.set_title(f'{config["name"]}\n$d={d}$',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)

plt.suptitle('Channel Penalty Comparison: Frobenius vs Spectral Norm for BEC',
            fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('channel_penalty_all_configs.pdf', dpi=300, bbox_inches='tight')
plt.savefig('channel_penalty_all_configs.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: channel_penalty_all_configs.pdf/png")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for config in configs:
    d = config['dimension']
    K_f = config['K_f']
    K_s = config['K_s']
    
    print(f"\n{config['name']}:")
    print(f"  Dimension: {d}")
    print(f"  K_f (Frobenius): {K_f:.6f}")
    print(f"  K_s (Spectral): {K_s:.6f}")
    print(f"  K_s/K_f ratio: {K_s/K_f:.2f}")
    
    # At p_o = 0.1
    p_test = 0.1
    frob = compute_frobenius_penalty(K_f, d, p_test)
    spec = compute_spectral_penalty(K_s, d, p_test)
    
    print(f"\n  At p_o = 0.1:")
    print(f"    Frobenius penalty: {frob:.6f}")
    print(f"    Spectral penalty:  {spec:.6f}")
    print(f"    Spectral/Frobenius: {spec/frob:.3f}")
    print(f"    Spectral improvement: {(1 - spec/frob)*100:.1f}%")

print("\n" + "="*80)
print("Files generated:")
print("  • channel_penalty_cnn4mnist.pdf/png")
print("  • channel_penalty_fcn4mnist.pdf/png")
print("  • channel_penalty_cnn9cifar10.pdf/png")
print("  • channel_penalty_all_configs.pdf/png (combined)")
print("="*80)
