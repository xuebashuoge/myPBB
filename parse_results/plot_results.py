# cnn-4, mnist, bec-0.1, spec, epoch
# vanillaimport matplotlib.pyplot as plt
import numpy as np
import json
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# ==========================================
# 1. USER CONFIGURATION & DATA INPUT
# ==========================================

# Path to the folder containing your JSONs
BASE_DIR = 'results/posterior/'

# PLACEHOLDER: Paste your JSON filenames here.
# Ensure the keys ("FCN-4", "CNN-4", etc.) match the order you want on the X-axis.
DATA_FILES = {
    "BEC": {  # Subplot (a)
        "FCN-4\n(MNIST)": {
            "Standard ERM": "fcn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-vanilla_perc-pri0.5_epoch-pri20_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-spec_seed7_bounds.json",       # <--- Paste filename
            "Proposed Robust": "fcn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan1.0-bec-outage0.1-spec-kl0.01_perc-pri0.5_epoch-pri20_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-spec_seed7_bounds.json"  # <--- Paste filename
        },
        "CNN-4\n(MNIST)": {
            "Standard ERM": "cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-vanilla_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-spec_seed7_bounds.json",      # <--- Paste filename
            "Proposed Robust": "cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan1.0-bec-outage0.1-spec-kl0.01_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer2_mcsamples500_norm-spec_seed7_bounds.json" # <--- Paste filename
        },
        "CNN-9\n(CIFAR-10)": {
            "Standard ERM": "cnn-9_cifar10_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch10_bs250_lr0.001_mon0.95_dp0.2_objective-vanilla_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer4_mcsamples500_norm-spec_seed7_bounds.json",      # <--- Paste filename
            "Proposed Robust": "cnn-9_cifar10_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch10_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan1.0-bec-outage0.1-spec-kl0.01_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/bec-outage0.1_chan-layer4_mcsamples500_norm-spec_seed7_bounds.json" # <--- Paste filename
        }
    },
    "Rayleigh": {  # Subplot (b)
        "FCN-4\n(MNIST)": {
            "Standard ERM": "fcn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-vanilla_perc-pri0.5_epoch-pri20_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/rayleigh-zf-tx1.0-noise1.0_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json",       # <--- Paste filename
            "Proposed Robust": "fcn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch20_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan1.0-rayleigh-zf-tx1.0-noise1.0-frob-kl0.1_perc-pri0.5_epoch-pri20_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/rayleigh-zf-tx1.0-noise1.0_chan-layer2_mcsamples500_norm-frob_seed7_bounds.json"  # <--- Paste filename
        },
        "CNN-4\n(MNIST)": {
            "Standard ERM": "cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch50_bs250_lr0.001_mon0.95_dp0.2_objective-vanilla_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/rayleigh-zf-tx1.0-noise1.0_chan-layer2_mcsamples500_norm-spec_seed7_bounds.json",      # <--- Paste filename
            "Proposed Robust": "cnn-4_mnist_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch50_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan0.1-rayleigh-zf-tx1.0-noise1.0-spec-kl0.005_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/rayleigh-zf-tx1.0-noise1.0_chan-layer2_mcsamples500_norm-spec_seed7_bounds.json" # <--- Paste filename
        },
        "CNN-9\n(CIFAR-10)": {
            "Standard ERM": "cnn-9_cifar10_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch10_bs250_lr0.001_mon0.95_dp0.2_objective-vanilla_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/rayleigh-zf-tx1.0-noise1.0_chan-layer4_mcsamples500_norm-spec_seed7_bounds.json",      # <--- Paste filename
            "Proposed Robust": "cnn-9_cifar10_learnt_gaussian_sig0.03_bounded-pmin1e-05_epoch10_bs250_lr0.001_mon0.95_dp0.2_objective-channel_gradient-chan0.01-rayleigh-zf-tx1.0-noise1.0-spec-kl0.01_perc-pri0.5_epoch-pri70_bs-pri250_lr-pri0.005_mom-pri0.99_dp-pri0.2_seed7/bounds/rayleigh-zf-tx1.0-noise1.0_chan-layer4_mcsamples500_norm-spec_seed7_bounds.json" # <--- Paste filename
        }
    }
}

# Metric Selection
# Metric Selection (UPDATED FOR NLL / CROSS ENTROPY)
METRIC_KEYS = {
    "population": "stochastic_loss_mc",     # Population Risk (NLL)
    "empirical": "empirical_nll_loss",      # Empirical Risk (NLL)
    "kl": "kl_final",                       # KL Term (numerator)
    "channel": "channel_term",           # Channel Term
    "divisor_k": "k",                        # Key to divide KL by (e.g., training set size)
    "bound": "bound_ce_rhs"
}

# ==========================================
# 2. DATA EXTRACTION HELPER
# ==========================================

def get_metrics(filename):
    """Parses JSON and returns (emp, kl, channel, pop)"""
    filepath = os.path.join(BASE_DIR, filename)
    
    # Mock data for demonstration if file doesn't exist
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Generating dummy NLL data.")
        # Generates slightly higher values typical for NLL compared to 0-1 error
        return np.random.rand() * 0.5, np.random.rand() * 0.2, np.random.rand() * 0.1, np.random.rand() * 0.7

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract values using the NLL keys
    emp = data.get(METRIC_KEYS["empirical"], 0.0)
    
    # Calculate KL term (KL / k)
    k_val = data.get(METRIC_KEYS["divisor_k"], 1.0) 
    if k_val == 0: k_val = 1.0 # Prevent div by zero
    kl = data.get(METRIC_KEYS["kl"], 0.0) / k_val
    
    chan = data.get(METRIC_KEYS["channel"], 0.0)
    pop = data.get(METRIC_KEYS["population"], 0.0)

    bound = data.get(METRIC_KEYS["bound"], 0.0)
    
    return emp, kl, chan, pop, bound

# ==========================================
# 3. PLOTTING SETUP & LOOP
# ==========================================

# IEEE Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'font.size': 12,        # Increased slightly for individual plots
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (6, 6) # Square-ish aspect ratio for individual plots
})

# Define the Models (Loop targets) and Channels (X-axis)
model_keys = ["FCN-4\n(MNIST)", "CNN-4\n(MNIST)", "CNN-9\n(CIFAR-10)"]
model_file_suffixes = ["fcn4_mnist", "cnn4_mnist", "cnn9_cifar10"] # Safe filenames

channels = ["BEC", "Rayleigh"]
channel_display_names = ["BEC\n($p_o=0.1$)", "Rayleigh\n(SNR=10dB)"]

# Colors
c_emp = '#1f77b4'  # Blue
c_kl = '#ff7f0e'   # Orange
c_chan = '#d62728' # Red

bar_width = 0.35

# Custom Legend Handles
handles = [
    # Colors
    mpatches.Patch(color=c_emp, label='Empirical Risk'),
    mpatches.Patch(color=c_kl, label='KL Term'),
    mpatches.Patch(color=c_chan, label='Channel term'),
    # Spacer
    mpatches.Patch(color='none', label=' '), 
    # Patterns
    mpatches.Patch(facecolor='white', edgecolor='black', label='Standard ERM'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Proposed Robust'),
    # Marker
    mlines.Line2D([], [], color='black', marker='D', linestyle='None',
                          markersize=6, label='Pop. Risk')
]

# ==========================================
# 4. DRAWING LOOP (One Figure per Model)
# ==========================================

for idx, (model_key, file_suffix) in enumerate(zip(model_keys, model_file_suffixes)):
    # Create a new independent figure for each model
    fig, ax = plt.subplots()
    
    x = np.arange(len(channels))
    
    # Iterate over channels (BEC, Rayleigh) for the current model
    for i, channel in enumerate(channels):
        if model_key not in DATA_FILES[channel]:
            print(f"Skipping {model_key} in {channel}")
            continue
            
        methods = DATA_FILES[channel][model_key]
        
        # We assume 2 methods: Standard ERM (pos 0) and Proposed Robust (pos 1)
        for j, (method_name, filename) in enumerate(methods.items()):
            emp, kl, chan, pop, bound = get_metrics(filename)

            print(f"Model: {model_key}, Channel: {channel}, Method: {method_name} -> Pop: {pop:.4f}, Bound: {bound:.4f}, Emp: {emp:.4f}, KL: {kl:.4f}, Chan: {chan:.4f}")
            
            # X position: Shift left for ERM, right for Robust
            pos = x[i] - bar_width/2 if j == 0 else x[i] + bar_width/2
            
            # Style logic
            hatch = '///' if "Robust" in method_name else None
            
            # Stack 1: Empirical Risk (Bottom)
            ax.bar(pos, emp, bar_width, color=c_emp, edgecolor='black', hatch=hatch)
            
            # Stack 2: KL Term (Middle) - Starts at `emp`
            ax.bar(pos, kl, bar_width, bottom=emp, color=c_kl, edgecolor='black', hatch=hatch)
            
            # Stack 3: Channel term (Top) - Starts at `emp + kl`
            ax.bar(pos, chan, bar_width, bottom=emp + kl, color=c_chan, edgecolor='black', hatch=hatch)
            
            # Overlay Marker: Population Risk
            ax.scatter(pos, pop, color='black', marker='D', s=50, zorder=10)

            # --- ADDED: Method Name Label on Top ---
            total_height = emp + kl + chan
            # Shorten names for better fit
            display_name = "ERM" if "Standard" in method_name else "Robust"
            ax.text(pos, total_height + 0.02, display_name, 
                    ha='center', va='bottom', fontsize=9, rotation=0, color='black')

    # Formatting
    clean_title = model_key.replace('\n', ' ')
    ax.set_title(f"{clean_title}")
    ax.set_xlabel("Channel Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(channel_display_names)
    ax.set_ylabel("Generalization Bound / NLL Loss") 
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add Legend to each individual plot
    # Adjusted 'ncol' to 2 or 3 to fit the narrower aspect ratio better
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False, fontsize=9)

    plt.tight_layout()
    
    # Save the individual figure
    output_filename = f"parse_results/result_{file_suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    # print(f"Generated: {output_filename}")
    
    # Clear memory for next iteration
    plt.close(fig)
