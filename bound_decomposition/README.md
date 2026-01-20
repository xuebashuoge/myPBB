# Bound Decomposition Analysis

This folder contains comprehensive analysis and visualizations of PAC-Bayes bounds across different configurations.

## File Structure

### Analysis Scripts
- **`analyze_and_visualize_bounds.py`**: Main analysis script that:
  - Loads all posterior results from JSON files
  - Verifies bound validity (RHS ≥ LHS)
  - Computes bound decompositions
  - Generates visualizations
  - Exports summary statistics

- **`create_additional_plots.py`**: Creates supplementary visualizations including:
  - Epoch evolution plots
  - Channel comparison plots
  - Prior comparison plots

### Output Files

#### Summary Data
- **`bound_summary_statistics.csv`**: Complete tabular data for all 216 configurations
  - Columns include: model, dataset, prior_type, epoch, channel_type, channel_spec, norm_type, loss_type
  - Decomposition: lhs, rhs, empirical, channel_term, kl_term, gap, relative_gap
  - Additional metrics: dimension, lipschitz, kl_final

#### Main Visualizations

Each figure corresponds to a specific **epoch** and **norm type** combination.

**Figure Layout** (2 rows × 3 columns = 6 subfigures):
```
┌─────────────────────────────────────────────────────────────┐
│                 Epoch X - NORM Norm                          │
│            Cross-Entropy / 0-1 Error Bound Decomposition     │
├──────────────────┬──────────────────┬──────────────────────┤
│  FCN-4 (MNIST)   │  CNN-4 (MNIST)   │  CNN-9 (CIFAR10)     │
│  Random Prior    │  Random Prior    │  Random Prior        │
├──────────────────┼──────────────────┼──────────────────────┤
│  FCN-4 (MNIST)   │  CNN-4 (MNIST)   │  CNN-9 (CIFAR10)     │
│  Learnt Prior    │  Learnt Prior    │  Learnt Prior        │
└──────────────────┴──────────────────┴──────────────────────┘
```

**Naming Convention**: `epoch{N}_{norm}_{loss}_decomposition.pdf`
- `{N}`: Training epoch (10, 20, or 50)
- `{norm}`: Norm type (frob or spec)
- `{loss}`: Loss type (ce or 01)

**Examples**:
- `epoch10_frob_ce_decomposition.pdf`: Epoch 10, Frobenius norm, Cross-Entropy loss
- `epoch50_spec_01_decomposition.pdf`: Epoch 50, Spectral norm, 0-1 Error

#### Subfigure Details

Each subfigure shows:
- **Bars**: Different channel configurations (BEC, Rayleigh, Rayleigh-ZF with various parameters)
- **Bar decomposition** (stacked, bottom to top):
  1. **Empirical Risk** (green): Training set performance
  2. **Channel Term** (red): Wireless channel penalty (∝ Lipschitz constant)
  3. **KL Term** (blue): Complexity term = KL(posterior||prior) / √n
- **Dashed line** (black): Population risk (LHS) - the ground truth we want to bound
- **Total bar height**: Upper bound (RHS)

## Understanding the Visualizations

### Bound Decomposition Formula

```
RHS (Upper Bound) = Empirical Risk + Channel Term + KL Term / √n
```

The bound is **valid** when `RHS ≥ LHS` (population risk).

### Color Coding
- 🟢 **Green (Empirical)**: How well the model performs on training data
- 🔴 **Red (Channel)**: Cost of wireless channel noise
- 🔵 **Blue (KL)**: Model complexity / prior mismatch
- ⚫ **Black dashed line**: True population risk (estimated via Monte Carlo)

### Channel Types
- **BEC**: Binary Erasure Channel with outage probability (0.1 or 0.5)
- **Rayleigh**: Rayleigh fading channel with SNR (0dB or 10dB)
- **Rayleigh-ZF**: Rayleigh with zero-forcing, SNR (0dB or 10dB)

### Reading the Plots

**Good bound (tight)**:
- Total bar height is close to the dashed line
- Small gap between RHS and LHS

**Loose bound**:
- Total bar height much higher than dashed line
- Large gap indicates conservative bound

**Violated bound** ⚠️:
- Dashed line above the bar top
- Indicates RHS < LHS (bound doesn't hold)
- Most violations occur at early epochs (10-20) with specific channel configurations

## Key Findings

### 1. Bound Validity
- **92.6%** of Cross-Entropy bounds are valid (200/216)
- **79.2%** of 0-1 Error bounds are valid (171/216)
- Violations mostly occur at early epochs (10-20) with high noise channels

### 2. Component Dominance
- **KL Term**: ~96-98% of the bound (dominant)
- **Empirical Risk**: ~1-3.5% of the bound
- **Channel Term**: ~0.7% of the bound

**Implication**: Prior learning is crucial for tight bounds!

### 3. Prior Comparison
- **Learnt priors** reduce gap by **10-40×** compared to random priors
- FCN-4/MNIST rand: 32,837% gap → learnt: 2,827% gap (11.6× improvement)
- CNN-4/MNIST rand: 36,158% gap → learnt: 3,053% gap (11.8× improvement)

### 4. Training Evolution
- Bounds tighten as training progresses
- Epoch 50 bounds are significantly tighter than epoch 10
- Learnt prior benefits compound with more training

### 5. Norm Comparison
- **Frobenius norm**: More conservative, fewer violations
- **Spectral norm**: Tighter bounds when valid, but more violations
- Recommendation: Use Frobenius for reliability

## Usage

### Running the Analysis

```bash
cd /Users/yangshuo/Git/myPBB
python3 bound_decomposition/analyze_and_visualize_bounds.py
```

This will:
1. Scan all `results/posterior/*/bounds/*.json` files
2. Verify bound validity
3. Generate all visualizations
4. Export `bound_summary_statistics.csv`

### Creating Additional Plots

```bash
python3 bound_decomposition/create_additional_plots.py
```

This generates:
- Epoch evolution comparisons
- Channel type comparisons
- Prior effectiveness analysis

## Output Summary

**Total configurations analyzed**: 216
- 3 models (FCN-4, CNN-4, CNN-9)
- 2 datasets (MNIST, CIFAR-10)
- 2 prior types (rand, learnt)
- 3 epochs (10, 20, 50)
- 3 channel types (BEC, Rayleigh, Rayleigh-ZF)
- 2 norm types (frob, spec)
- 2 loss types (CE, 0-1)

**Generated files**:
- 12 main decomposition figures (6 epoch-norm combos × 2 loss types)
- 1 summary CSV file
- 1 analysis report (ANALYSIS_REPORT.md)

## References

For detailed analysis and interpretation, see:
- **`ANALYSIS_REPORT.md`**: Comprehensive findings and recommendations
- **`bound_summary_statistics.csv`**: Raw data for custom analysis

---

*Generated on 2025-12-18*
*Analysis script: analyze_and_visualize_bounds.py*
