import os
import json
import re
import math
import pandas as pd
import glob

# Configuration
RESULTS_DIR = 'results/posterior/'
OUTPUT_DIR = 'parse_results/'

# Regex patterns for parsing folder and file names
REGEX_FLOAT = r"[-+]?\d*\.\d+|\d+"

# Explicit Objective Order for Sorting
OBJECTIVE_ORDER = {
    'vanilla': 0,
    'channel_grad': 1,
    'channel_norm': 2,
    'channel': 3  # Assigned a rank, but will be filtered out
}

def get_snr_db(tx_power, noise_var):
    """Converts Tx Power and Noise Var to SNR in dB."""
    try:
        if noise_var == 0: return float('inf')
        snr_linear = tx_power / noise_var
        return 10 * math.log10(snr_linear)
    except Exception:
        return None

def parse_channel_specs(spec_str):
    """
    Parses the channel string (e.g., 'rayleigh-zf-tx1.0-noise0.1' or 'bec-outage0.5')
    Returns: type, display_param, raw_specs
    """
    spec_str = spec_str.lower()
    
    # Filter: We only care about bec and rayleigh_zf per instructions
    if 'rayleigh-zf' in spec_str:
        # Extract tx and noise
        tx_match = re.search(f"tx({REGEX_FLOAT})", spec_str)
        noise_match = re.search(f"noise({REGEX_FLOAT})", spec_str)
        
        if tx_match and noise_match:
            tx = float(tx_match.group(1))
            noise = float(noise_match.group(1))
            snr = get_snr_db(tx, noise)
            return 'rayleigh_zf', snr, {'tx': tx, 'noise': noise, 'snr_db': snr}
            
    elif 'bec' in spec_str:
        outage_match = re.search(f"outage({REGEX_FLOAT})", spec_str)
        if outage_match:
            outage = float(outage_match.group(1))
            return 'bec', outage, {'outage': outage}

    return None, None, None

def parse_folder_name(folder_name):
    """
    Deconstructs the massive folder name into a dictionary of config values.
    """
    info = {}
    
    # Basic Splits
    parts = folder_name.split('_')
    
    # 1. Model and Data
    info['model_full'] = parts[0] # e.g., fcn-4
    
    if 'mnist' in folder_name: info['dataset'] = 'mnist'
    elif 'cifar10' in folder_name: info['dataset'] = 'cifar10'
    else: info['dataset'] = 'unknown'

    # 2. Prior Type
    if 'learnt' in folder_name: info['prior_type'] = 'learnt'
    elif 'rand' in folder_name: info['prior_type'] = 'rand'
    else: info['prior_type'] = 'unknown'

    lr_match = re.search(f"_lr({REGEX_FLOAT})_", folder_name)
    info['learning_rate'] = float(lr_match.group(1)) if lr_match else None

    # 3. Objective
    obj_match = re.search(r"objective-(channel_grad|channel_norm|vanilla|channel)", folder_name)
    if obj_match:
        info['objective'] = obj_match.group(1)
    else:
        info['objective'] = 'unknown'

    # 4. Extract Penalties if channel objective
    if 'channel' in info['objective']:
        chan_pen_match = re.search(f"chan({REGEX_FLOAT})", folder_name)
        info['channel_penalty'] = float(chan_pen_match.group(1)) if chan_pen_match else 0.0
        
        kl_pen_match = re.search(f"kl({REGEX_FLOAT})", folder_name)
        info['kl_penalty'] = float(kl_pen_match.group(1)) if kl_pen_match else 0.0
        
        if 'spec' in folder_name: info['norm_type'] = 'spec'
        elif 'frob' in folder_name: info['norm_type'] = 'frob'
        else: info['norm_type'] = 'unknown'
        
    else:
        info['channel_penalty'] = 0.0
        info['kl_penalty'] = 1.0 
        info['norm_type'] = 'N/A'

    # 5. Extract Train Epochs
    epoch_match = re.search(f"_epoch({REGEX_FLOAT})_", folder_name)
    info['train_epochs'] = float(epoch_match.group(1)) if epoch_match else 0

    # 6. Extract Channel Specs from Folder Name
    if info['objective'] != 'vanilla':
        if 'rayleigh-zf' in folder_name:
            c_type, c_param, _ = parse_channel_specs(folder_name)
            info['train_channel_type'] = c_type
            info['train_channel_param'] = c_param
        elif 'bec' in folder_name:
            c_type, c_param, _ = parse_channel_specs(folder_name)
            info['train_channel_type'] = c_type
            info['train_channel_param'] = c_param
    else:
        info['train_channel_type'] = 'None'
        info['train_channel_param'] = 0

    return info

def process_json_file(file_path):
    """Reads the JSON and returns specific metrics using flat keys."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        res = {}
        
        # --- 01 Error Metrics (Classification) ---
        res['risk_01_lhs'] = data.get('bound_01_lhs')
        res['bound_01_rhs'] = data.get('bound_01_rhs')
        res['emp_01_error'] = data.get('empirical_01_error')
        res['train_01_error'] = data.get('train_01_error')
        res['stoch_01_error'] = data.get('stochastic_01_error')
        res['stoch_01_mc'] = data.get('stochastic_01_error_mc')
        res['stoch_01_wired'] = data.get('stochastic_01_error_wired')

        # --- CE Metrics (Cross Entropy / NLL) ---
        res['risk_ce_lhs'] = data.get('bound_ce_lhs')
        res['bound_ce_rhs'] = data.get('bound_ce_rhs')
        res['emp_ce_loss'] = data.get('empirical_nll_loss') # NLL is CE loss here
        res['train_ce_loss'] = data.get('train_nll_loss')
        
        # Stochastic CE Metrics
        res['stoch_ce_loss'] = data.get('stochastic_loss')
        res['stoch_ce_mc'] = data.get('stochastic_loss_mc')
        res['stoch_ce_wired'] = data.get('stochastic_loss_wired')
        
        # --- Bound Components & Constants ---
        res['kl_term'] = data.get('kl_final')
        res['channel_term'] = data.get('channel_term')
        res['lipschitz'] = data.get('Lipschitz_constant')
        res['dimension'] = data.get('dimension')
        res['n_bound'] = data.get('n_bound')
        
        return res
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def check_bound_validity(lhs, rhs):
    """Returns a checkmark if lhs <= rhs, else cross."""
    if lhs is None or rhs is None:
        return "N/A"
    return "✓" if lhs <= rhs else "✗"

def main():
    data_rows = []

    if not os.path.exists(RESULTS_DIR):
        print(f"Directory {RESULTS_DIR} not found. Please ensure the path is correct.")
        return

    print(f"Scanning {RESULTS_DIR}...")

    # --- 1. Parsing Loop ---
    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        config = parse_folder_name(folder_name)
        
        # Filtering
        if config['prior_type'] != 'learnt': continue
        if config['objective'] == 'channel': continue

        bound_dir = os.path.join(folder_path, 'bounds')
        if not os.path.exists(bound_dir): continue

        json_files = glob.glob(os.path.join(bound_dir, '*.json'))

        for json_file in json_files:
            filename = os.path.basename(json_file)
            
            spec_part = filename.split('_chan')[0]
            c_type, c_param, _ = parse_channel_specs(spec_part)
            
            if c_type not in ['bec', 'rayleigh_zf']: continue
            
            norm_match = re.search(r"norm-([a-zA-Z0-9]+)", filename)
            final_norm_type = norm_match.group(1) if norm_match else 'unknown'

            metrics = process_json_file(json_file)
            if not metrics: continue

            valid_ce = check_bound_validity(metrics['risk_ce_lhs'], metrics['bound_ce_rhs'])
            valid_01 = check_bound_validity(metrics['risk_01_lhs'], metrics['bound_01_rhs'])

            obj_rank = OBJECTIVE_ORDER.get(config['objective'], 99)

            if c_type == 'rayleigh_zf':
                formatted_param = f"{c_param} dB"
            elif c_type == 'bec':
                formatted_param = f"outage {c_param}"
            else:
                formatted_param = str(c_param)

            row = {
                'Model': config['model_full'],
                'Dataset': config['dataset'],
                'Epochs': config['train_epochs'],
                'Channel Type': c_type,
                'Channel Param': formatted_param,
                'Sort Param': c_param,
                'Norm Type': final_norm_type,
                'Obj Rank': obj_rank,
                'Objective': config['objective'],
                'Learning Rate': config['learning_rate'],
                'KL Penalty': config['kl_penalty'],
                'Chan Penalty': config['channel_penalty'],
                
                # Metrics needed for comparison later
                'Pop Loss CE (MC)': metrics['stoch_ce_mc'],
                'Pop Err 01 (MC)': metrics['stoch_01_mc'],
                
                # Full row data
                'Risk CE (LHS)': metrics['risk_ce_lhs'],
                'Bound CE (RHS)': metrics['bound_ce_rhs'],
                'Valid CE': valid_ce,
                'Emp Loss CE': metrics['emp_ce_loss'],
                
                'Risk 01 (LHS)': metrics['risk_01_lhs'],
                'Bound 01 (RHS)': metrics['bound_01_rhs'],
                'Valid 01': valid_01,
                'Emp Err 01': metrics['emp_01_error'],
                
                'KL Term': metrics['kl_term'],
                'Channel Term': metrics['channel_term'],
                'Lipschitz': metrics['lipschitz'],
                'Dimension': metrics['dimension'],
                
                'Train Loss CE': metrics['train_ce_loss'],
                'Pop Loss CE': metrics['stoch_ce_loss'],
                'Pop Loss CE (Wired)': metrics['stoch_ce_wired'],

                'Train Err 01': metrics['train_01_error'],
                'Pop Err 01': metrics['stoch_01_error'],
                'Pop Err 01 (Wired)': metrics['stoch_01_wired'],
                
                'Folder': folder_name,
                'File': filename
            }
            
            data_rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data_rows)

    if df.empty:
        print("No matching results found.")
        return

    # --- 2. Effectiveness Comparison Logic ---
    print("Calculating effectiveness vs Vanilla...")
    
    # Build Lookup for Vanilla Baselines
    # Key: (Model, Dataset, Epochs, Channel Type, Sort Param, Norm Type)
    baseline_lookup = {}
    vanilla_df = df[df['Objective'] == 'vanilla']
    for _, row in vanilla_df.iterrows():
        key = (row['Model'], row['Dataset'], row['Epochs'], row['Channel Type'], row['Sort Param'], row['Norm Type'])
        baseline_lookup[key] = {
            'ce': row['Pop Loss CE (MC)'],
            '01': row['Pop Err 01 (MC)']
        }

    def check_effectiveness(row, metric_type):
        # Comparison logic: <= Vanilla is effective (✓)
        if row['Objective'] == 'vanilla':
            return "-" # Self
        
        key = (row['Model'], row['Dataset'], row['Epochs'], row['Channel Type'], row['Sort Param'], row['Norm Type'])
        baseline = baseline_lookup.get(key)
        
        if not baseline:
            return "?" # Baseline missing for this config
            
        val = row['Pop Loss CE (MC)'] if metric_type == 'ce' else row['Pop Err 01 (MC)']
        base_val = baseline[metric_type]
        
        if val is None or base_val is None:
            return "N/A"
            
        return "✓" if val <= base_val else "✗"

    # Add Columns
    df['Eff CE'] = df.apply(lambda r: check_effectiveness(r, 'ce'), axis=1)
    df['Eff 01'] = df.apply(lambda r: check_effectiveness(r, '01'), axis=1)

    print(f"Total parsed records: {len(df)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 3. Grouping and Saving ---
    groups = df.groupby(['Model', 'Dataset', 'Channel Type'])

    for (model, dataset, channel_type), group_df in groups:
        filename_base = f"comparison_{model}_{dataset}_{channel_type}"
        csv_path = os.path.join(OUTPUT_DIR, f"{filename_base}.csv")
        xlsx_path = os.path.join(OUTPUT_DIR, f"{filename_base}.xlsx")
        md_path = os.path.join(OUTPUT_DIR, f"{filename_base}.md")

        # Sorting
        sorted_df = group_df.sort_values(
            by=['Sort Param', 'Norm Type', 'Epochs', 'Obj Rank', 'Objective', 'Learning Rate',  'KL Penalty', 'Chan Penalty'], 
            ascending=[True, True, True, True, True, True, True, True]
        )

        # Drop helper columns
        display_df = sorted_df.drop(columns=['Obj Rank', 'Sort Param'])

        # Reorder Columns to place Eff next to Valid
        # Identify columns we want in specific order
        cols = list(display_df.columns)
        
        # Helper to move col_name to after target_name
        def move_col(df_cols, col_name, target_name):
            if col_name in df_cols and target_name in df_cols:
                df_cols.remove(col_name)
                idx = df_cols.index(target_name)
                df_cols.insert(idx + 1, col_name)
            return df_cols

        cols = move_col(cols, 'Eff CE', 'Valid CE')
        cols = move_col(cols, 'Eff 01', 'Valid 01')
        
        display_df = display_df[cols]

        # Save CSV
        display_df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

        # Save Excel with Highlighting
        try:
            # We attempt to use xlsxwriter for highlighting
            with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
                display_df.to_excel(writer, sheet_name='Comparison', index=False)
                workbook = writer.book
                worksheet = writer.sheets['Comparison']
                
                # Format: Light Yellow for Vanilla rows
                vanilla_format = workbook.add_format({'bg_color': '#FFFFE0', 'bold': False})

                metric_format = workbook.add_format({'bg_color': '#E0F7FA', 'bold': False})
                wired_format = workbook.add_format({'bg_color': '#F1F8E9', 'bold': False})

                # Find the column index for 'Objective'
                # pandas headers are row 0, data starts row 1
                obj_col_idx = display_df.columns.get_loc('Objective')
                
                # Convert index to Excel column letter (e.g., 0 -> A, 26 -> AA)
                from xlsxwriter.utility import xl_col_to_name
                col_char = xl_col_to_name(obj_col_idx)
                
                (max_row, max_col) = display_df.shape
                
                # Apply conditional format to the entire data range
                # Range: A2 : LastColumn LastRow
                rng = f"A2:{xl_col_to_name(max_col-1)}{max_row+1}"
                
                worksheet.conditional_format(rng, {
                    'type': 'formula',
                    'criteria': f'=${col_char}2="vanilla"',
                    'format': vanilla_format
                })

                # List of headers you want to highlight
                metric_headers = ['Pop Loss CE (MC)', 'Pop Err 01 (MC)']

                # Loop through the DataFrame columns to find indices
                for col_name in metric_headers:
                    if col_name in display_df.columns:
                        # Get the integer index of the column (0, 1, 2...)
                        col_idx = display_df.columns.get_loc(col_name)
                        
                        # Apply format to the column
                        # args: (first_col, last_col, width, cell_format)
                        # We set width=20 for better visibility, and apply the highlight format
                        worksheet.conditional_format(1, col_idx, max_row, col_idx, {
                            'type': 'formula',
                            'criteria': '=TRUE',
                            'format': metric_format
                        })
                        worksheet.set_column(col_idx, col_idx, 15)

                wired_headers = ['Pop Loss CE (Wired)', 'Pop Err 01 (Wired)']

                # Loop through the DataFrame columns to find indices
                for col_name in wired_headers:
                    if col_name in display_df.columns:
                        # Get the integer index of the column (0, 1, 2...)
                        col_idx = display_df.columns.get_loc(col_name)
                        
                        # Apply format to the column
                        # args: (first_col, last_col, width, cell_format)
                        # We set width=20 for better visibility, and apply the highlight format
                        worksheet.conditional_format(1, col_idx, max_row, col_idx, {
                            'type': 'formula',
                            'criteria': '=TRUE',
                            'format': wired_format
                        })
                        worksheet.set_column(col_idx, col_idx, 15)

                worksheet.freeze_panes(1, 0)  # Freeze header row

                worksheet.autofilter(0, 0, max_row, max_col - 1)  # Enable autofilter
                
            print(f"Saved {xlsx_path} (with highlighting)")
        except Exception as e:
            # Fallback to standard save if xlsxwriter fails
            print(f"Excel formatting failed ({e}), saving plain Excel...")
            display_df.to_excel(xlsx_path, index=False)
            print(f"Saved {xlsx_path}")

        # Save Markdown
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(display_df.to_markdown(index=False))
            print(f"Saved {md_path}")
        except Exception as e:
            print(f"Could not save Markdown: {e}")

        # Preview
        try:
            print(f"\n--- Preview: {model} on {dataset} ({channel_type}) ---")
            preview_cols = ['Objective', 'Channel Param', 'Risk CE (LHS)', 'Valid CE', 'Eff CE', 'Pop Loss CE (MC)']
            avail = [c for c in preview_cols if c in display_df.columns]
            print(display_df[avail].head(10).to_string(index=False))
        except Exception:
            pass

if __name__ == "__main__":
    main()