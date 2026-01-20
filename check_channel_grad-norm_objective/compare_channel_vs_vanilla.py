import csv
import json
import math
import re
from pathlib import Path
from statistics import median

POSTERIOR_ROOT = Path('results/posterior')


def parse_dir_name(name: str):
    parts = name.split('_')
    model = parts[0]
    dataset = parts[1]
    prior_type = parts[2] if len(parts) > 2 else None

    epoch_match = re.search(r'epoch(\d+)_bs', name)
    epoch = int(epoch_match.group(1)) if epoch_match else None

    objective = None
    if 'objective-' in name:
        tail = name.split('objective-')[1]
        stop_tokens = ['_perc', '_seed', '_epoch-pri', '_bs']
        end_idx = len(tail)
        for tok in stop_tokens:
            pos = tail.find(tok)
            if pos != -1:
                end_idx = min(end_idx, pos)
        candidate = tail[:end_idx]
        if '-chan' in candidate:
            candidate = candidate.split('-chan')[0]
        objective = candidate

    chan_penalty = None
    chan_match = re.search(r'-chan([0-9.]+)', name)
    if chan_match:
        chan_penalty = float(chan_match.group(1))

    kl_penalty = None
    kl_match = re.search(r'-kl([0-9.]+)', name)
    if kl_match:
        kl_penalty = float(kl_match.group(1))

    channel_type = None
    outage = None
    tx_power = None
    noise_var = None

    bec_match = re.search(r'bec-outage([0-9.]+)', name)
    if bec_match:
        channel_type = 'bec'
        outage = float(bec_match.group(1))

    rayleigh_zf_match = re.search(r'rayleigh-zf-tx([0-9.]+)-noise([0-9.]+)', name)
    if rayleigh_zf_match:
        channel_type = 'rayleigh_zf'
        tx_power = float(rayleigh_zf_match.group(1))
        noise_var = float(rayleigh_zf_match.group(2))

    rayleigh_match = re.search(r'rayleigh-tx([0-9.]+)-noise([0-9.]+)', name)
    if rayleigh_match and channel_type is None:
        channel_type = 'rayleigh'
        tx_power = float(rayleigh_match.group(1))
        noise_var = float(rayleigh_match.group(2))

    snr_db = None
    if tx_power is not None and noise_var is not None and noise_var != 0:
        snr = tx_power / noise_var
        if snr > 0:
            snr_db = 10 * math.log10(snr)

    return {
        'model': model,
        'dataset': dataset,
        'prior_type': prior_type,
        'epoch': epoch,
        'objective': objective,
        'chan_penalty': chan_penalty,
        'kl_penalty': kl_penalty,
        'channel_type': channel_type,
        'outage': outage,
        'tx_power': tx_power,
        'noise_var': noise_var,
        'snr_db': snr_db,
    }


def parse_file_name(stem: str):
    channel_type = None
    outage = None
    tx_power = None
    noise_var = None
    norm_type = None

    bec_match = re.search(r'bec-outage([0-9.]+)', stem)
    if bec_match:
        channel_type = 'bec'
        outage = float(bec_match.group(1))

    rayleigh_zf_match = re.search(r'rayleigh-zf-tx([0-9.]+)-noise([0-9.]+)', stem)
    if rayleigh_zf_match:
        channel_type = 'rayleigh_zf'
        tx_power = float(rayleigh_zf_match.group(1))
        noise_var = float(rayleigh_zf_match.group(2))

    rayleigh_match = re.search(r'rayleigh-tx([0-9.]+)-noise([0-9.]+)', stem)
    if rayleigh_match and channel_type is None:
        channel_type = 'rayleigh'
        tx_power = float(rayleigh_match.group(1))
        noise_var = float(rayleigh_match.group(2))

    norm_match = re.search(r'norm-([a-z]+)', stem)
    if norm_match:
        norm_type = norm_match.group(1)

    snr_db = None
    if tx_power is not None and noise_var is not None and noise_var != 0:
        snr = tx_power / noise_var
        if snr > 0:
            snr_db = 10 * math.log10(snr)

    return {
        'channel_type': channel_type,
        'outage': outage,
        'tx_power': tx_power,
        'noise_var': noise_var,
        'snr_db': snr_db,
        'norm_type': norm_type,
    }


def load_result(json_path: Path, meta: dict):
    with open(json_path) as f:
        data = json.load(f)

    lhs = data.get('bound_ce_lhs')
    rhs = data.get('bound_ce_rhs')
    bound_valid = False
    if lhs is not None and rhs is not None:
        if not (isinstance(lhs, float) and math.isnan(lhs)) and not (isinstance(rhs, float) and math.isnan(rhs)):
            bound_valid = lhs <= rhs

    stoch_err = data.get('stochastic_01_error')
    if isinstance(stoch_err, float) and math.isnan(stoch_err):
        stoch_err = None

    return {
        **meta,
        'json_path': str(json_path),
        'bound_ce_lhs': lhs,
        'bound_ce_rhs': rhs,
        'bound_valid': bound_valid,
        'stochastic_01_error': stoch_err,
    }


def collect_results():
    records = []
    for dir_path in POSTERIOR_ROOT.iterdir():
        if not dir_path.is_dir():
            continue
        meta = parse_dir_name(dir_path.name)
        objective = meta['objective']
        if objective not in {'vanilla', 'channel_gradient', 'channel_norm'}:
            continue
        if meta['prior_type'] == 'rand':
            continue
        bounds_dir = dir_path / 'bounds'
        if not bounds_dir.exists():
            continue
        for json_path in bounds_dir.glob('*.json'):
            file_meta = parse_file_name(json_path.stem)
            # Merge channel info: file spec overrides dir spec for vanilla cases
            merged = {**meta}
            for key in ['channel_type', 'outage', 'tx_power', 'noise_var', 'snr_db']:
                if file_meta.get(key) is not None:
                    merged[key] = file_meta[key]
            merged['norm_type'] = file_meta.get('norm_type')
            records.append(load_result(json_path, merged))
    return records


def filter_records(records):
    cleaned = []
    for r in records:
        if r['channel_type'] is None:
            continue
        if r['channel_type'] == 'rayleigh':
            continue
        if r['objective'] != 'vanilla' and r.get('norm_type') == 'spec':
            continue
        if r['objective'] != 'vanilla' and r['chan_penalty'] is None:
            continue
        cleaned.append(r)
    return cleaned


def best_vanilla_baselines(records):
    baselines = {}
    for r in records:
        if r['objective'] != 'vanilla':
            continue
        if r.get('norm_type') and r['norm_type'] != 'frob':
            continue
        if r['stochastic_01_error'] is None:
            continue
        key = (r['model'], r['dataset'], r['epoch'], r['channel_type'], r.get('outage'), r.get('tx_power'), r.get('noise_var'))
        current = baselines.get(key)
        if current is None or r['stochastic_01_error'] < current['stochastic_01_error']:
            baselines[key] = r
    return baselines


def find_improvements(records, baselines):
    channel_rows = []
    improvements = []
    invalid_bounds = []
    for r in records:
        if r['objective'] == 'vanilla':
            continue
        baseline = None
        delta = None
        if r['stochastic_01_error'] is not None:
            key = (r['model'], r['dataset'], r['epoch'], r['channel_type'], r.get('outage'), r.get('tx_power'), r.get('noise_var'))
            baseline = baselines.get(key)
            if baseline and baseline['stochastic_01_error'] is not None:
                delta = baseline['stochastic_01_error'] - r['stochastic_01_error']
        row = {**r, 'baseline_error': baseline['stochastic_01_error'] if baseline else None, 'improvement': delta}
        channel_rows.append(row)
        if delta is not None and delta > 0:
            improvements.append(row)
        if not r['bound_valid']:
            invalid_bounds.append(row)
    return channel_rows, improvements, invalid_bounds


def summarize_by_hparam(improvements, key):
    summary = []
    buckets = {}
    for r in improvements:
        val = r.get(key)
        if val is None:
            continue
        buckets.setdefault(val, []).append(r['improvement'])
    for val, vals in sorted(buckets.items()):
        summary.append((val, len(vals), median(vals)))
    return summary


def write_csv(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'model', 'dataset', 'epoch', 'objective', 'chan_penalty', 'kl_penalty',
        'channel_type', 'outage', 'tx_power', 'noise_var', 'snr_db', 'norm_type',
        'stochastic_01_error', 'baseline_error', 'improvement', 'bound_valid', 'json_path'
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def write_markdown(improvements, invalid_bounds, path: Path):
    lines = []
    lines.append('# Channel vs Vanilla Summary')
    lines.append('')
    lines.append('## Top Improvements (sorted by delta)')
    lines.append('| model | dataset | epoch | objective | chan_penalty | kl_penalty | channel | vanilla_err | channel_err | delta | bound_valid |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(improvements, key=lambda x: x['improvement'], reverse=True)[:50]:
        ch = format_channel(r)
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['epoch']} | {r['objective']} | {r['chan_penalty']} | {r['kl_penalty']} | {ch} | "
            f"{r['baseline_error']:.4f} | {r['stochastic_01_error']:.4f} | {r['improvement']:.4f} | {r['bound_valid']} |"
        )

    lines.append('')
    lines.append('## Invalid Bounds (lhs>rhs or NaN)')
    lines.append('| model | dataset | epoch | objective | chan_penalty | kl_penalty | channel | lhs | rhs | json |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in invalid_bounds[:50]:
        ch = format_channel(r)
        lhs = r.get('bound_ce_lhs')
        rhs = r.get('bound_ce_rhs')
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['epoch']} | {r['objective']} | {r['chan_penalty']} | {r['kl_penalty']} | {ch} | {lhs} | {rhs} | {r['json_path']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))


def format_channel(r):
    if r['channel_type'] == 'bec':
        return f"BEC outage={r['outage']}"
    tx = r.get('tx_power')
    noise = r.get('noise_var')
    snr_db = r.get('snr_db')
    if tx is not None and noise is not None:
        snr_part = f"SNR={snr_db:.2f} dB" if snr_db is not None else f"tx={tx}, noise={noise}"
        return f"Rayleigh-ZF ({snr_part})"
    return r['channel_type']


def main():
    records = collect_results()
    records = filter_records(records)
    objective_counts = {}
    for r in records:
        objective_counts[r['objective']] = objective_counts.get(r['objective'], 0) + 1
    baselines = best_vanilla_baselines(records)
    channel_rows, improvements, invalid_bounds = find_improvements(records, baselines)

    print(f"Total filtered records: {len(records)}")
    print(f"Records by objective: {objective_counts}")
    print(f"Baselines: {len(baselines)}")
    print(f"Channel candidates: {len([r for r in records if r['objective'] != 'vanilla'])}")
    print(f"Improved over vanilla: {len(improvements)}")

    out_dir = Path('check_channel_grad-norm_objective')
    write_csv(improvements, out_dir / 'improvements.csv')
    write_csv(channel_rows, out_dir / 'channel_vs_vanilla_all.csv')
    write_markdown(improvements, invalid_bounds, out_dir / 'results_table.md')

    if improvements:
        print("\nTop improvements (sorted by delta):")
        for r in sorted(improvements, key=lambda x: x['improvement'], reverse=True)[:20]:
            ch = format_channel(r)
            print(
                f"model={r['model']}, dataset={r['dataset']}, epoch={r['epoch']}, "
                f"obj={r['objective']}, chan_penalty={r['chan_penalty']}, kl={r['kl_penalty']}, "
                f"channel={ch}, pop_err={r['stochastic_01_error']:.4f}, "
                f"vanilla_err={r['baseline_error']:.4f}, delta={r['improvement']:.4f}, "
                f"bound_valid={r['bound_valid']}"
            )

    if invalid_bounds:
        print("\nChannel runs with invalid bounds (lhs>rhs or NaN):")
        for r in invalid_bounds[:20]:
            ch = format_channel(r)
            lhs = r['bound_ce_lhs']
            rhs = r['bound_ce_rhs']
            print(
                f"model={r['model']}, dataset={r['dataset']}, epoch={r['epoch']}, obj={r['objective']}, "
                f"chan_penalty={r['chan_penalty']}, kl={r['kl_penalty']}, channel={ch}, lhs={lhs}, rhs={rhs}"
            )

    for key in ['chan_penalty', 'kl_penalty', 'epoch']:
        summary = summarize_by_hparam(improvements, key)
        if summary:
            print(f"\nMedian delta by {key} (value, count, median):")
            for val, count, med in summary:
                print(f"{val}: n={count}, median_improvement={med:.4f}")


if __name__ == '__main__':
    main()
