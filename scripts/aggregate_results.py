#!/usr/bin/env python3
"""
Aggregate results from multi-GPU evaluation runs and create combined plots.

Usage:
    python scripts/aggregate_results.py --results_dir ./results/mask_prob_eval
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime


def load_all_results(results_dir):
    """Load all results.json files from subdirectories."""
    all_results = []
    
    # Find all results.json files
    # Pattern 1: results_dir/gpu*_mask*/timestamp/results.json
    # Pattern 2: results_dir/timestamp/results.json
    patterns = [
        os.path.join(results_dir, 'gpu*_mask*', '*', 'results.json'),
        os.path.join(results_dir, 'gpu*_mask*', 'results.json'),
        os.path.join(results_dir, '*', 'results.json'),
    ]
    
    result_files = []
    for pattern in patterns:
        result_files.extend(glob(pattern))
    
    # Remove duplicates
    result_files = list(set(result_files))
    
    if not result_files:
        print(f"No results.json files found in {results_dir}")
        print(f"Searched patterns: {patterns}")
        return []
    
    print(f"Found {len(result_files)} result files:")
    for f in sorted(result_files):
        print(f"  - {f}")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    all_results.append(data)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return all_results


def merge_duplicate_configs(all_results):
    """Merge results with same mask_prob and noise_sigma (average them)."""
    merged = {}
    
    for r in all_results:
        key = (r['mask_prob'], r['noise_sigma'])
        if key not in merged:
            merged[key] = r
        else:
            # Average the results
            existing = merged[key]
            existing['total_nmse_all'].extend(r.get('total_nmse_all', [r['total_nmse_mean']]))
            existing['total_nmse_mean'] = np.mean(existing['total_nmse_all'])
            existing['total_nmse_std'] = np.std(existing['total_nmse_all'])
    
    return list(merged.values())


def plot_combined_results(all_results, save_dir, channel_names=None):
    """Create combined plots from aggregated results."""
    
    if channel_names is None:
        channel_names = ['AoA_1', 'AoA_2', 'AoA_3', 'Amp_1', 'Amp_2', 'Amp_3']
    
    # Sort by mask_prob
    all_results = sorted(all_results, key=lambda x: (x['noise_sigma'], x['mask_prob']))
    
    # Get unique values
    mask_probs = sorted(list(set(r['mask_prob'] for r in all_results)))
    noise_sigmas = sorted(list(set(r['noise_sigma'] for r in all_results)))
    
    print(f"\nAggregated data:")
    print(f"  Mask probabilities: {mask_probs}")
    print(f"  Noise sigmas: {noise_sigmas}")
    print(f"  Total configs: {len(all_results)}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine if this is a grid evaluation
    grid_mode = len(noise_sigmas) > 1 and len(mask_probs) > 1
    
    if grid_mode:
        _plot_grid_results(all_results, save_dir, mask_probs, noise_sigmas, channel_names)
    else:
        _plot_single_sweep_results(all_results, save_dir, mask_probs, noise_sigmas, channel_names)
    
    # Save combined results
    results_path = os.path.join(save_dir, 'combined_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {results_path}")


def _plot_single_sweep_results(all_results, save_dir, mask_probs, noise_sigmas, channel_names):
    """Plot results for single parameter sweep (mask prob or noise sigma)."""
    
    # Plot 1: Total NMSE vs Mask Probability
    fig1, ax1 = plt.subplots(figsize=(5,4))
    
    for ns in noise_sigmas:
        results = [r for r in all_results if r['noise_sigma'] == ns]
        results = sorted(results, key=lambda x: x['mask_prob'])
        
        mps = [r['mask_prob'] for r in results]
        total_means = [r['total_nmse_mean'] for r in results]
        total_stds = [r['total_nmse_std'] for r in results]
        
        label = f'σ={ns:.3f}' if len(noise_sigmas) > 1 else None
        ax1.errorbar(mps, total_means, yerr=total_stds, 
                     marker='o', capsize=5, linewidth=2, markersize=8, label=label)
    
    ax1.set_xlabel('Mask Probability (% missing)', fontsize=12)
    ax1.set_ylabel('NMSE', fontsize=12)
    # ax1.set_title('Total NMSE vs Mask Probability (Combined Results)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(mask_probs)
    ax1.set_xticklabels([f'{p*100:.0f}%' for p in mask_probs])
    if len(noise_sigmas) > 1:
        ax1.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_nmse_vs_mask_prob.pdf'), dpi=150)
    plt.close()
    print(f"Saved: combined_nmse_vs_mask_prob.pdf")
    
    # Plot 2: Per-channel NMSE
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    
    # Use results from first noise sigma
    ns = noise_sigmas[0]
    results = [r for r in all_results if r['noise_sigma'] == ns]
    results = sorted(results, key=lambda x: x['mask_prob'])
    mps = [r['mask_prob'] for r in results]
    
    for c_idx in range(6):
        c_key = str(c_idx)
        if c_key in results[0].get('channel_nmse_mean', {}):
            means = [r['channel_nmse_mean'][c_key] for r in results]
            stds = [r['channel_nmse_std'][c_key] for r in results]
            
            label = channel_names[c_idx] if c_idx < len(channel_names) else f'Channel {c_idx+1}'
            ax2.errorbar(mps, means, yerr=stds, 
                        marker='o', capsize=3, linewidth=2, markersize=6,
                        label=label, color=colors[c_idx])
    
    ax2.set_xlabel('Mask Probability (% missing)', fontsize=12)
    ax2.set_ylabel('NMSE', fontsize=12)
    # ax2.set_title('Per-Channel NMSE vs Mask Probability', fontsize=14)
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(mask_probs)
    ax2.set_xticklabels([f'{p*100:.0f}%' for p in mask_probs])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_nmse_per_channel.pdf'), dpi=150)
    plt.close()
    print(f"Saved: combined_nmse_per_channel.pdf")
    
    # Plot 3: AoA vs Amplitude
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    
    aoa_means = []
    aoa_stds = []
    amp_means = []
    amp_stds = []
    
    for r in results:
        aoa_vals = [r['channel_nmse_mean'].get(str(i), 0) for i in range(3)]
        amp_vals = [r['channel_nmse_mean'].get(str(i), 0) for i in range(3, 6)]
        
        aoa_means.append(np.mean(aoa_vals))
        amp_means.append(np.mean(amp_vals))
        aoa_stds.append(np.std(aoa_vals))
        amp_stds.append(np.std(amp_vals))
    
    ax3.errorbar(mps, aoa_means, yerr=aoa_stds, 
                 marker='s', capsize=5, linewidth=2, markersize=8, 
                 label='AoA (avg)', color='blue')
    ax3.errorbar(mps, amp_means, yerr=amp_stds, 
                 marker='^', capsize=5, linewidth=2, markersize=8, 
                 label='Amplitude (avg)', color='red')
    
    ax3.set_xlabel('Mask Probability (% missing)', fontsize=12)
    ax3.set_ylabel('NMSE', fontsize=12)
    # ax3.set_title('AoA vs Amplitude NMSE Comparison', fontsize=14)
    ax3.legend(loc='best', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(mask_probs)
    ax3.set_xticklabels([f'{p*100:.0f}%' for p in mask_probs])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_nmse_aoa_vs_amp.pdf'), dpi=150)
    plt.close()
    print(f"Saved: combined_nmse_aoa_vs_amp")


def _plot_grid_results(all_results, save_dir, mask_probs, noise_sigmas, channel_names):
    """Plot results for grid evaluation (mask_prob x noise_sigma)."""
    
    # Create heatmap
    nmse_matrix = np.zeros((len(noise_sigmas), len(mask_probs)))
    
    for r in all_results:
        try:
            mp_idx = mask_probs.index(r['mask_prob'])
            ns_idx = noise_sigmas.index(r['noise_sigma'])
            nmse_matrix[ns_idx, mp_idx] = r['total_nmse_mean']
        except ValueError:
            continue
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(nmse_matrix, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(mask_probs)))
    ax.set_xticklabels([f'{p*100:.0f}%' for p in mask_probs])
    ax.set_yticks(range(len(noise_sigmas)))
    ax.set_yticklabels([f'{s:.3f}' for s in noise_sigmas])
    
    ax.set_xlabel('Mask Probability (% missing)', fontsize=12)
    ax.set_ylabel('Noise Sigma', fontsize=12)
    # ax.set_title('Total NMSE Heatmap (Combined Results)', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NMSE', fontsize=12)
    
    # Add annotations
    for i in range(len(noise_sigmas)):
        for j in range(len(mask_probs)):
            if nmse_matrix[i, j] > 0:
                ax.text(j, i, f'{nmse_matrix[i, j]:.2e}',
                       ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_nmse_heatmap.pdf'), dpi=150)
    plt.close()
    print(f"Saved: combined_nmse_heatmap.pdf")
    
    # Also create line plots
    _plot_single_sweep_results(all_results, save_dir, mask_probs, noise_sigmas, channel_names)


def print_summary_table(all_results, channel_names=None):
    """Print a summary table of all results."""
    
    if channel_names is None:
        channel_names = ['AoA_1', 'AoA_2', 'AoA_3', 'Amp_1', 'Amp_2', 'Amp_3']
    
    all_results = sorted(all_results, key=lambda x: (x['noise_sigma'], x['mask_prob']))
    
    print("\n" + "="*120)
    print("COMBINED RESULTS SUMMARY")
    print("="*120)
    
    header = f"{'Mask %':>8} | {'Noise σ':>10} | {'Total NMSE':>12} | " + " | ".join([f'{n:>10}' for n in channel_names])
    print(header)
    print("-" * len(header))
    
    for r in all_results:
        row = f"{r['mask_prob']*100:>7.0f}% | {r['noise_sigma']:>10.4f} | {r['total_nmse_mean']:>12.4e} | "
        channel_vals = []
        for i in range(6):
            val = r.get('channel_nmse_mean', {}).get(str(i), 0)
            channel_vals.append(f"{val:>10.4e}")
        row += " | ".join(channel_vals)
        print(row)
    
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-GPU evaluation results')
    parser.add_argument('--results_dir', type=str, default='./results/mask_prob_eval',
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for combined results (default: results_dir/combined)')
    args = parser.parse_args()
    
    # Load all results
    all_results = load_all_results(args.results_dir)
    
    if not all_results:
        print("No results to aggregate!")
        return
    
    # Merge duplicates if any
    all_results = merge_duplicate_configs(all_results)
    
    # Print summary
    print_summary_table(all_results)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.results_dir, f'combined_{timestamp}')
    
    # Plot combined results
    plot_combined_results(all_results, output_dir)
    
    print(f"\nDone! Combined results saved to: {output_dir}")


if __name__ == '__main__':
    main()
