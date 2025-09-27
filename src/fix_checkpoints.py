#!/usr/bin/env python3
"""
Script to fix checkpoint loading issues in all tuning notebooks.
This script will update the checkpoint logic to properly handle resuming from the correct dataset/config.
"""

import os
import json
import re
from pathlib import Path

def fix_checkpoint_logic(content):
    """Fix the checkpoint loading logic in notebook content."""
    
    # Pattern to match the checkpoint loading section
    checkpoint_pattern = r'# Load checkpoint if exists\ncheckpoint_file = os\.path\.join\([^)]+\)\nif os\.path\.exists\(checkpoint_file\):\n    with open\(checkpoint_file, \'r\'\) as f:\n        checkpoint = json\.load\(f\)\n    print\(f"Resuming from checkpoint: \{checkpoint\[\'completed_configs\'\]\} configs completed"\)\nelse:\n    checkpoint = \{\'completed_configs\': 0, \'results\': \{\}\}\n    print\("Starting fresh run"\)\n\nBEST = checkpoint\.get\(\'results\', \{\}\)'
    
    # Replacement with fixed logic
    fixed_checkpoint = '''# Load checkpoint if exists
checkpoint_file = os.path.join(SAVE_DIR, 'passive_reg_checkpoint.json')
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    print(f"Resuming from checkpoint: {checkpoint.get('completed_configs', 0)} configs completed")
    BEST = checkpoint.get('results', {})
    
    # Determine which dataset to resume from
    dataset_configs = len(LR) * len(WD) * len(HIDDEN) * len(BS)
    completed_configs = checkpoint.get('completed_configs', 0)
    
    # Find which dataset we should resume from
    resume_dataset_idx = completed_configs // dataset_configs
    resume_config_idx = completed_configs % dataset_configs
    
    print(f"Resuming from dataset {resume_dataset_idx} ({DATASETS[resume_dataset_idx] if resume_dataset_idx < len(DATASETS) else 'completed'}), config {resume_config_idx}")
    
else:
    checkpoint = {'completed_configs': 0, 'results': {}}
    BEST = {}
    resume_dataset_idx = 0
    resume_config_idx = 0
    print("Starting fresh run")'''
    
    # Replace the checkpoint loading section
    content = re.sub(checkpoint_pattern, fixed_checkpoint, content, flags=re.MULTILINE | re.DOTALL)
    
    # Fix the main loop to handle resuming properly
    loop_pattern = r'for dataset in DATASETS:'
    fixed_loop = '''# Process datasets starting from the resume point
for dataset_idx, dataset in enumerate(DATASETS):'''
    
    content = content.replace(loop_pattern, fixed_loop)
    
    # Add the dataset skipping logic
    dataset_logic = '''    # Determine starting point for this dataset
    if dataset_idx < resume_dataset_idx:
        # This dataset is already completed, skip it
        print(f"Skipping {dataset} (already completed)")
        continue
    elif dataset_idx == resume_dataset_idx:
        # This is the dataset we need to resume from
        start_config_idx = resume_config_idx
        print(f"Resuming {dataset} from config {start_config_idx + 1}/{dataset_configs}")
    else:
        # This dataset hasn't been started yet
        start_config_idx = 0
        print(f"Starting {dataset} from config 1/{dataset_configs}")'''
    
    # Find where to insert the dataset logic
    insert_point = content.find('    dataset_configs = len(LR) * len(WD) * len(HIDDEN) * len(BS)')
    if insert_point != -1:
        # Find the end of the dataset_configs line
        end_line = content.find('\n', insert_point)
        content = content[:end_line] + '\n' + dataset_logic + content[end_line:]
    
    # Fix the config loop to handle skipping
    config_loop_pattern = r'    for lr, wd, hidden, bs in itertools\.product\(LR, WD, HIDDEN, BS\):'
    fixed_config_loop = '''    config_count = 0
    for lr, wd, hidden, bs in itertools.product(LR, WD, HIDDEN, BS):
        # Skip configs that were already completed
        if config_count < start_config_idx:
            config_count += 1
            continue'''
    
    content = content.replace(config_loop_pattern, fixed_config_loop)
    
    # Add config_count increment at the end of the loop
    content = content.replace('        with open(checkpoint_file, \'w\') as f:\n            json.dump(checkpoint, f, indent=2)', 
                             '        with open(checkpoint_file, \'w\') as f:\n            json.dump(checkpoint, f, indent=2)\n        \n        config_count += 1')
    
    return content

def fix_notebook(notebook_path):
    """Fix a single notebook file."""
    print(f"Fixing {notebook_path}")
    
    try:
        with open(notebook_path, 'r') as f:
            content = f.read()
        
        # Check if this is a regression notebook
        if 'passive_reg' in str(notebook_path):
            # Uncomment the code first
            content = content.replace('# # Load checkpoint if exists', '# Load checkpoint if exists')
            content = content.replace('# checkpoint_file', 'checkpoint_file')
            content = content.replace('# if os.path.exists', 'if os.path.exists')
            content = content.replace('#     with open', '    with open')
            content = content.replace('#     print', '    print')
            content = content.replace('# else:', 'else:')
            content = content.replace('#     checkpoint', '    checkpoint')
            content = content.replace('#     print', '    print')
            content = content.replace('# BEST = checkpoint', 'BEST = checkpoint')
            content = content.replace('# # Calculate total configs', '# Calculate total configs')
            content = content.replace('# total_configs = len', 'total_configs = len')
            content = content.replace('# start_time = time.time()', 'start_time = time.time()')
            content = content.replace('# for dataset in DATASETS:', 'for dataset in DATASETS:')
            content = content.replace('#     if dataset not in BEST:', '    if dataset not in BEST:')
            content = content.replace('#         BEST[dataset] = {"best_cfg": None, "best_metric": np.inf, "history": []}', '        BEST[dataset] = {"best_cfg": None, "best_metric": np.inf, "history": []}')
            content = content.replace('#     print(f"\\n=== Tuning {dataset} ===")', '    print(f"\\n=== Tuning {dataset} ===")')
            content = content.replace('#     best_metric = BEST[dataset]["best_metric"]', '    best_metric = BEST[dataset]["best_metric"]')
            content = content.replace('#     best_cfg = BEST[dataset]["best_cfg"]', '    best_cfg = BEST[dataset]["best_cfg"]')
            content = content.replace('#     hist = BEST[dataset]["history"]', '    hist = BEST[dataset]["history"]')
            content = content.replace('#     dataset_configs = len(LR) * len(WD) * len(HIDDEN) * len(BS)', '    dataset_configs = len(LR) * len(WD) * len(HIDDEN) * len(BS)')
            content = content.replace('#     # Create progress bar for this dataset', '    # Create progress bar for this dataset')
            content = content.replace('#     pbar = tqdm(total=dataset_configs, desc=f"{dataset} configs",', '    pbar = tqdm(total=dataset_configs, desc=f"{dataset} configs",')
            content = content.replace('#                 initial=len(hist), position=0, leave=True)', '                initial=len(hist), position=0, leave=True)')
            content = content.replace('#     for lr, wd, hidden, bs in itertools.product(LR, WD, HIDDEN, BS):', '    for lr, wd, hidden, bs in itertools.product(LR, WD, HIDDEN, BS):')
            content = content.replace('#         config_idx = len(hist) + 1', '        config_idx = len(hist) + 1')
            content = content.replace('#         print(f\'Config {config_idx}/{dataset_configs}: lr={lr}, wd={wd}, hidden={hidden}, bs={bs}\')', '        print(f\'Config {config_idx}/{dataset_configs}: lr={lr}, wd={wd}, hidden={hidden}, bs={bs}\')')
            content = content.replace('#         res = evaluate_config_cv(dataset, lr, wd, hidden, bs)', '        res = evaluate_config_cv(dataset, lr, wd, hidden, bs)')
            content = content.replace('#         res.update({"lr": lr, "wd": wd, "hidden": hidden, "bs": bs})', '        res.update({"lr": lr, "wd": wd, "hidden": hidden, "bs": bs})')
            content = content.replace('#         hist.append(res)', '        hist.append(res)')
            content = content.replace('#         if res[\'rmse_mean\'] < best_metric:', '        if res[\'rmse_mean\'] < best_metric:')
            content = content.replace('#             best_metric = res[\'rmse_mean\']', '            best_metric = res[\'rmse_mean\']')
            content = content.replace('#             best_cfg = {"lr": lr, "wd": wd, "hidden": hidden, "bs": bs}', '            best_cfg = {"lr": lr, "wd": wd, "hidden": hidden, "bs": bs}')
            content = content.replace('#         # Update progress bar', '        # Update progress bar')
            content = content.replace('#         pbar.update(1)', '        pbar.update(1)')
            content = content.replace('#         pbar.set_postfix({\'best_rmse\': f"{best_metric:.4f}"})', '        pbar.set_postfix({\'best_rmse\': f"{best_metric:.4f}"})')
            content = content.replace('#         # Save checkpoint after each config', '        # Save checkpoint after each config')
            content = content.replace('#         checkpoint[\'completed_configs\'] += 1', '        checkpoint[\'completed_configs\'] += 1')
            content = content.replace('#         BEST[dataset] = {"best_cfg": best_cfg, "best_metric": best_metric, "history": hist}', '        BEST[dataset] = {"best_cfg": best_cfg, "best_metric": best_metric, "history": hist}')
            content = content.replace('#         with open(checkpoint_file, \'w\') as f:', '        with open(checkpoint_file, \'w\') as f:')
            content = content.replace('#             json.dump(checkpoint, f, indent=2)', '            json.dump(checkpoint, f, indent=2)')
            content = content.replace('#     pbar.close()', '    pbar.close()')
            content = content.replace('#     BEST[dataset] = {"best_cfg": best_cfg, "best_metric": best_metric, "history": hist}', '    BEST[dataset] = {"best_cfg": best_cfg, "best_metric": best_metric, "history": hist}')
            content = content.replace('#     print(f"Best config for {dataset}: {best_cfg} (RMSE: {best_metric:.4f})")', '    print(f"Best config for {dataset}: {best_cfg} (RMSE: {best_metric:.4f})")')
            content = content.replace('# # Save final results', '# Save final results')
            content = content.replace('# with open(os.path.join(SAVE_DIR, \'passive_reg_best.json\'), \'w\') as f:', 'with open(os.path.join(SAVE_DIR, \'passive_reg_best.json\'), \'w\') as f:')
            content = content.replace('#     json.dump(BEST, f, indent=2)', '    json.dump(BEST, f, indent=2)')
            content = content.replace('# # Clean up checkpoint file', '# Clean up checkpoint file')
            content = content.replace('# if os.path.exists(checkpoint_file):', 'if os.path.exists(checkpoint_file):')
            content = content.replace('#     os.remove(checkpoint_file)', '    os.remove(checkpoint_file)')
            content = content.replace('# total_time = time.time() - start_time', 'total_time = time.time() - start_time')
            content = content.replace('# print(f"\\nTotal time: {total_time/3600:.2f} hours")', 'print(f"\\nTotal time: {total_time/3600:.2f} hours")')
            content = content.replace('# print(f"Average time per config: {total_time/total_configs:.2f} seconds")', 'print(f"Average time per config: {total_time/total_configs:.2f} seconds")')
        
        # Apply the checkpoint fix
        content = fix_checkpoint_logic(content)
        
        # Write back to file
        with open(notebook_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed {notebook_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {notebook_path}: {e}")
        return False

def main():
    """Main function to fix all notebooks."""
    script_dir = Path(__file__).parent
    
    # List of notebooks to fix
    notebooks = [
        "passive_reg_tuning.ipynb",
        "passive_cls_tuning.ipynb",
    ]
    
    print("🔧 Fixing checkpoint loading issues in all notebooks...")
    
    success_count = 0
    for notebook in notebooks:
        notebook_path = script_dir / notebook
        if notebook_path.exists():
            if fix_notebook(notebook_path):
                success_count += 1
        else:
            print(f"⚠️  {notebook} not found, skipping...")
    
    print(f"\n✅ Fixed {success_count}/{len(notebooks)} notebooks")
    print("🎉 Checkpoint loading issues have been resolved!")

if __name__ == "__main__":
    main()
