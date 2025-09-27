#!/usr/bin/env python3
"""
Script to run all tuning notebooks sequentially.
This script executes all the *_tuning.ipynb notebooks in the correct order.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_notebook(notebook_path):
    """Convert notebook to Python script and run it."""
    print(f"\n{'='*60}")
    print(f"Running: {notebook_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        venv_path = '/home/david/.local/share/virtualenvs/Machine-Learning-441-fvr4FmDE'
        python_path = f'{venv_path}/bin/python'
        
        # Check if virtual environment exists
        if not os.path.exists(python_path):
            print(f"❌ Virtual environment not found at {venv_path}")
            return False
        
        # Convert notebook to Python script
        temp_script = notebook_path.with_suffix('.py')
        
        # Use nbconvert to convert to Python
        convert_result = subprocess.run([
            python_path, '-m', 'jupyter', 'nbconvert', '--to', 'python', 
            '--output', str(temp_script), str(notebook_path)
        ], capture_output=True, text=True, timeout=60)
        
        if convert_result.returncode != 0:
            print(f"❌ Failed to convert notebook to Python script")
            print(f"STDERR: {convert_result.stderr}")
            return False
        
        print(f"✅ Converted notebook to: {temp_script}")
        
        # Run the Python script
        result = subprocess.run([python_path, str(temp_script)], 
                               capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"✅ SUCCESS: {notebook_path} completed in {elapsed/60:.1f} minutes")
            
            # Clean up temporary script
            if temp_script.exists():
                temp_script.unlink()
            
            return True
        else:
            print(f"❌ ERROR: {notebook_path} failed")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
            # Clean up temporary script
            if temp_script.exists():
                temp_script.unlink()
            
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {notebook_path} took longer than 1 hour")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {notebook_path} - {str(e)}")
        return False

def main():
    """Main function to run all tuning notebooks."""
    # Get the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Define the notebooks to run in order
    notebooks = [
        "passive_reg_tuning.ipynb",
        # "passive_cls_tuning.ipynb", 
        # "reg_uncertainty_tuning.ipynb",
        # "cls_uncertainty_tuning.ipynb",
        # "reg_sensitivity_tuning.ipynb",
        # "cls_sensitivity_tuning.ipynb"
    ]
    
    # Check which notebooks exist and are not empty
    valid_notebooks = []
    for notebook in notebooks:
        notebook_path = script_dir / notebook
        if notebook_path.exists():
            # Check if file has content (more than just whitespace)
            with open(notebook_path, 'r') as f:
                content = f.read().strip()
            if len(content) > 10:  # Has actual content
                valid_notebooks.append(notebook)
            else:
                print(f"⚠️  {notebook} exists but appears to be empty, skipping...")
        else:
            print(f"⚠️  {notebook} not found, skipping...")
    
    notebooks = valid_notebooks
    
    print("🚀 Starting sequential execution of all tuning notebooks...")
    print(f"📁 Working directory: {script_dir}")
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    for notebook in notebooks:
        notebook_path = script_dir / notebook
        
        if not notebook_path.exists():
            print(f"⚠️  WARNING: {notebook} not found, skipping...")
            results[notebook] = "NOT_FOUND"
            continue
            
        # Run the notebook
        success = run_notebook(notebook_path)
        results[notebook] = "SUCCESS" if success else "FAILED"
        
        # If a notebook fails, ask user if they want to continue
        if not success:
            print(f"\n❌ {notebook} failed!")
            response = input("Do you want to continue with the remaining notebooks? (y/n): ")
            if response.lower() != 'y':
                print("🛑 Stopping execution as requested.")
                break
    
    # Print summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    for notebook, status in results.items():
        status_emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
        print(f"{status_emoji} {notebook}: {status}")
    
    print(f"\n⏱️  Total execution time: {total_time/3600:.2f} hours")
    
    # Count successes
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    total_count = len(results)
    
    print(f"📈 Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("🎉 All notebooks completed successfully!")
    else:
        print("⚠️  Some notebooks failed. Check the output above for details.")

if __name__ == "__main__":
    main()
