# Running All Tuning Notebooks

This script (`run_all_tuning.py`) executes all the `*_tuning.ipynb` notebooks sequentially.

## Usage

```bash
# From the src directory
python run_all_tuning.py

# Or make it executable and run directly
chmod +x run_all_tuning.py
./run_all_tuning.py
```

## What it does

The script will run the following notebooks in order:

1. **passive_reg_tuning.ipynb** - Passive regression hyperparameter tuning
2. **passive_cls_tuning.ipynb** - Passive classification hyperparameter tuning  
3. **reg_uncertainty_tuning.ipynb** - Regression uncertainty-based active learning
4. **cls_uncertainty_tuning.ipynb** - Classification uncertainty-based active learning
5. **reg_sensitivity_tuning.ipynb** - Regression sensitivity-based active learning
6. **cls_sensitivity_tuning.ipynb** - Classification sensitivity-based active learning

## Features

- **Progress tracking**: Shows which notebook is currently running
- **Time tracking**: Displays execution time for each notebook
- **Error handling**: Continues with remaining notebooks if one fails
- **Summary report**: Shows success/failure status for all notebooks
- **Checkpoint support**: Each notebook has built-in checkpointing to resume if interrupted

## Requirements

- Jupyter notebook environment
- All required Python packages (torch, sklearn, matplotlib, tqdm, etc.)
- Sufficient disk space for results and figures

## Output

- Results saved to `../report/figures/`
- Checkpoint files created during execution (automatically cleaned up)
- Progress bars and timing information displayed in terminal

## Notes

- Each notebook can take several hours to complete
- Total execution time may be 10+ hours depending on your hardware
- The script will skip empty or missing notebooks with a warning
- You can interrupt and restart - notebooks will resume from checkpoints
