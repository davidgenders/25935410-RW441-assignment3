#!/usr/bin/env bash
set -e

# Set this to your venv path
VENV_PATH="/home/david/.local/share/virtualenvs/Machine-Learning-441-fvr4FmDE/bin/python3"

# Activate the venv
# source "$VENV_PATH/bin/activate"

# # Kernel name (must match the name registered with jupyter kernelspec)
# KERNEL_NAME="python3"

for nb in \
  passive_reg_tuning.ipynb \
#   cls_sensitivity_tuning.ipynb \
#   cls_uncertainty_tuning.ipynb \
#   reg_sensitivity_tuning.ipynb \
#   reg_uncertainty_tuning.ipynb \
#   compare_classification.ipynb \
#   compare_regression.ipynb \
do
  jupyter nbconvert --to notebook --execute "$nb" \
    --inplace \
    --ExecutePreprocessor.kernel_name=$VENV_PATH \
    --ExecutePreprocessor.timeout=0
done