import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality parameters
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (7, 2.5)

# Classification results at maximum budget
datasets_cls = ['Iris', 'Wine', 'Breast Cancer']
passive_cls = [0.9583, 0.9916, 0.9855]
sensitivity_cls = [0.9400, 0.9722, 0.9579]
entropy_cls = [0.9600, 0.9722, 0.9544]
margin_cls = [0.9667, 0.9667, 0.9561]
least_conf_cls = [0.9600, 0.9722, 0.9579]

x = np.arange(len(datasets_cls))
width = 0.15

fig, ax = plt.subplots()
ax.bar(x - 2*width, passive_cls, width, label='Passive', color='#2E86AB', edgecolor='black', linewidth=0.5)
ax.bar(x - width, sensitivity_cls, width, label='Sensitivity', color='#A23B72', edgecolor='black', linewidth=0.5)
ax.bar(x, entropy_cls, width, label='Entropy', color='#F18F01', edgecolor='black', linewidth=0.5)
ax.bar(x + width, margin_cls, width, label='Margin', color='#C73E1D', edgecolor='black', linewidth=0.5)
ax.bar(x + 2*width, least_conf_cls, width, label='Least Conf.', color='#6A994E', edgecolor='black', linewidth=0.5)

ax.set_ylabel('Accuracy')
ax.set_xlabel('Dataset')
ax.set_xticks(x)
ax.set_xticklabels(datasets_cls)
ax.set_ylim([0.92, 1.0])
ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('../report/figures/cls_final_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('../report/figures/cls_final_comparison.png', bbox_inches='tight', dpi=300)
plt.close()

# Regression results at maximum budget
datasets_reg = ['Diabetes', 'Wine Quality', 'California']
passive_reg = [55.84, 0.658, 0.539]
sensitivity_reg = [139.17, 0.892, 1.002]
uncertainty_reg = [143.94, 1.668, np.nan]  # No California uncertainty data

fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))

for idx, (dataset, passive_val, sens_val, unc_val) in enumerate(zip(
    datasets_reg, passive_reg, sensitivity_reg, uncertainty_reg)):
    
    methods = ['Passive', 'Sensitivity', 'Uncertainty']
    values = [passive_val, sens_val, unc_val]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = axes[idx].bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    axes[idx].set_title(dataset, fontsize=10, pad=8)
    axes[idx].set_ylabel('RMSE' if idx == 0 else '')
    axes[idx].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    axes[idx].tick_params(axis='x', rotation=45)
    
    # Mark best performer with star
    if not np.isnan(values[0]):
        best_idx = np.nanargmin(values)
        axes[idx].plot(best_idx, values[best_idx], marker='*', 
                      markersize=10, color='gold', markeredgecolor='black', 
                      markeredgewidth=0.5, zorder=3)

plt.tight_layout()
plt.savefig('../report/figures/reg_final_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('../report/figures/reg_final_comparison.png', bbox_inches='tight', dpi=300)
plt.close()

print("Figures saved successfully!")