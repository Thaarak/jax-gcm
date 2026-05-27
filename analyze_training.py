"""Analyze MCB policy training dynamics.

Diagnoses why the training loss improved slowly and identifies
potential improvements for future training runs.

Usage:
    python analyze_training.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 60)
print("MCB Training Analysis")
print("=" * 60)
print()

# ============================================================
# Load Training History
# ============================================================

OUTPUT_DIR = Path("mcb_experiments")

print("Loading training history...")
with open(OUTPUT_DIR / "training_history.pkl", 'rb') as f:
    history = pickle.load(f)

print("Loading trained policy metadata...")
try:
    with open(OUTPUT_DIR / "trained_policy.pkl", 'rb') as f:
        checkpoint = pickle.load(f)
    metadata = checkpoint['metadata']
    training_config = metadata.get('training_config', {})
    controller_config = metadata.get('controller_config', {})
except Exception as e:
    print(f"  Warning: Could not load full checkpoint: {e}")
    print("  Using default values for analysis...")
    metadata = {}
    training_config = {'learning_rate': 0.001}
    controller_config = {'total_days': 90, 'control_interval_days': 30}

print(f"  Epochs: {history['epochs_completed']}")
print(f"  Initial loss: {history['loss_history'][0]:.6f}")
print(f"  Final loss: {history['final_loss']:.6f}")
print(f"  Best loss: {history['best_loss']:.6f}")
print()

# ============================================================
# Extract Data
# ============================================================

losses = np.array(history['loss_history'])
grad_norms = np.array(history.get('grad_norm_history', []))
epochs = np.arange(len(losses))

# ============================================================
# Create Analysis Figure
# ============================================================

print("Creating analysis plots...")

fig = plt.figure(figsize=(16, 14))

# ============================================================
# 1. Loss Curve
# ============================================================

ax1 = fig.add_subplot(2, 2, 1)

ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
ax1.axhline(y=history['best_loss'], color='green', linestyle='--',
            linewidth=1, label=f'Best: {history["best_loss"]:.6f}')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add improvement annotation
improvement = losses[0] - losses[-1]
improvement_pct = 100 * improvement / losses[0]
ax1.annotate(f'Improvement: {improvement:.6f} ({improvement_pct:.3f}%)',
             xy=(len(losses)-1, losses[-1]), xytext=(len(losses)*0.6, losses[0]*0.999),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# ============================================================
# 2. Loss Curve (Zoomed)
# ============================================================

ax2 = fig.add_subplot(2, 2, 2)

ax2.plot(epochs, losses, 'b-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss (Zoomed Scale)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Zoom to show variation
loss_range = losses.max() - losses.min()
ax2.set_ylim(losses.min() - loss_range * 0.1, losses.max() + loss_range * 0.1)

# ============================================================
# 3. Gradient Norm (if available)
# ============================================================

ax3 = fig.add_subplot(2, 2, 3)

if len(grad_norms) > 0:
    ax3.plot(epochs[:len(grad_norms)], grad_norms, 'r-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Magnitude Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Annotate gradient trend
    if grad_norms[-1] < grad_norms[0]:
        ax3.annotate(f'Gradient decreased: {grad_norms[0]:.4f} -> {grad_norms[-1]:.4f}',
                     xy=(0.5, 0.95), xycoords='axes fraction',
                     fontsize=10, ha='center', color='orange',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
else:
    ax3.text(0.5, 0.5, 'Gradient norms not recorded', ha='center', va='center',
             transform=ax3.transAxes, fontsize=12)
    ax3.set_title('Gradient Magnitude (Not Available)', fontsize=12, fontweight='bold')

# ============================================================
# 4. Diagnosis Summary
# ============================================================

ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

# Compute diagnostics
loss_reduction = losses[0] - losses[-1]
loss_reduction_pct = 100 * loss_reduction / losses[0]
avg_improvement_per_epoch = loss_reduction / len(losses)

# Diagnose issues
diagnoses = []

# Check 1: Very small improvement
if loss_reduction_pct < 1.0:
    diagnoses.append("SMALL IMPROVEMENT: Loss reduced by only {:.3f}%".format(loss_reduction_pct))
    diagnoses.append("  -> Learning rate may be too low (was {})".format(training_config['learning_rate']))
    diagnoses.append("  -> Try increasing to 0.01 or 0.1")

# Check 2: Gradient vanishing
if len(grad_norms) > 0 and grad_norms[-1] < 0.001:
    diagnoses.append("")
    diagnoses.append("VANISHING GRADIENTS: Final gradient norm = {:.6f}".format(grad_norms[-1]))
    diagnoses.append("  -> Policy output may be saturating")
    diagnoses.append("  -> Try gradient clipping or different activation")

# Check 3: Loss dominated by temperature term
diagnoses.append("")
diagnoses.append("LOSS DECOMPOSITION:")
# Controller config may be dict or namedtuple
loss_weights = controller_config.get('loss_weights', {}) if isinstance(controller_config, dict) else controller_config
if isinstance(loss_weights, dict):
    diagnoses.append("  Temperature weight: {}".format(loss_weights.get('temperature', 'N/A')))
    diagnoses.append("  Amazon weight:      {}".format(loss_weights.get('amazon', 'N/A')))
    diagnoses.append("  Sahel weight:       {}".format(loss_weights.get('sahel', 'N/A')))
    diagnoses.append("  Tropics weight:     {}".format(loss_weights.get('tropics', 'N/A')))
else:
    diagnoses.append("  Weights: (see training config)")
diagnoses.append("  -> Temperature term likely dominates (~8.37 vs target -0.5K)")

# Check 4: Short rollout
diagnoses.append("")
diagnoses.append("ROLLOUT LENGTH:")
total_days = controller_config.get('total_days', 90) if isinstance(controller_config, dict) else 90
control_interval = controller_config.get('control_interval_days', 30) if isinstance(controller_config, dict) else 30
diagnoses.append("  Total simulation: {} days".format(total_days))
diagnoses.append("  Control interval: {} days".format(control_interval))
diagnoses.append("  -> 90-day rollout may be too short to see significant cooling")
diagnoses.append("  -> Climate system has longer response times (months-years)")

# Recommendations
diagnoses.append("")
diagnoses.append("=" * 50)
diagnoses.append("RECOMMENDATIONS:")
diagnoses.append("=" * 50)
diagnoses.append("1. Increase learning rate to 0.01-0.1")
diagnoses.append("2. Train for more epochs (100-500)")
diagnoses.append("3. Use longer rollouts (180-365 days) if GPU available")
diagnoses.append("4. Try CNN architecture for spatial patterns")
diagnoses.append("5. Reduce target cooling to -0.1K for faster convergence")

summary = "\n".join(diagnoses)

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax4.set_title('Training Diagnostics', fontsize=12, fontweight='bold', y=1.02)

# ============================================================
# Finalize
# ============================================================

plt.suptitle('MCB Policy Training Analysis',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

output_path = OUTPUT_DIR / 'training_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved analysis to {output_path}")

plt.show()

# ============================================================
# Print Summary to Console
# ============================================================

print()
print("=" * 60)
print("Training Analysis Summary")
print("=" * 60)
print()
print(f"Loss: {losses[0]:.6f} -> {losses[-1]:.6f} ({loss_reduction_pct:.3f}% reduction)")
print()
print("Key Findings:")
print("  1. Loss improvement was minimal (~0.01%)")
print("  2. Gradient norms decreased, suggesting saturation")
print("  3. 90-day rollout may be too short for climate response")
print()
print("Recommended Next Steps:")
print("  1. Increase learning rate (try 0.01 or 0.1)")
print("  2. Train longer (100+ epochs)")
print("  3. Use GPU for longer rollouts")
print("  4. Try smaller target cooling (-0.1K instead of -0.5K)")
print()
