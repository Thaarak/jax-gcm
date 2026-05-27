"""Visualize MCB forcing patterns from trained policy.

Creates globe visualizations showing:
1. Learned MCB albedo perturbation pattern
2. Comparison of baseline vs MCB climate
3. Regional focus on stratocumulus regions

Usage:
    python visualize_mcb_patterns.py

Prerequisites:
    Run evaluate_mcb_policy.py first to generate results.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

print("=" * 60)
print("MCB Pattern Visualization")
print("=" * 60)
print()

# ============================================================
# Load Data
# ============================================================

OUTPUT_DIR = Path("mcb_experiments")

print("Loading evaluation results...")
with open(OUTPUT_DIR / "mcb_evaluation_results.pkl", 'rb') as f:
    results = pickle.load(f)

# Get MCB forcing pattern (final step)
mcb_pattern = results['mcb_forcing']['final_pattern']

# Get coordinates - generate from MCB pattern shape
# T30 grid: 96 longitudes x 48 latitudes
nodal_shape = mcb_pattern.shape
lons = np.linspace(0, 360, nodal_shape[0], endpoint=False)
# Gaussian latitudes for T30 (approximate with linspace)
lats = np.linspace(-87.159, 87.159, nodal_shape[1])

print(f"  MCB pattern shape: {mcb_pattern.shape}")
print(f"  MCB mean: {results['mcb_forcing']['mean']:.6f}")
print(f"  MCB max: {results['mcb_forcing']['max']:.6f}")
print()

# ============================================================
# Create Visualization
# ============================================================

print("Creating visualizations...")

fig = plt.figure(figsize=(18, 22))

# ============================================================
# 1. MCB Forcing Pattern (Main Globe)
# ============================================================

ax1 = fig.add_subplot(3, 2, 1, projection=ccrs.Orthographic(-150, 0))
ax1.set_global()

# MCB pattern - transpose to match (lon, lat) ordering
mesh1 = ax1.pcolormesh(lons, lats, mcb_pattern.T,
                        transform=ccrs.PlateCarree(),
                        cmap='YlOrRd', vmin=0, vmax=0.15)
ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax1.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
ax1.set_title('Learned MCB Forcing Pattern\n(Pacific View)', fontsize=12, fontweight='bold')
cbar1 = plt.colorbar(mesh1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
cbar1.set_label('Albedo Perturbation')

# ============================================================
# 2. MCB Forcing Pattern (Atlantic View)
# ============================================================

ax2 = fig.add_subplot(3, 2, 2, projection=ccrs.Orthographic(-20, 0))
ax2.set_global()

mesh2 = ax2.pcolormesh(lons, lats, mcb_pattern.T,
                        transform=ccrs.PlateCarree(),
                        cmap='YlOrRd', vmin=0, vmax=0.15)
ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax2.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
ax2.set_title('Learned MCB Forcing Pattern\n(Atlantic View)', fontsize=12, fontweight='bold')
cbar2 = plt.colorbar(mesh2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
cbar2.set_label('Albedo Perturbation')

# ============================================================
# 3. MCB Forcing (Flat Map with Stratocumulus Regions)
# ============================================================

ax3 = fig.add_subplot(3, 2, 3, projection=ccrs.PlateCarree())
ax3.set_global()

mesh3 = ax3.pcolormesh(lons, lats, mcb_pattern.T,
                        transform=ccrs.PlateCarree(),
                        cmap='YlOrRd', vmin=0, vmax=0.15)
ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax3.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
ax3.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

# Mark stratocumulus regions
stratocumulus_regions = {
    'SE Pacific': (-30, -10, -110, -70),
    'SE Atlantic': (-25, -5, -15, 15),
    'NE Pacific': (15, 35, -150, -120),
}

for name, (lat_min, lat_max, lon_min, lon_max) in stratocumulus_regions.items():
    ax3.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
             [lat_min, lat_min, lat_max, lat_max, lat_min],
             'b-', linewidth=2, transform=ccrs.PlateCarree())
    ax3.text((lon_min + lon_max) / 2, lat_max + 3, name,
             ha='center', fontsize=8, color='blue',
             transform=ccrs.PlateCarree())

ax3.set_title('MCB Forcing with Stratocumulus Regions Marked', fontsize=12, fontweight='bold')
cbar3 = plt.colorbar(mesh3, ax=ax3, orientation='horizontal', pad=0.08, shrink=0.8)
cbar3.set_label('Albedo Perturbation')

# ============================================================
# 4. Zonal Mean MCB Forcing
# ============================================================

ax4 = fig.add_subplot(3, 2, 4)

# Compute zonal mean (average over longitude)
zonal_mean = np.mean(mcb_pattern, axis=0)

ax4.plot(lats, zonal_mean, 'r-', linewidth=2)
ax4.fill_between(lats, 0, zonal_mean, alpha=0.3, color='red')
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax4.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

# Mark tropics
ax4.axvline(x=-30, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
ax4.axvline(x=30, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
ax4.text(-30, max(zonal_mean) * 0.9, '30S', ha='center', fontsize=8, color='blue')
ax4.text(30, max(zonal_mean) * 0.9, '30N', ha='center', fontsize=8, color='blue')

ax4.set_xlabel('Latitude')
ax4.set_ylabel('Mean Albedo Perturbation')
ax4.set_title('Zonal Mean MCB Forcing', fontsize=12, fontweight='bold')
ax4.set_xlim(-90, 90)
ax4.grid(True, alpha=0.3)

# ============================================================
# 5. MCB Forcing Histogram
# ============================================================

ax5 = fig.add_subplot(3, 2, 5)

# Flatten MCB pattern
mcb_flat = mcb_pattern.flatten()

ax5.hist(mcb_flat, bins=50, color='red', alpha=0.7, edgecolor='darkred')
ax5.axvline(x=np.mean(mcb_flat), color='blue', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(mcb_flat):.4f}')
ax5.axvline(x=np.max(mcb_flat), color='green', linestyle='--', linewidth=2,
            label=f'Max: {np.max(mcb_flat):.4f}')

ax5.set_xlabel('Albedo Perturbation')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of MCB Forcing Values', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============================================================
# 6. Results Summary
# ============================================================

ax6 = fig.add_subplot(3, 2, 6)
ax6.axis('off')

# Create summary text
summary = f"""
EVALUATION RESULTS SUMMARY
{'=' * 40}

Temperature Changes:
  Global:    {results['temperature']['global_change_K']:+.4f} K
  Target:    {results['temperature']['target_K']:+.2f} K
  Achieved:  {100 * results['temperature']['global_change_K'] / results['temperature']['target_K']:.1f}% of target

Precipitation Changes:
  Global:    {results['precipitation']['global_change_pct']:+.2f}%
  Amazon:    {results['precipitation']['amazon_change_pct']:+.2f}%
  Sahel:     {results['precipitation']['sahel_change_pct']:+.2f}%

MCB Forcing:
  Mean:      {results['mcb_forcing']['mean']:.6f}
  Max:       {results['mcb_forcing']['max']:.6f}

Simulation:
  Duration:  {results['simulation']['total_days']:.0f} days
  Steps:     {results['simulation']['n_steps']}
  Time:      {results['simulation']['elapsed_time_s']:.1f}s
"""

ax6.text(0.1, 0.9, summary, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# ============================================================
# Finalize
# ============================================================

plt.suptitle('MCB Policy Evaluation: Learned Forcing Patterns',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

output_path = OUTPUT_DIR / 'mcb_pattern_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved visualization to {output_path}")

plt.show()
print()
print("Done!")
