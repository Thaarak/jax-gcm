"""Visualize MCB baseline simulation on globe projections.

Creates globe visualizations of key climate variables from the
1-year baseline simulation.

Usage:
    python visualize_baseline_globe.py
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load data
print("Loading baseline simulation data...")
ds = xr.open_dataset('mcb_experiments/baseline_predictions.nc')

# Get coordinates
lons = ds.lon.values
lats = ds.lat.values

# Create figure with multiple globe views
fig = plt.figure(figsize=(16, 20))

# ============================================================
# 1. Surface Temperature (Annual Mean)
# ============================================================
ax1 = fig.add_subplot(3, 2, 1, projection=ccrs.Orthographic(-60, 20))
ax1.set_global()

# Compute annual mean surface temperature
tsfc = ds['surface_flux.tsfc'].mean(dim='time').values
tsfc_celsius = tsfc - 273.15

# Plot
mesh1 = ax1.pcolormesh(lons, lats, tsfc_celsius.T,
                        transform=ccrs.PlateCarree(),
                        cmap='RdYlBu_r', vmin=-40, vmax=35)
ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax1.set_title('Surface Temperature (Annual Mean)', fontsize=12, fontweight='bold')
cbar1 = plt.colorbar(mesh1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
cbar1.set_label('Temperature (°C)')

# ============================================================
# 2. Surface Temperature (Different View - Pacific)
# ============================================================
ax2 = fig.add_subplot(3, 2, 2, projection=ccrs.Orthographic(-150, 0))
ax2.set_global()

mesh2 = ax2.pcolormesh(lons, lats, tsfc_celsius.T,
                        transform=ccrs.PlateCarree(),
                        cmap='RdYlBu_r', vmin=-40, vmax=35)
ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax2.set_title('Surface Temperature (Pacific View)', fontsize=12, fontweight='bold')
cbar2 = plt.colorbar(mesh2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
cbar2.set_label('Temperature (°C)')

# ============================================================
# 3. Total Precipitation (Annual Mean)
# ============================================================
ax3 = fig.add_subplot(3, 2, 3, projection=ccrs.Orthographic(-60, 0))
ax3.set_global()

# Compute annual mean precipitation
precip = (ds['convection.precnv'] + ds['condensation.precls']).mean(dim='time').values
# Convert to more intuitive units (multiply by scale factor for mm/day approximation)
precip_scaled = precip * 1000  # Scale for visualization

mesh3 = ax3.pcolormesh(lons, lats, precip_scaled.T,
                        transform=ccrs.PlateCarree(),
                        cmap='Blues', vmin=0, vmax=100)
ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax3.set_title('Precipitation (Annual Mean)', fontsize=12, fontweight='bold')
cbar3 = plt.colorbar(mesh3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
cbar3.set_label('Precipitation (scaled units)')

# ============================================================
# 4. Cloud Cover (Annual Mean)
# ============================================================
ax4 = fig.add_subplot(3, 2, 4, projection=ccrs.Orthographic(100, 20))
ax4.set_global()

clouds = ds['shortwave_rad.cloudc'].mean(dim='time').values

mesh4 = ax4.pcolormesh(lons, lats, clouds.T,
                        transform=ccrs.PlateCarree(),
                        cmap='Greys', vmin=0, vmax=1)
ax4.add_feature(cfeature.COASTLINE, linewidth=0.5, color='blue')
ax4.set_title('Cloud Cover (Annual Mean)', fontsize=12, fontweight='bold')
cbar4 = plt.colorbar(mesh4, ax=ax4, orientation='horizontal', pad=0.05, shrink=0.8)
cbar4.set_label('Cloud Fraction')

# ============================================================
# 5. Zonal Wind at Surface Level (Annual Mean)
# ============================================================
ax5 = fig.add_subplot(3, 2, 5, projection=ccrs.Orthographic(0, 45))
ax5.set_global()

# Surface level wind (level 0 is surface after flipping)
u_wind = ds['u_wind'].isel(level=0).mean(dim='time').values

mesh5 = ax5.pcolormesh(lons, lats, u_wind.T,
                        transform=ccrs.PlateCarree(),
                        cmap='RdBu_r', vmin=-20, vmax=20)
ax5.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax5.set_title('Surface Zonal Wind (Annual Mean)', fontsize=12, fontweight='bold')
cbar5 = plt.colorbar(mesh5, ax=ax5, orientation='horizontal', pad=0.05, shrink=0.8)
cbar5.set_label('U-wind (m/s)')

# ============================================================
# 6. Specific Humidity at Surface (Annual Mean)
# ============================================================
ax6 = fig.add_subplot(3, 2, 6, projection=ccrs.Orthographic(-90, -20))
ax6.set_global()

# Surface humidity
q = ds['specific_humidity'].isel(level=0).mean(dim='time').values

mesh6 = ax6.pcolormesh(lons, lats, q.T,
                        transform=ccrs.PlateCarree(),
                        cmap='YlGnBu', vmin=0, vmax=20)
ax6.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax6.set_title('Surface Specific Humidity (Annual Mean)', fontsize=12, fontweight='bold')
cbar6 = plt.colorbar(mesh6, ax=ax6, orientation='horizontal', pad=0.05, shrink=0.8)
cbar6.set_label('Specific Humidity (g/kg)')

# ============================================================
# Finalize
# ============================================================
plt.suptitle('JCM Baseline Climate Simulation (1 Year)\nNo MCB Forcing Applied',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

# Save
output_path = 'mcb_experiments/baseline_globe_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved visualization to {output_path}")

plt.show()
