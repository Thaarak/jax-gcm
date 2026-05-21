"""Visualize MCB (Marine Cloud Brightening) forcing and its effects.

This script creates visualizations to help understand:
1. Where MCB deployment regions are located
2. How MCB changes ocean surface albedo
3. The spatial pattern of MCB forcing

Run with: python visualize_mcb.py
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import jax.numpy as jnp

# Import MCB components
from jcm.mcb import (
    MCBConfig,
    create_region_mask,
    create_stratocumulus_mask,
    create_teleconnection_mask,
    compute_mcb_sea_albedo,
    STRATOCUMULUS_REGIONS,
    TELECONNECTION_REGIONS,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords


def setup_coordinates():
    """Set up the model coordinates."""
    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    grid = coords.horizontal

    # Get lat/lon in degrees for plotting
    lats = np.array(jnp.rad2deg(grid.latitudes))
    lons = np.array(jnp.rad2deg(grid.longitudes))

    # Create 2D meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')

    return coords, grid, lats, lons, lon_grid, lat_grid


def plot_stratocumulus_regions(grid, lats, lons, lon_grid, lat_grid):
    """Plot all stratocumulus MCB deployment regions."""

    # Create fake ocean mask (all ocean for visualization)
    fmask = jnp.zeros(grid.nodal_shape)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('MCB Deployment Regions (Stratocumulus Zones)', fontsize=14, fontweight='bold')

    regions = list(STRATOCUMULUS_REGIONS.keys())

    for idx, region_name in enumerate(regions):
        ax = axes[idx // 3, idx % 3]

        # Get mask for this region
        mask = np.array(create_stratocumulus_mask(grid, fmask, [region_name]))

        # Plot
        im = ax.pcolormesh(lon_grid, lat_grid, mask, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'{region_name.replace("_", " ").title()}', fontsize=12)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)

        # Add region bounds as text
        bounds = STRATOCUMULUS_REGIONS[region_name]
        ax.axhline(y=bounds[0], color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=bounds[1], color='red', linestyle='--', alpha=0.5)

        # Add gridlines
        ax.grid(True, alpha=0.3)

    # Use last subplot for combined view
    ax = axes[1, 2]
    combined_mask = np.array(create_stratocumulus_mask(grid, fmask, regions))
    im = ax.pcolormesh(lon_grid, lat_grid, combined_mask, cmap='Blues', vmin=0, vmax=1)
    ax.set_title('All Regions Combined', fontsize=12)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcb_deployment_regions.png', dpi=150, bbox_inches='tight')
    print("Saved: mcb_deployment_regions.png")
    plt.show()


def plot_teleconnection_regions(grid, lats, lons, lon_grid, lat_grid):
    """Plot teleconnection monitoring regions (Amazon, Sahel, etc.)."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Teleconnection Monitoring Regions\n(Where we watch for MCB side effects)',
                 fontsize=14, fontweight='bold')

    regions = list(TELECONNECTION_REGIONS.keys())
    colors = ['Greens', 'Oranges', 'Purples', 'Reds']

    for idx, region_name in enumerate(regions):
        ax = axes[idx // 2, idx % 2]

        # Get mask for this region
        mask = np.array(create_teleconnection_mask(grid, region_name))

        # Plot
        im = ax.pcolormesh(lon_grid, lat_grid, mask, cmap=colors[idx], vmin=0, vmax=1)
        ax.set_title(f'{region_name.replace("_", " ").title()}', fontsize=12)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.grid(True, alpha=0.3)

        # Add annotation about why this region matters
        if region_name == 'amazon':
            ax.text(0.02, 0.98, 'Critical: Precipitation', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', color='darkgreen')
        elif region_name == 'sahel':
            ax.text(0.02, 0.98, 'Critical: Drought risk', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', color='darkorange')

    plt.tight_layout()
    plt.savefig('mcb_teleconnection_regions.png', dpi=150, bbox_inches='tight')
    print("Saved: mcb_teleconnection_regions.png")
    plt.show()


def plot_albedo_changes(grid, lats, lons, lon_grid, lat_grid):
    """Plot before/after albedo with MCB forcing."""

    fmask = jnp.zeros(grid.nodal_shape)  # All ocean
    nodal_shape = grid.nodal_shape

    # Create MCB config for SE Pacific
    se_pacific_mask = create_stratocumulus_mask(grid, fmask, ['se_pacific'])
    mcb_config = MCBConfig.uniform(
        nodal_shape,
        perturbation=0.08,  # 8% increase
        region_mask=se_pacific_mask,
    )

    # Base albedo (constant ocean albedo)
    base_albedo = 0.07
    base_albedo_field = np.full(nodal_shape, base_albedo)

    # MCB-enhanced albedo
    mcb_albedo = np.array(compute_mcb_sea_albedo(mcb_config, base_albedo, day_of_year=180))

    # Albedo change
    albedo_change = mcb_albedo - base_albedo_field

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('MCB Effect on Ocean Surface Albedo (SE Pacific Deployment)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Base albedo
    ax = axes[0]
    im = ax.pcolormesh(lon_grid, lat_grid, base_albedo_field, cmap='YlOrRd', vmin=0, vmax=0.2)
    ax.set_title('Before MCB\n(Base Ocean Albedo)', fontsize=11)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    plt.colorbar(im, ax=ax, label='Albedo', shrink=0.8)
    ax.grid(True, alpha=0.3)

    # Plot 2: MCB albedo
    ax = axes[1]
    im = ax.pcolormesh(lon_grid, lat_grid, mcb_albedo, cmap='YlOrRd', vmin=0, vmax=0.2)
    ax.set_title('After MCB\n(Enhanced Albedo)', fontsize=11)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    plt.colorbar(im, ax=ax, label='Albedo', shrink=0.8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Albedo change (the MCB effect)
    ax = axes[2]
    im = ax.pcolormesh(lon_grid, lat_grid, albedo_change, cmap='Reds', vmin=0, vmax=0.1)
    ax.set_title('MCB Effect\n(Albedo Increase)', fontsize=11)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    plt.colorbar(im, ax=ax, label='Δ Albedo', shrink=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcb_albedo_change.png', dpi=150, bbox_inches='tight')
    print("Saved: mcb_albedo_change.png")
    plt.show()


def plot_multi_region_deployment(grid, lats, lons, lon_grid, lat_grid):
    """Plot MCB deployment across multiple regions with different intensities."""

    fmask = jnp.zeros(grid.nodal_shape)
    nodal_shape = grid.nodal_shape

    # Create different MCB intensities for different regions
    se_pacific_mask = np.array(create_stratocumulus_mask(grid, fmask, ['se_pacific']))
    ne_pacific_mask = np.array(create_stratocumulus_mask(grid, fmask, ['ne_pacific']))
    se_atlantic_mask = np.array(create_stratocumulus_mask(grid, fmask, ['se_atlantic']))

    # Create composite forcing: different strengths in different regions
    mcb_forcing = np.zeros(nodal_shape)
    mcb_forcing += se_pacific_mask * 0.10   # 10% in SE Pacific (strongest)
    mcb_forcing += ne_pacific_mask * 0.06   # 6% in NE Pacific
    mcb_forcing += se_atlantic_mask * 0.04  # 4% in SE Atlantic (weakest)

    # Create config
    mcb_config = MCBConfig.from_spatial_field(
        jnp.array(mcb_forcing),
        active_mask=jnp.ones(nodal_shape),
    )

    # Compute albedo
    base_albedo = 0.07
    mcb_albedo = np.array(compute_mcb_sea_albedo(mcb_config, base_albedo, day_of_year=180))
    albedo_change = mcb_albedo - base_albedo

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Multi-Region MCB Deployment Strategy', fontsize=14, fontweight='bold')

    # Plot 1: MCB forcing pattern
    ax = axes[0]
    im = ax.pcolormesh(lon_grid, lat_grid, mcb_forcing * 100, cmap='OrRd', vmin=0, vmax=12)
    ax.set_title('MCB Forcing Pattern\n(Albedo Perturbation %)', fontsize=11)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('MCB Strength (%)')
    ax.grid(True, alpha=0.3)

    # Add region labels
    ax.text(250, -15, 'SE Pacific\n(10%)', fontsize=9, ha='center', color='darkred')
    ax.text(230, 25, 'NE Pacific\n(6%)', fontsize=9, ha='center', color='darkred')
    ax.text(350, -12, 'SE Atlantic\n(4%)', fontsize=9, ha='center', color='darkred')

    # Plot 2: Resulting albedo change
    ax = axes[1]
    im = ax.pcolormesh(lon_grid, lat_grid, albedo_change * 100, cmap='YlOrRd', vmin=0, vmax=12)
    ax.set_title('Resulting Albedo Change\n(% increase in reflectivity)', fontsize=11)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Albedo Change (%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcb_multi_region_deployment.png', dpi=150, bbox_inches='tight')
    print("Saved: mcb_multi_region_deployment.png")
    plt.show()


def plot_seasonal_variation(grid, lats, lons, lon_grid, lat_grid):
    """Plot how MCB forcing can vary seasonally."""

    fmask = jnp.zeros(grid.nodal_shape)
    nodal_shape = grid.nodal_shape

    # Create seasonal weights (stronger in summer, weaker in winter)
    days = np.arange(365)
    # Northern hemisphere summer peak (day ~180)
    seasonal_weights = 0.3 + 0.7 * np.maximum(0, np.cos(2 * np.pi * (days - 180) / 365))

    # Create MCB config with seasonal variation
    se_pacific_mask = create_stratocumulus_mask(grid, fmask, ['se_pacific'])
    mcb_config = MCBConfig.uniform(
        nodal_shape,
        perturbation=0.10,
        region_mask=se_pacific_mask,
        temporal_weights=jnp.array(seasonal_weights),
    )

    base_albedo = 0.07

    # Compute albedo for different seasons
    winter_albedo = np.array(compute_mcb_sea_albedo(mcb_config, base_albedo, day_of_year=0))
    spring_albedo = np.array(compute_mcb_sea_albedo(mcb_config, base_albedo, day_of_year=90))
    summer_albedo = np.array(compute_mcb_sea_albedo(mcb_config, base_albedo, day_of_year=180))
    fall_albedo = np.array(compute_mcb_sea_albedo(mcb_config, base_albedo, day_of_year=270))

    # Create figure
    fig = plt.figure(figsize=(14, 10))

    # Top plot: seasonal weight curve
    ax_top = fig.add_subplot(3, 1, 1)
    ax_top.plot(days, seasonal_weights, 'b-', linewidth=2)
    ax_top.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Winter (Day 0)')
    ax_top.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='Spring (Day 90)')
    ax_top.axvline(x=180, color='red', linestyle='--', alpha=0.5, label='Summer (Day 180)')
    ax_top.axvline(x=270, color='orange', linestyle='--', alpha=0.5, label='Fall (Day 270)')
    ax_top.set_xlabel('Day of Year')
    ax_top.set_ylabel('MCB Strength Multiplier')
    ax_top.set_title('Seasonal MCB Deployment Schedule', fontsize=12, fontweight='bold')
    ax_top.legend(loc='upper right')
    ax_top.set_xlim(0, 365)
    ax_top.set_ylim(0, 1.1)
    ax_top.grid(True, alpha=0.3)

    # Bottom plots: albedo maps for each season
    seasons = [
        ('Winter (Jan)', winter_albedo, 'Blues'),
        ('Spring (Apr)', spring_albedo, 'Greens'),
        ('Summer (Jul)', summer_albedo, 'Reds'),
        ('Fall (Oct)', fall_albedo, 'Oranges'),
    ]

    for idx, (name, albedo, cmap) in enumerate(seasons):
        ax = fig.add_subplot(3, 4, 5 + idx)
        change = albedo - base_albedo
        im = ax.pcolormesh(lon_grid, lat_grid, change * 100, cmap=cmap, vmin=0, vmax=10)
        ax.set_title(f'{name}', fontsize=10)
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Lon (°)', fontsize=8)
        ax.set_ylabel('Lat (°)', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.6, label='Δ Albedo %')

    # Add summary text
    fig.text(0.5, 0.02,
             'MCB strength varies seasonally: strongest in summer (when more solar radiation), weakest in winter',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig('mcb_seasonal_variation.png', dpi=150, bbox_inches='tight')
    print("Saved: mcb_seasonal_variation.png")
    plt.show()


def plot_gradient_sensitivity(grid, lats, lons, lon_grid, lat_grid):
    """Visualize gradient sensitivity - how each grid cell affects the output."""
    import jax

    fmask = jnp.zeros(grid.nodal_shape)
    nodal_shape = grid.nodal_shape

    # Create MCB config
    mcb_config = MCBConfig.uniform(nodal_shape, perturbation=0.05)
    base_albedo = 0.07

    # Define loss function (mean albedo)
    def loss_fn(perturbation):
        cfg = mcb_config.copy(albedo_perturbation=perturbation)
        albedo = compute_mcb_sea_albedo(cfg, base_albedo, day_of_year=180)
        return jnp.mean(albedo)

    # Compute gradient
    grad = jax.grad(loss_fn)(mcb_config.albedo_perturbation)
    grad_np = np.array(grad)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Gradient Visualization: Sensitivity of Output to MCB Forcing',
                 fontsize=14, fontweight='bold')

    # Plot 1: Gradient values
    ax = axes[0]
    im = ax.pcolormesh(lon_grid, lat_grid, grad_np * 1000, cmap='coolwarm',
                       vmin=-1, vmax=1)
    ax.set_title('∂(Mean Albedo)/∂(MCB Perturbation)\n(×1000 for visibility)', fontsize=11)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    plt.colorbar(im, ax=ax, label='Gradient (×1000)', shrink=0.8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute gradient (importance map)
    ax = axes[1]
    im = ax.pcolormesh(lon_grid, lat_grid, np.abs(grad_np) * 1000, cmap='hot',
                       vmin=0, vmax=1)
    ax.set_title('Gradient Magnitude (Importance Map)\nBrighter = More Influential', fontsize=11)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    plt.colorbar(im, ax=ax, label='|Gradient| (×1000)', shrink=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcb_gradient_sensitivity.png', dpi=150, bbox_inches='tight')
    print("Saved: mcb_gradient_sensitivity.png")
    plt.show()


def create_summary_figure(grid, lats, lons, lon_grid, lat_grid):
    """Create a single summary figure showing the key MCB concepts."""

    fmask = jnp.zeros(grid.nodal_shape)
    nodal_shape = grid.nodal_shape

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Marine Cloud Brightening (MCB) Implementation Summary',
                 fontsize=16, fontweight='bold')

    # 1. Deployment regions
    ax1 = fig.add_subplot(2, 3, 1)
    all_regions_mask = np.array(create_stratocumulus_mask(grid, fmask))
    im = ax1.pcolormesh(lon_grid, lat_grid, all_regions_mask, cmap='Blues')
    ax1.set_title('1. MCB Deployment Zones\n(Stratocumulus Regions)', fontsize=11)
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-90, 90)
    ax1.grid(True, alpha=0.3)

    # 2. Teleconnection monitoring
    ax2 = fig.add_subplot(2, 3, 2)
    amazon = np.array(create_teleconnection_mask(grid, 'amazon'))
    sahel = np.array(create_teleconnection_mask(grid, 'sahel'))
    monitoring = amazon + sahel * 0.5
    im = ax2.pcolormesh(lon_grid, lat_grid, monitoring, cmap='RdYlGn_r')
    ax2.set_title('2. Teleconnection Monitoring\n(Amazon & Sahel)', fontsize=11)
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-90, 90)
    ax2.grid(True, alpha=0.3)

    # 3. MCB forcing pattern
    ax3 = fig.add_subplot(2, 3, 3)
    se_pacific_mask = np.array(create_stratocumulus_mask(grid, fmask, ['se_pacific']))
    mcb_forcing = se_pacific_mask * 0.08
    im = ax3.pcolormesh(lon_grid, lat_grid, mcb_forcing * 100, cmap='OrRd', vmin=0, vmax=10)
    ax3.set_title('3. MCB Forcing Pattern\n(8% in SE Pacific)', fontsize=11)
    ax3.set_xlabel('Longitude (°)')
    ax3.set_ylabel('Latitude (°)')
    ax3.set_xlim(0, 360)
    ax3.set_ylim(-90, 90)
    plt.colorbar(im, ax=ax3, label='%', shrink=0.6)
    ax3.grid(True, alpha=0.3)

    # 4. Before MCB
    ax4 = fig.add_subplot(2, 3, 4)
    base_albedo = np.full(nodal_shape, 0.07)
    im = ax4.pcolormesh(lon_grid, lat_grid, base_albedo * 100, cmap='YlOrRd', vmin=5, vmax=20)
    ax4.set_title('4. Ocean Albedo BEFORE MCB\n(~7% everywhere)', fontsize=11)
    ax4.set_xlabel('Longitude (°)')
    ax4.set_ylabel('Latitude (°)')
    ax4.set_xlim(0, 360)
    ax4.set_ylim(-90, 90)
    plt.colorbar(im, ax=ax4, label='Albedo %', shrink=0.6)
    ax4.grid(True, alpha=0.3)

    # 5. After MCB
    ax5 = fig.add_subplot(2, 3, 5)
    mcb_config = MCBConfig.from_spatial_field(jnp.array(mcb_forcing), jnp.ones(nodal_shape))
    mcb_albedo = np.array(compute_mcb_sea_albedo(mcb_config, 0.07, day_of_year=180))
    im = ax5.pcolormesh(lon_grid, lat_grid, mcb_albedo * 100, cmap='YlOrRd', vmin=5, vmax=20)
    ax5.set_title('5. Ocean Albedo AFTER MCB\n(Up to 15% in SE Pacific)', fontsize=11)
    ax5.set_xlabel('Longitude (°)')
    ax5.set_ylabel('Latitude (°)')
    ax5.set_xlim(0, 360)
    ax5.set_ylim(-90, 90)
    plt.colorbar(im, ax=ax5, label='Albedo %', shrink=0.6)
    ax5.grid(True, alpha=0.3)

    # 6. The difference (MCB effect)
    ax6 = fig.add_subplot(2, 3, 6)
    difference = (mcb_albedo - 0.07) * 100
    im = ax6.pcolormesh(lon_grid, lat_grid, difference, cmap='Reds', vmin=0, vmax=10)
    ax6.set_title('6. MCB EFFECT\n(Albedo Increase)', fontsize=11)
    ax6.set_xlabel('Longitude (°)')
    ax6.set_ylabel('Latitude (°)')
    ax6.set_xlim(0, 360)
    ax6.set_ylim(-90, 90)
    plt.colorbar(im, ax=ax6, label='Δ Albedo %', shrink=0.6)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcb_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: mcb_summary.png")
    plt.show()


def main():
    """Run all visualizations."""
    print("=" * 60)
    print("MCB Visualization Suite")
    print("=" * 60)
    print()

    # Setup
    print("Setting up coordinates...")
    coords, grid, lats, lons, lon_grid, lat_grid = setup_coordinates()
    print(f"Grid shape: {grid.nodal_shape}")
    print(f"Latitude range: {lats.min():.1f}° to {lats.max():.1f}°")
    print(f"Longitude range: {lons.min():.1f}° to {lons.max():.1f}°")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 40)

    print("\n1. Stratocumulus deployment regions...")
    plot_stratocumulus_regions(grid, lats, lons, lon_grid, lat_grid)

    print("\n2. Teleconnection monitoring regions...")
    plot_teleconnection_regions(grid, lats, lons, lon_grid, lat_grid)

    print("\n3. Albedo changes from MCB...")
    plot_albedo_changes(grid, lats, lons, lon_grid, lat_grid)

    print("\n4. Multi-region deployment strategy...")
    plot_multi_region_deployment(grid, lats, lons, lon_grid, lat_grid)

    print("\n5. Seasonal variation...")
    plot_seasonal_variation(grid, lats, lons, lon_grid, lat_grid)

    print("\n6. Gradient sensitivity...")
    plot_gradient_sensitivity(grid, lats, lons, lon_grid, lat_grid)

    print("\n7. Summary figure...")
    create_summary_figure(grid, lats, lons, lon_grid, lat_grid)

    print()
    print("=" * 60)
    print("All visualizations complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  - mcb_deployment_regions.png")
    print("  - mcb_teleconnection_regions.png")
    print("  - mcb_albedo_change.png")
    print("  - mcb_multi_region_deployment.png")
    print("  - mcb_seasonal_variation.png")
    print("  - mcb_gradient_sensitivity.png")
    print("  - mcb_summary.png")


if __name__ == "__main__":
    main()
