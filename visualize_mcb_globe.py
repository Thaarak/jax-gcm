"""Visualize MCB on 3D globe projections.

This script creates globe-based visualizations that show MCB deployment
regions and effects on a realistic Earth representation.

Run with: python visualize_mcb_globe.py
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import jax.numpy as jnp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import MCB components
from jcm.mcb import (
    MCBConfig,
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

    # Get lat/lon in degrees
    lats = np.array(jnp.rad2deg(grid.latitudes))
    lons = np.array(jnp.rad2deg(grid.longitudes))

    return coords, grid, lats, lons


def add_earth_features(ax):
    """Add realistic Earth features to a cartopy axis."""
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='darkgray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', linestyle=':')


def plot_mcb_deployment_globe(grid, lats, lons):
    """Plot MCB deployment regions on multiple globe views."""

    fmask = jnp.zeros(grid.nodal_shape)

    # Create combined mask of all stratocumulus regions
    all_regions_mask = np.array(create_stratocumulus_mask(grid, fmask))

    # Create 2D coordinate arrays for plotting
    lon_2d, lat_2d = np.meshgrid(lons, lats, indexing='ij')

    # Create figure with multiple globe views
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('MCB Deployment Regions on Earth', fontsize=18, fontweight='bold', y=0.95)

    # Different viewpoints to show all regions
    viewpoints = [
        ('Pacific View', -120, 0),      # Looking at Pacific
        ('Atlantic View', -20, 0),       # Looking at Atlantic
        ('Global View', -60, 20),        # Tilted global view
    ]

    for idx, (title, center_lon, center_lat) in enumerate(viewpoints):
        ax = fig.add_subplot(1, 3, idx + 1,
                            projection=ccrs.Orthographic(center_lon, center_lat))

        # Add Earth features
        ax.add_feature(cfeature.OCEAN, facecolor='#E6F3FF', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#F5F5DC', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#666666', zorder=2)

        # Plot MCB regions
        mesh = ax.pcolormesh(
            lon_2d, lat_2d, all_regions_mask,
            transform=ccrs.PlateCarree(),
            cmap='YlOrRd',
            alpha=0.8,
            vmin=0, vmax=1,
            zorder=3
        )

        # Add gridlines
        gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5)

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_global()

    # Add colorbar
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
    cbar = plt.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('MCB Deployment Intensity', fontsize=11)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['No MCB', 'Partial', 'Full MCB'])

    # Add legend text
    fig.text(0.5, 0.02,
             'Orange/Red regions: Stratocumulus cloud zones targeted for Marine Cloud Brightening',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.1, 1, 0.93])
    plt.savefig('mcb_globe_deployment.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: mcb_globe_deployment.png")
    plt.show()


def plot_individual_regions_globe(grid, lats, lons):
    """Plot each MCB region on its own globe, centered on that region."""

    fmask = jnp.zeros(grid.nodal_shape)
    lon_2d, lat_2d = np.meshgrid(lons, lats, indexing='ij')

    # Region centers for viewpoints (approximate centers)
    region_centers = {
        'se_pacific': (-90, -15),
        'ne_pacific': (-130, 25),
        'se_atlantic': (0, -12),
        'canary': (-25, 25),
        'benguela': (10, -25),
    }

    region_descriptions = {
        'se_pacific': 'SE Pacific\n(Peru/Chile Coast)',
        'ne_pacific': 'NE Pacific\n(California Coast)',
        'se_atlantic': 'SE Atlantic\n(Namibia/Angola Coast)',
        'canary': 'Canary Current\n(NW Africa)',
        'benguela': 'Benguela Current\n(SW Africa)',
    }

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle('Individual MCB Deployment Regions', fontsize=16, fontweight='bold', y=0.98)

    for idx, (region_name, (center_lon, center_lat)) in enumerate(region_centers.items()):
        ax = fig.add_subplot(1, 5, idx + 1,
                            projection=ccrs.Orthographic(center_lon, center_lat))

        # Get mask for this region
        mask = np.array(create_stratocumulus_mask(grid, fmask, [region_name]))

        # Add Earth features
        ax.add_feature(cfeature.OCEAN, facecolor='#E6F3FF', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#F5F5DC', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#666666', zorder=2)

        # Plot this region's mask
        mesh = ax.pcolormesh(
            lon_2d, lat_2d, mask,
            transform=ccrs.PlateCarree(),
            cmap='Reds',
            alpha=0.85,
            vmin=0, vmax=1,
            zorder=3
        )

        ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.3)
        ax.set_title(region_descriptions[region_name], fontsize=10, pad=5)
        ax.set_global()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('mcb_globe_individual_regions.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: mcb_globe_individual_regions.png")
    plt.show()


def plot_teleconnection_globe(grid, lats, lons):
    """Plot teleconnection monitoring regions on globes."""

    lon_2d, lat_2d = np.meshgrid(lons, lats, indexing='ij')

    # Region centers
    region_info = {
        'amazon': {
            'center': (-60, -5),
            'color': 'Greens',
            'title': 'Amazon Basin\n(Precipitation Monitoring)',
            'concern': 'Rainfall reduction risk'
        },
        'sahel': {
            'center': (10, 15),
            'color': 'Oranges',
            'title': 'Sahel Region\n(Drought Monitoring)',
            'concern': 'Drought intensification risk'
        },
        'south_asia_monsoon': {
            'center': (85, 20),
            'color': 'Purples',
            'title': 'South Asia\n(Monsoon Monitoring)',
            'concern': 'Monsoon disruption risk'
        },
        'north_atlantic': {
            'center': (-35, 52),
            'color': 'Blues',
            'title': 'North Atlantic\n(AMOC Monitoring)',
            'concern': 'Circulation change risk'
        },
    }

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Teleconnection Monitoring Regions\n(Where we watch for MCB side effects)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, (region_name, info) in enumerate(region_info.items()):
        ax = fig.add_subplot(1, 4, idx + 1,
                            projection=ccrs.Orthographic(info['center'][0], info['center'][1]))

        # Get mask
        mask = np.array(create_teleconnection_mask(grid, region_name))

        # Add Earth features
        ax.add_feature(cfeature.OCEAN, facecolor='#E6F3FF', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#F5F5DC', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#666666', zorder=2)

        # Plot region
        mesh = ax.pcolormesh(
            lon_2d, lat_2d, mask,
            transform=ccrs.PlateCarree(),
            cmap=info['color'],
            alpha=0.85,
            vmin=0, vmax=1,
            zorder=3
        )

        ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.3)
        ax.set_title(info['title'], fontsize=10, pad=5)
        ax.set_global()

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig('mcb_globe_teleconnections.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: mcb_globe_teleconnections.png")
    plt.show()


def plot_mcb_effect_globe(grid, lats, lons):
    """Plot MCB albedo change effect on globe."""

    fmask = jnp.zeros(grid.nodal_shape)
    nodal_shape = grid.nodal_shape
    lon_2d, lat_2d = np.meshgrid(lons, lats, indexing='ij')

    # Create MCB config
    se_pacific_mask = create_stratocumulus_mask(grid, fmask, ['se_pacific', 'ne_pacific'])
    mcb_config = MCBConfig.uniform(
        nodal_shape,
        perturbation=0.08,
        region_mask=se_pacific_mask,
    )

    # Compute albedo change
    base_albedo = 0.07
    mcb_albedo = np.array(compute_mcb_sea_albedo(mcb_config, base_albedo, day_of_year=180))
    albedo_change = (mcb_albedo - base_albedo) * 100  # Convert to percentage

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle('MCB Effect: Albedo Change', fontsize=16, fontweight='bold', y=0.98)

    # Three views
    views = [
        ('Before MCB\n(Uniform Ocean Albedo)', -120, 0, np.full(nodal_shape, base_albedo * 100), 'YlOrBr', (5, 15)),
        ('After MCB\n(Enhanced Reflectivity)', -120, 0, mcb_albedo * 100, 'YlOrBr', (5, 15)),
        ('MCB Effect\n(Albedo Increase %)', -120, 0, albedo_change, 'Reds', (0, 10)),
    ]

    for idx, (title, clon, clat, data, cmap, vlim) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1,
                            projection=ccrs.Orthographic(clon, clat))

        # Add Earth features
        ax.add_feature(cfeature.OCEAN, facecolor='#E6F3FF', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#F5F5DC', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#666666', zorder=2)

        # Plot data
        mesh = ax.pcolormesh(
            lon_2d, lat_2d, data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            alpha=0.85,
            vmin=vlim[0], vmax=vlim[1],
            zorder=3
        )

        ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.3)
        ax.set_title(title, fontsize=11, pad=5)
        ax.set_global()

        # Add colorbar below each plot
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                           shrink=0.6, pad=0.05, aspect=20)
        cbar.set_label('Albedo %' if idx < 2 else 'Δ Albedo %', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('mcb_globe_effect.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: mcb_globe_effect.png")
    plt.show()


def plot_combined_summary_globe(grid, lats, lons):
    """Create a comprehensive summary showing deployment and monitoring on one figure."""

    fmask = jnp.zeros(grid.nodal_shape)
    lon_2d, lat_2d = np.meshgrid(lons, lats, indexing='ij')

    # Get masks
    deployment_mask = np.array(create_stratocumulus_mask(grid, fmask))
    amazon_mask = np.array(create_teleconnection_mask(grid, 'amazon'))
    sahel_mask = np.array(create_teleconnection_mask(grid, 'sahel'))

    fig = plt.figure(figsize=(14, 10))

    # Main title
    fig.suptitle('Marine Cloud Brightening: Deployment & Monitoring Overview',
                 fontsize=16, fontweight='bold', y=0.96)

    # Large central globe - combined view
    ax_main = fig.add_subplot(2, 2, (1, 3), projection=ccrs.Orthographic(-60, 0))

    ax_main.add_feature(cfeature.OCEAN, facecolor='#E6F3FF', zorder=0)
    ax_main.add_feature(cfeature.LAND, facecolor='#F5F5DC', edgecolor='none', zorder=1)
    ax_main.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#666666', zorder=2)

    # Plot deployment regions (red)
    ax_main.pcolormesh(
        lon_2d, lat_2d, deployment_mask,
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        alpha=0.7,
        vmin=0, vmax=1,
        zorder=3
    )

    # Plot Amazon (green overlay)
    ax_main.pcolormesh(
        lon_2d, lat_2d, np.where(amazon_mask > 0, 1, np.nan),
        transform=ccrs.PlateCarree(),
        cmap='Greens',
        alpha=0.6,
        vmin=0, vmax=1,
        zorder=4
    )

    # Plot Sahel (orange overlay)
    ax_main.pcolormesh(
        lon_2d, lat_2d, np.where(sahel_mask > 0, 1, np.nan),
        transform=ccrs.PlateCarree(),
        cmap='Oranges',
        alpha=0.6,
        vmin=0, vmax=1,
        zorder=4
    )

    ax_main.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3)
    ax_main.set_title('Combined View: Deployment (Red) & Monitoring (Green/Orange)', fontsize=11)
    ax_main.set_global()

    # Side panel 1: Pacific deployment
    ax_pacific = fig.add_subplot(2, 2, 2, projection=ccrs.Orthographic(-110, 0))
    ax_pacific.add_feature(cfeature.OCEAN, facecolor='#E6F3FF', zorder=0)
    ax_pacific.add_feature(cfeature.LAND, facecolor='#F5F5DC', edgecolor='none', zorder=1)
    ax_pacific.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#666666', zorder=2)

    pacific_mask = np.array(create_stratocumulus_mask(grid, fmask, ['se_pacific', 'ne_pacific']))
    ax_pacific.pcolormesh(
        lon_2d, lat_2d, pacific_mask,
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        alpha=0.8,
        vmin=0, vmax=1,
        zorder=3
    )
    ax_pacific.set_title('Pacific MCB Zones', fontsize=10)
    ax_pacific.set_global()

    # Side panel 2: Amazon monitoring
    ax_amazon = fig.add_subplot(2, 2, 4, projection=ccrs.Orthographic(-60, -5))
    ax_amazon.add_feature(cfeature.OCEAN, facecolor='#E6F3FF', zorder=0)
    ax_amazon.add_feature(cfeature.LAND, facecolor='#F5F5DC', edgecolor='none', zorder=1)
    ax_amazon.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#666666', zorder=2)

    ax_amazon.pcolormesh(
        lon_2d, lat_2d, amazon_mask,
        transform=ccrs.PlateCarree(),
        cmap='Greens',
        alpha=0.8,
        vmin=0, vmax=1,
        zorder=3
    )
    ax_amazon.set_title('Amazon Monitoring Zone', fontsize=10)
    ax_amazon.set_global()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.7, label='MCB Deployment Zones'),
        Patch(facecolor='#6BCB77', alpha=0.7, label='Amazon (Precip. Monitoring)'),
        Patch(facecolor='#FFB347', alpha=0.7, label='Sahel (Drought Monitoring)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.savefig('mcb_globe_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: mcb_globe_summary.png")
    plt.show()


def plot_rotating_view(grid, lats, lons):
    """Create multiple rotation angles to see the full picture."""

    fmask = jnp.zeros(grid.nodal_shape)
    lon_2d, lat_2d = np.meshgrid(lons, lats, indexing='ij')

    # Get all deployment regions
    deployment_mask = np.array(create_stratocumulus_mask(grid, fmask))

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('MCB Deployment Regions: Full Global View', fontsize=16, fontweight='bold', y=0.98)

    # Six views rotating around the globe
    rotation_angles = [
        (0, 'Africa/Europe'),
        (60, 'Indian Ocean'),
        (120, 'Western Pacific'),
        (180, 'Central Pacific'),
        (-120, 'Eastern Pacific'),
        (-60, 'Americas'),
    ]

    for idx, (center_lon, label) in enumerate(rotation_angles):
        ax = fig.add_subplot(1, 6, idx + 1,
                            projection=ccrs.Orthographic(center_lon, 10))

        ax.add_feature(cfeature.OCEAN, facecolor='#E6F3FF', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#F5F5DC', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#666666', zorder=2)

        ax.pcolormesh(
            lon_2d, lat_2d, deployment_mask,
            transform=ccrs.PlateCarree(),
            cmap='YlOrRd',
            alpha=0.8,
            vmin=0, vmax=1,
            zorder=3
        )

        ax.set_title(label, fontsize=9, pad=3)
        ax.set_global()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('mcb_globe_rotation.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: mcb_globe_rotation.png")
    plt.show()


def main():
    """Run all globe visualizations."""
    print("=" * 60)
    print("MCB Globe Visualization Suite")
    print("=" * 60)
    print()

    # Setup
    print("Setting up coordinates...")
    coords, grid, lats, lons = setup_coordinates()
    print(f"Grid shape: {grid.nodal_shape}")
    print()

    # Generate visualizations
    print("Generating globe visualizations...")
    print("-" * 40)

    print("\n1. MCB deployment regions on globe...")
    plot_mcb_deployment_globe(grid, lats, lons)

    print("\n2. Individual deployment regions...")
    plot_individual_regions_globe(grid, lats, lons)

    print("\n3. Teleconnection monitoring regions...")
    plot_teleconnection_globe(grid, lats, lons)

    print("\n4. MCB albedo effect...")
    plot_mcb_effect_globe(grid, lats, lons)

    print("\n5. Combined summary view...")
    plot_combined_summary_globe(grid, lats, lons)

    print("\n6. Full rotation view...")
    plot_rotating_view(grid, lats, lons)

    print()
    print("=" * 60)
    print("All globe visualizations complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  - mcb_globe_deployment.png      (Main deployment regions)")
    print("  - mcb_globe_individual_regions.png (Each region separately)")
    print("  - mcb_globe_teleconnections.png (Monitoring regions)")
    print("  - mcb_globe_effect.png          (Albedo change effect)")
    print("  - mcb_globe_summary.png         (Combined overview)")
    print("  - mcb_globe_rotation.png        (360° view)")
    print()
    print("View with: open mcb_globe_summary.png")


if __name__ == "__main__":
    main()
