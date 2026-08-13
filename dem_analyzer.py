#!/usr/bin/env python3
"""
DEM Statistical Fingerprint Analyzer
Extracts Earth-like terrain statistics from a GeoTIFF DEM
to calibrate the procedural tectonic terrain generator.

Usage: python dem_analyzer.py <dem.tif>
Output: terrain_fingerprint.json + console summary
"""

import sys
import json
import numpy as np
from collections import Counter
import math

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════

def compute_slope(data, resolution_m):
    """Finite-difference slope in radians, using 4-neighbor average."""
    dzdx = np.zeros_like(data)
    dzdy = np.zeros_like(data)
    dzdx[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2.0 * resolution_m)
    dzdy[1:-1, :] = (data[2:, :] - data[:-2, :]) / (2.0 * resolution_m)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return np.arctan(slope)


def hypsometric_curve(data, num_bins=200):
    """Cumulative height distribution: fraction >= each height."""
    valid = data[~np.isnan(data)]
    sorted_vals = np.sort(valid)
    n = len(sorted_vals)
    heights = np.linspace(sorted_vals[0], sorted_vals[-1], num_bins)
    fractions = np.array([np.sum(valid >= h) / n for h in heights])
    return heights, fractions


def detect_peaks(data, min_prominence=50.0, neighborhood=5):
    """Detect local maxima with prominence filtering."""
    from scipy.ndimage import maximum_filter
    local_max = (data == maximum_filter(data, size=neighborhood))
    
    peaks = []
    max_y, max_x = data.shape
    for y in range(neighborhood, max_y - neighborhood):
        for x in range(neighborhood, max_x - neighborhood):
            if local_max[y, x] and not np.isnan(data[y, x]):
                val = data[y, x]
                if val <= 0:
                    continue
                # Simple prominence: drop to nearest lower value
                min_in_radius = np.min(data[max(0,y-30):min(max_y,y+30),
                                            max(0,x-30):min(max_x,x+30)])
                prominence = val - min_in_radius
                if prominence >= min_prominence:
                    peaks.append((y, x, val, prominence))
    
    return peaks


def slope_histogram(slope_rad, land_mask, num_bins=50):
    """Slope distribution over land areas only."""
    land_slopes = np.degrees(slope_rad[land_mask & ~np.isnan(slope_rad)])
    counts, edges = np.histogram(land_slopes, bins=num_bins, range=(0, 90))
    return edges, counts


def fractal_dimension_boxcounting(data, threshold=0):
    """Estimate fractal dimension of coastline (0m contour) using box counting."""
    binary = (data > threshold).astype(np.int8)
    h, w = binary.shape
    
    sizes = []
    counts = []
    box_size = 2
    while box_size <= min(h, w) // 2:
        # Count boxes containing both land and water (coastline)
        n_boxes = 0
        for y in range(0, h - box_size, box_size):
            for x in range(0, w - box_size, box_size):
                patch = binary[y:y+box_size, x:x+box_size]
                if patch.max() != patch.min():  # contains both land and water
                    n_boxes += 1
        if n_boxes > 0:
            sizes.append(box_size)
            counts.append(n_boxes)
        box_size *= 2
    
    if len(sizes) < 3:
        return 1.0  # not enough data
    
    # Linear fit: log(N) = -D * log(s) + C
    log_s = np.log(sizes)
    log_n = np.log(counts)
    slope, _ = np.polyfit(log_s, log_n, 1)
    return -slope  # fractal dimension


def compute_drainage_density(data, resolution_m):
    """Approximate drainage density using D8 flow accumulation."""
    h, w = data.shape
    # Fill depressions (simple version: just use raw data for speed)
    
    # D8 flow direction
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, -1, -1, -1, 0, 1, 1, 1]
    dd = [1, math.sqrt(2), 1, math.sqrt(2), 1, math.sqrt(2), 1, math.sqrt(2)]
    
    # Compute flow direction for each cell (land only)
    flow_dir = np.full((h, w), -1, dtype=np.int8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if np.isnan(data[y, x]) or data[y, x] <= 0:
                continue
            max_slope = 0
            best_dir = -1
            for d in range(8):
                nx, ny = x + dx[d], y + dy[d]
                if 0 <= nx < w and 0 <= ny < h and not np.isnan(data[ny, nx]):
                    drop = (data[y, x] - data[ny, nx]) / dd[d]
                    if drop > max_slope:
                        max_slope = drop
                        best_dir = d
            if best_dir >= 0:
                flow_dir[y, x] = best_dir
    
    # Flow accumulation (topological order: high to low)
    # Flatten and sort by elevation descending
    land_indices = np.argwhere((data > 0) & ~np.isnan(data))
    sorted_idx = land_indices[np.argsort(-data[land_indices[:, 0], land_indices[:, 1]])]
    
    accumulation = np.ones((h, w), dtype=np.float32)
    for idx in sorted_idx:
        y, x = idx
        d = flow_dir[y, x]
        if d >= 0:
            ny, nx = y + dy[d], x + dx[d]
            accumulation[ny, nx] += accumulation[y, x]
    
    # Drainage density: fraction of cells that are "river" cells
    # A cell is a river if its accumulation exceeds threshold
    river_threshold = (1000.0 / resolution_m)**2  # ~1km² catchment
    river_cells = (accumulation > river_threshold) & (data > 0) & ~np.isnan(data)
    land_cells = (data > 0) & ~np.isnan(data)
    
    total_river_length = np.sum(river_cells) * resolution_m / 1000.0  # km
    total_land_area = np.sum(land_cells) * resolution_m**2 / 1e6  # km²
    
    return total_river_length, total_land_area


def variogram_sample(data, max_lag=50, sample_rate=10):
    """Compute variogram at various lags to estimate fractal dimension.
    gamma(h) = 0.5 * E[(Z(x+h) - Z(x))^2]
    For fractal terrain: gamma(h) ∝ h^(2H) where H = 3 - D
    """
    h, w = data.shape
    lags = np.arange(1, max_lag + 1)
    gamma = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # Subsample for speed
        diffs = []
        for y in range(0, h - lag, sample_rate):
            for x in range(0, w - lag, sample_rate):
                if not np.isnan(data[y, x]) and not np.isnan(data[y, x + lag]):
                    diffs.append((data[y, x] - data[y, x + lag])**2)
                if not np.isnan(data[y, x]) and not np.isnan(data[y + lag, x]):
                    diffs.append((data[y, x] - data[y + lag, x])**2)
        if diffs:
            gamma[i] = 0.5 * np.mean(diffs)
    
    # Fit power law: gamma(h) = a * h^(2H)
    valid = gamma > 0
    if np.sum(valid) < 5:
        return lags, gamma, 2.3, 0.0
    
    log_h = np.log(lags[valid])
    log_g = np.log(gamma[valid])
    slope, intercept = np.polyfit(log_h, log_g, 1)
    H = slope / 2.0  # Hurst exponent
    D = 3.0 - H       # Fractal dimension
    r2 = np.corrcoef(log_h, log_g)[0, 1]**2
    
    return lags, gamma, D, r2


def landmass_statistics(data, resolution_m):
    """Compute landmass count, sizes, and shape metrics using CCL."""
    from scipy import ndimage
    
    land = (data > 0) & ~np.isnan(data)
    
    # Connected component labeling
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_components = ndimage.label(land, structure=structure)
    
    if n_components == 0:
        return {'count': 0, 'sizes': [], 'largest_fraction': 0}
    
    sizes = []
    for i in range(1, n_components + 1):
        size_px = np.sum(labeled == i)
        sizes.append(size_px * resolution_m**2 / 1e6)  # km²
    
    sizes.sort(reverse=True)
    total_land = np.sum(land) * resolution_m**2 / 1e6
    
    return {
        'count': n_components,
        'sizes_km2': sizes[:20],  # top 20
        'largest_fraction': sizes[0] / total_land if sizes else 0,
        'total_land_km2': total_land
    }


# ══════════════════════════════════════════════════════════════
#  Main analysis
# ══════════════════════════════════════════════════════════════

def analyze_dem(filepath, output_json='terrain_fingerprint.json'):
    print("=" * 60)
    print(" DEM Statistical Fingerprint Analyzer")
    print("=" * 60)
    
    with rasterio.open(filepath) as src:
        print(f"\n[1/9] Reading DEM ({src.width}x{src.height})...")
        # For large DEMs, read at reduced resolution if needed
        if src.width * src.height > 50_000_000:
            # Subsample to ~2000x2000 max for analysis speed
            scale = max(src.width, src.height) / 2000.0
            scale = max(1, int(scale))
            print(f"  Subsample factor: {scale}x (for analysis speed)")
            data = src.read(1, out_shape=(src.height // scale, src.width // scale))
            data = data.astype(np.float32)
            resolution_m = abs(src.transform[0]) * scale * 111320.0 * math.cos(math.radians(39.0))
        else:
            data = src.read(1).astype(np.float32)
            resolution_m = abs(src.transform[0]) * 111320.0 * math.cos(math.radians(39.0))
        
        bounds = src.bounds
        crs = str(src.crs)
    
    h, w = data.shape
    land_mask = (data > 0) & ~np.isnan(data)
    ocean_mask = (data <= 0) & ~np.isnan(data)
    
    result = {}
    
    # ── 1. Basic statistics ────────────────────────────────────
    print("[2/9] Computing basic statistics...")
    valid = data[~np.isnan(data)]
    result['basic'] = {
        'width_px': w, 'height_px': h,
        'resolution_m': round(resolution_m, 1),
        'min_elevation': float(np.nanmin(data)),
        'max_elevation': float(np.nanmax(data)),
        'mean_elevation': float(np.nanmean(data)),
        'std_elevation': float(np.nanstd(data)),
        'median_elevation': float(np.nanmedian(data)),
        'land_fraction': float(np.mean(land_mask)),
        'ocean_fraction': float(np.mean(ocean_mask)),
    }
    print(f"  Range: {result['basic']['min_elevation']:.0f}m to {result['basic']['max_elevation']:.0f}m")
    print(f"  Mean: {result['basic']['mean_elevation']:.0f}m, Std: {result['basic']['std_elevation']:.0f}m")
    print(f"  Land: {result['basic']['land_fraction']*100:.1f}%, Ocean: {result['basic']['ocean_fraction']*100:.1f}%")
    
    # ── 2. Hypsometric curve ──────────────────────────────────
    print("[3/9] Computing hypsometric curve...")
    heights, fractions = hypsometric_curve(data, 100)
    result['hypsometric'] = {
        'heights': heights.tolist(),
        'cumulative_fraction': fractions.tolist(),
        'percentiles': {
            'p5': float(np.percentile(valid, 5)),
            'p10': float(np.percentile(valid, 10)),
            'p25': float(np.percentile(valid, 25)),
            'p50': float(np.percentile(valid, 50)),
            'p75': float(np.percentile(valid, 75)),
            'p90': float(np.percentile(valid, 90)),
            'p95': float(np.percentile(valid, 95)),
            'p99': float(np.percentile(valid, 99)),
        }
    }
    print(f"  P5={result['hypsometric']['percentiles']['p5']:.0f}m, P50={result['hypsometric']['percentiles']['p50']:.0f}m, P95={result['hypsometric']['percentiles']['p95']:.0f}m")
    
    # ── 3. Land elevation stats (above sea level only) ────────
    print("[4/9] Computing land elevation statistics...")
    land_vals = data[land_mask]
    result['land_elevation'] = {
        'min': float(np.min(land_vals)),
        'max': float(np.max(land_vals)),
        'mean': float(np.mean(land_vals)),
        'std': float(np.std(land_vals)),
        'median': float(np.median(land_vals)),
        'percentiles': {
            'p25': float(np.percentile(land_vals, 25)),
            'p50': float(np.percentile(land_vals, 50)),
            'p75': float(np.percentile(land_vals, 75)),
            'p90': float(np.percentile(land_vals, 90)),
            'p95': float(np.percentile(land_vals, 95)),
            'p99': float(np.percentile(land_vals, 99)),
        }
    }
    print(f"  Land mean: {result['land_elevation']['mean']:.0f}m, max: {result['land_elevation']['max']:.0f}m")
    
    # ── 4. Slope distribution ─────────────────────────────────
    print("[5/9] Computing slope distribution...")
    slope_rad = compute_slope(data, resolution_m)
    slope_edges, slope_counts = slope_histogram(slope_rad, land_mask, 50)
    result['slope'] = {
        'edges_deg': slope_edges.tolist(),
        'counts': slope_counts.tolist(),
        'mean_deg': float(np.mean(np.degrees(slope_rad[land_mask]))),
        'median_deg': float(np.median(np.degrees(slope_rad[land_mask]))),
        'std_deg': float(np.std(np.degrees(slope_rad[land_mask]))),
        'percentiles_deg': {
            'p50': float(np.percentile(np.degrees(slope_rad[land_mask]), 50)),
            'p75': float(np.percentile(np.degrees(slope_rad[land_mask]), 75)),
            'p90': float(np.percentile(np.degrees(slope_rad[land_mask]), 90)),
            'p95': float(np.percentile(np.degrees(slope_rad[land_mask]), 95)),
            'p99': float(np.percentile(np.degrees(slope_rad[land_mask]), 99)),
        }
    }
    print(f"  Mean slope: {result['slope']['mean_deg']:.1f}°, Median: {result['slope']['median_deg']:.1f}°")
    print(f"  P95 slope: {result['slope']['percentiles_deg']['p95']:.1f}°")
    
    # ── 5. Peak analysis ──────────────────────────────────────
    print("[6/9] Detecting peaks...")
    try:
        peaks = detect_peaks(data, min_prominence=100.0, neighborhood=5)
        peak_heights = sorted([float(p[2]) for p in peaks], reverse=True)
        peak_prominences = sorted([float(p[3]) for p in peaks], reverse=True)
        
        result['peaks'] = {
            'count': len(peaks),
            'top_heights': peak_heights[:50],
            'top_prominences': peak_prominences[:50],
            'height_percentiles': {
                'p50': float(np.percentile(peak_heights, 50)) if peak_heights else 0,
                'p75': float(np.percentile(peak_heights, 75)) if peak_heights else 0,
                'p90': float(np.percentile(peak_heights, 90)) if peak_heights else 0,
                'p95': float(np.percentile(peak_heights, 95)) if peak_heights else 0,
                'max': float(max(peak_heights)) if peak_heights else 0,
            }
        }
        print(f"  Peaks detected: {len(peaks)}")
        if peak_heights:
            print(f"  Top peaks: {peak_heights[:5]}")
    except ImportError:
        result['peaks'] = {'count': 0, 'note': 'scipy not available'}
        print("  (scipy not available — skipping peak detection)")
    
    # ── 6. Fractal dimension ──────────────────────────────────
    print("[7/9] Computing fractal dimension...")
    try:
        lags, gamma_vals, D, r2 = variogram_sample(data, max_lag=min(50, min(h, w)//10), sample_rate=10)
        result['fractal'] = {
            'variogram_lags': lags.tolist(),
            'variogram_gamma': gamma_vals.tolist(),
            'fractal_dimension_D': round(D, 3),
            'hurst_exponent_H': round(3.0 - D, 3),
            'fit_r2': round(r2, 4),
        }
        print(f"  Fractal dimension D = {D:.3f} (H = {3-D:.3f}), R² = {r2:.4f}")
    except Exception as e:
        result['fractal'] = {'error': str(e)}
        print(f"  Error: {e}")
    
    # ── 7. Landmass statistics ────────────────────────────────
    print("[8/9] Computing landmass statistics...")
    try:
        lm_stats = landmass_statistics(data, resolution_m)
        result['landmasses'] = lm_stats
        print(f"  Landmasses: {lm_stats['count']}")
        print(f"  Largest: {lm_stats['sizes_km2'][0]:.0f} km² ({lm_stats['largest_fraction']*100:.1f}% of land)")
    except ImportError:
        result['landmasses'] = {'count': 0, 'note': 'scipy not available'}
        print("  (scipy not available)")
    
    # ── 8. Drainage density ───────────────────────────────────
    print("[9/9] Computing drainage density...")
    try:
        # Use subset for speed if too large
        if h * w > 5_000_000:
            sub = data[::2, ::2]  # half resolution
            sub_res = resolution_m * 2
        else:
            sub = data
            sub_res = resolution_m
        
        river_km, land_km2 = compute_drainage_density(sub, sub_res)
        result['drainage'] = {
            'total_river_km': round(river_km, 1),
            'total_land_km2': round(land_km2, 1),
            'density_km_per_km2': round(river_km / land_km2, 6) if land_km2 > 0 else 0,
        }
        print(f"  River network: {river_km:.0f} km over {land_km2:.0f} km²")
        print(f"  Drainage density: {river_km/land_km2*1000:.1f} m/km²")
    except Exception as e:
        result['drainage'] = {'error': str(e)}
        print(f"  Error: {e}")
    
    # ── Save JSON ─────────────────────────────────────────────
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nFingerprint saved to: {output_json}")
    
    # ── Print calibration recommendations ─────────────────────
    print("\n" + "=" * 60)
    print(" CALIBRATION RECOMMENDATIONS")
    print("=" * 60)
    
    land_frac = result['basic']['land_fraction']
    elev_range = result['basic']['max_elevation'] - result['basic']['min_elevation']
    land_mean = result['land_elevation']['mean']
    land_std = result['land_elevation']['std']
    
    print(f"\n  Sea level target:     {1.0 - land_frac:.3f} (ocean fraction = {1.0-land_frac:.2%})")
    print(f"  Mountain max:          ~{result['land_elevation']['percentiles']['p99']:.0f}m")
    print(f"  Trench depth:          ~{abs(result['basic']['min_elevation']):.0f}m")
    print(f"  Continental base:      ~{land_mean:.0f}m (mean land elevation)")
    print(f"  Land std dev:          {land_std:.0f}m")
    print(f"  Mean slope:            {result['slope']['mean_deg']:.1f}°")
    print(f"  Fractal dimension:     {result.get('fractal', {}).get('fractal_dimension_D', 'N/A')}")
    print(f"  Max/World ratio:       {result['land_elevation']['max']/111320.0:.6f} (height/horizontal)")
    
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        dem_path = r'c:\VulkanProject\resources\dem.tif'
    else:
        dem_path = sys.argv[1]
    
    analyze_dem(dem_path, r'c:\VulkanProject\terrain_fingerprint.json')
