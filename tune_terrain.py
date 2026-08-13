#!/usr/bin/env python3
"""
Terrain Parameter Optimizer
Tunes the procedural tectonic terrain generator to match a reference DEM.

Approach:
1. Load reference DEM fingerprint (from dem_analyzer.py)
2. Define parameter search space
3. Run terrain generator with different parameter combinations
4. Score each run against reference statistics
5. Use iterative random search + local refinement to find best parameters

Usage: python tune_terrain.py [--evals N] [--ref fingerprint.json]
"""

import json
import subprocess
import sys
import os
import time
import random
import math
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════

GENERATOR_EXE = r'c:\VulkanProject\tune_terrain_gen.exe'
REF_FINGERPRINT = r'c:\VulkanProject\terrain_fingerprint.json'
NUM_EVALS = 40  # total parameter sets to try

# Parameter search space: [min, max]
PARAM_SPACE = {
    'numPlates':       (8, 16),
    'continentalFrac': (0.20, 0.50),
    'mountainMax':     (15.0, 40.0),
    'trenchMax':       (8.0, 25.0),
    'seaLevel':        (0.35, 0.60),
    'continentalBase': (0.5, 3.0),
    'iterations':      (60, 150),
    'coastalPlainW':   (100.0, 500.0),
}

# Integer parameters (rounded to int)
INT_PARAMS = {'numPlates', 'iterations'}

# Fixed seed for reproducibility of optimization
OPT_SEED = 12345


# ══════════════════════════════════════════════════════════════
#  Reference fingerprint loader
# ══════════════════════════════════════════════════════════════

def load_reference(filepath):
    """Load reference DEM fingerprint."""
    with open(filepath) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
#  Terrain generator runner
# ══════════════════════════════════════════════════════════════

def run_generator(params, seed=None):
    """Run the C++ terrain generator with given parameters. Returns metrics dict."""
    if seed is None:
        seed = random.randint(1, 2**31 - 1)
    
    cmd = [
        GENERATOR_EXE,
        str(seed),
        str(params['numPlates']),
        f"{params['continentalFrac']:.4f}",
        f"{params['mountainMax']:.2f}",
        f"{params['trenchMax']:.2f}",
        f"{params['seaLevel']:.3f}",
        f"{params['continentalBase']:.2f}",
        str(params['iterations']),
        f"{params['coastalPlainW']:.1f}",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  ERROR: generator crashed: {result.stderr[:200]}")
            return None
        return json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print("  ERROR: generator timed out")
        return None
    except json.JSONDecodeError as e:
        print(f"  ERROR: invalid JSON: {e}")
        print(f"  stdout: {result.stdout[:200]}")
        return None


# ══════════════════════════════════════════════════════════════
#  Scoring function
# ══════════════════════════════════════════════════════════════

def compute_score(gen_metrics, ref):
    """
    Compute a score (0 = perfect match, higher = worse).
    Compares statistical distributions between generated terrain and reference DEM.
    """
    score = 0.0
    
    # 1. Land fraction match (weight: 0.20)
    gen_land = gen_metrics.get('landFrac', 0.5)
    ref_land = ref['basic']['land_fraction']
    land_diff = abs(gen_land - ref_land)
    score += 0.20 * min(land_diff / 0.3, 1.0)  # normalize: 0.3 diff = max penalty
    
    # 2. Elevation range match (weight: 0.15)
    gen_range = gen_metrics.get('elevMax', 0) - gen_metrics.get('elevMin', 0)
    ref_range = ref['basic']['max_elevation'] - ref['basic']['min_elevation']
    # Scale reference range to our world size
    ref_range_scaled = ref_range * (4096.0 / 1700000.0)  # Turkey DEM is ~1700km
    range_ratio = gen_range / max(ref_range_scaled, 1.0)
    range_score = abs(math.log2(max(range_ratio, 0.01)))
    score += 0.15 * min(range_score / 5.0, 1.0)
    
    # 3. Slope distribution match (weight: 0.20)
    gen_slope_mean = gen_metrics.get('slopeMean', 5.0)
    ref_slope_mean = ref['slope']['mean_deg']
    slope_diff = abs(gen_slope_mean - ref_slope_mean)
    score += 0.20 * min(slope_diff / 15.0, 1.0)
    
    # 4. Hypsometric curve match (weight: 0.15)
    # Compare the shape of the height distribution
    gen_hypso = gen_metrics.get('hypso', [])
    if gen_hypso:
        gen_norm = [(h - gen_metrics['elevMin']) / max(gen_range, 1.0) for h in gen_hypso]
        ref_hypsometric = ref.get('hypsometric', {})
        ref_heights = ref_hypsometric.get('heights', [])
        ref_fractions = ref_hypsometric.get('cumulative_fraction', [])
        
        if ref_heights and ref_fractions:
            ref_range_h = ref_heights[-1] - ref_heights[0]
            ref_norm = [(h - ref_heights[0]) / max(ref_range_h, 1.0) for h in ref_heights]
            
            # Interpolate to same number of points and compute L2 distance
            n_pts = min(len(gen_norm), len(ref_norm))
            gen_sample = np.interp(np.linspace(0, 1, n_pts), np.linspace(0, 1, len(gen_norm)), gen_norm)
            ref_sample = np.interp(np.linspace(0, 1, n_pts), np.linspace(0, 1, len(ref_norm)), ref_norm)
            hypso_dist = np.mean((gen_sample - ref_sample)**2)
            score += 0.15 * min(hypso_dist * 10.0, 1.0)
    
    # 5. Drainage / river density (weight: 0.10)
    gen_rivers = gen_metrics.get('riverCount', 0)
    gen_area = 4096.0 * 4096.0 * gen_land / 1e6  # km²
    gen_density = gen_rivers / max(gen_area, 0.01)
    
    ref_river_km = ref.get('drainage', {}).get('total_river_km', 100000)
    ref_area_km2 = ref.get('drainage', {}).get('total_land_km2', 500000)
    ref_density = ref_river_km / max(ref_area_km2, 1.0)
    
    density_ratio = gen_density / max(ref_density, 0.001)
    score += 0.10 * min(abs(math.log2(max(density_ratio, 0.01))) / 5.0, 1.0)
    
    # 6. Peak density (weight: 0.10)
    gen_peaks = gen_metrics.get('peakCount', 0)
    gen_peak_density = gen_peaks / max(gen_area, 0.01)
    
    ref_peaks = ref.get('peaks', {}).get('count', 1000)
    ref_peak_density = ref_peaks / max(ref_area_km2, 1.0)
    
    peak_ratio = gen_peak_density / max(ref_peak_density, 0.001)
    score += 0.10 * min(abs(math.log2(max(peak_ratio, 0.01))) / 5.0, 1.0)
    
    # 7. Landmass count bonus (weight: 0.10)
    # Reward having multiple landmasses (not just one blob)
    gen_landmasses = gen_metrics.get('landmassCount', 1)
    landmass_score = 0.0
    if gen_landmasses < 3:
        landmass_score = 1.0  # penalize single-landmass worlds
    elif gen_landmasses < 10:
        landmass_score = 0.5
    elif gen_landmasses < 30:
        landmass_score = 0.1
    score += 0.10 * landmass_score
    
    return score


# ══════════════════════════════════════════════════════════════
#  Parameter sampling
# ══════════════════════════════════════════════════════════════

def random_params(rng):
    """Generate a random parameter set within the search space."""
    p = {}
    for key, (lo, hi) in PARAM_SPACE.items():
        if key in INT_PARAMS:
            p[key] = rng.randint(int(lo), int(hi))
        else:
            p[key] = lo + rng.random() * (hi - lo)
    return p


def perturb_params(base, rng, scale=0.15):
    """Perturb a parameter set by a small random amount."""
    p = {}
    for key, (lo, hi) in PARAM_SPACE.items():
        val = base[key]
        delta = rng.uniform(-scale, scale) * (hi - lo)
        val += delta
        val = max(lo, min(hi, val))
        if key in INT_PARAMS:
            val = int(round(val))
        p[key] = val
    return p


# ══════════════════════════════════════════════════════════════
#  Main optimization loop
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Terrain Parameter Optimizer")
    print("=" * 60)
    
    # Check generator exists
    if not os.path.exists(GENERATOR_EXE):
        print(f"ERROR: Generator not found at {GENERATOR_EXE}")
        print("Build it with: g++ -std=c++20 -I . tune_terrain_gen.cpp -o tune_terrain_gen.exe")
        sys.exit(1)
    
    # Load reference
    print(f"\n[1] Loading reference fingerprint: {REF_FINGERPRINT}")
    ref = load_reference(REF_FINGERPRINT)
    print(f"    Reference: {ref['basic']['min_elevation']:.0f}m to {ref['basic']['max_elevation']:.0f}m")
    print(f"    Land fraction: {ref['basic']['land_fraction']:.1%}")
    print(f"    Mean slope: {ref['slope']['mean_deg']:.1f}°")
    print(f"    Drainage density: {ref['drainage']['density_km_per_km2']:.3f} km/km²")
    
    # Setup random
    rng = random.Random(OPT_SEED)
    
    results = []
    
    # Phase 1: Random exploration (30 evals)
    print(f"\n[2] Phase 1: Random exploration ({NUM_EVALS // 2} evaluations)")
    phase1_evals = NUM_EVALS // 2
    
    for i in range(phase1_evals):
        params = random_params(rng)
        seed = rng.randint(1, 2**31 - 1)
        
        print(f"\n  Eval {i+1}/{phase1_evals}: plates={params['numPlates']} "
              f"mtn={params['mountainMax']:.0f} trn={params['trenchMax']:.0f} "
              f"sea={params['seaLevel']:.2f} base={params['continentalBase']:.1f} "
              f"iter={params['iterations']} seed={seed}")
        
        t0 = time.time()
        gen = run_generator(params, seed)
        elapsed = time.time() - t0
        
        if gen is None:
            print(f"    FAILED ({elapsed:.1f}s)")
            continue
        
        score = compute_score(gen, ref)
        print(f"    Score={score:.4f}  land={gen['landFrac']:.1%}  "
              f"range=[{gen['elevMin']:.1f}, {gen['elevMax']:.1f}]  "
              f"slope={gen['slopeMean']:.1f}°  peaks={gen['peakCount']}  "
              f"rivers={gen['riverCount']}  ({elapsed:.1f}s)")
        
        results.append({
            'params': params,
            'seed': seed,
            'score': score,
            'metrics': gen,
        })
    
    # Sort and show best so far
    results.sort(key=lambda r: r['score'])
    
    print(f"\n[3] Best from exploration: score={results[0]['score']:.4f}")
    print(f"    params={results[0]['params']}")
    
    # Phase 2: Local refinement around top performers
    print(f"\n[4] Phase 2: Local refinement ({NUM_EVALS - phase1_evals} evaluations)")
    top_k = min(5, len(results))
    parents = [r['params'] for r in results[:top_k]]
    
    phase2_evals = NUM_EVALS - phase1_evals
    for i in range(phase2_evals):
        parent = parents[i % len(parents)]
        # Decreasing perturbation scale
        scale = 0.15 * (1.0 - i / phase2_evals)
        params = perturb_params(parent, rng, scale)
        seed = rng.randint(1, 2**31 - 1)
        
        print(f"\n  Eval {phase1_evals+i+1}/{NUM_EVALS}: plates={params['numPlates']} "
              f"mtn={params['mountainMax']:.0f} trn={params['trenchMax']:.0f} "
              f"sea={params['seaLevel']:.2f} base={params['continentalBase']:.1f} "
              f"iter={params['iterations']} seed={seed}")
        
        t0 = time.time()
        gen = run_generator(params, seed)
        elapsed = time.time() - t0
        
        if gen is None:
            continue
        
        score = compute_score(gen, ref)
        print(f"    Score={score:.4f}  land={gen['landFrac']:.1%}  "
              f"range=[{gen['elevMin']:.1f}, {gen['elevMax']:.1f}]  "
              f"slope={gen['slopeMean']:.1f}°  peaks={gen['peakCount']}  "
              f"rivers={gen['riverCount']}  ({elapsed:.1f}s)")
        
        results.append({
            'params': params,
            'seed': seed,
            'score': score,
            'metrics': gen,
        })
    
    # Final results
    results.sort(key=lambda r: r['score'])
    
    print("\n" + "=" * 60)
    print(" OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nTop 5 parameter sets:")
    for i, r in enumerate(results[:5]):
        p = r['params']
        print(f"\n  #{i+1}  Score: {r['score']:.4f}  Seed: {r['seed']}")
        print(f"      numPlates={p['numPlates']} continentalFrac={p['continentalFrac']:.3f}")
        print(f"      mountainMax={p['mountainMax']:.1f} trenchMax={p['trenchMax']:.1f}")
        print(f"      seaLevel={p['seaLevel']:.3f} continentalBase={p['continentalBase']:.2f}")
        print(f"      iterations={p['iterations']} coastalPlainW={p['coastalPlainW']:.0f}")
    
    # Save results
    best_config = {
        'numPlates': results[0]['params']['numPlates'],
        'continentalFrac': results[0]['params']['continentalFrac'],
        'mountainMax': results[0]['params']['mountainMax'],
        'trenchMax': results[0]['params']['trenchMax'],
        'seaLevel': results[0]['params']['seaLevel'],
        'continentalBase': results[0]['params']['continentalBase'],
        'iterations': results[0]['params']['iterations'],
        'coastalPlainW': results[0]['params']['coastalPlainW'],
        'best_score': results[0]['score'],
        'best_seed': results[0]['seed'],
    }
    
    output_path = r'c:\VulkanProject\best_terrain_params.json'
    with open(output_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nBest config saved to: {output_path}")
    
    # Print copy-paste ready code for VulkanEngine.cpp
    print("\n" + "-" * 60)
    print(" Copy-paste into VulkanEngine.cpp createMasterTectonicMap():")
    print("-" * 60)
    print(f'    tcfg.numPlates       = {best_config["numPlates"]};')
    print(f'    tcfg.continentalFrac = {best_config["continentalFrac"]:.4f}f;')
    print(f'    tcfg.mountainMax     = {best_config["mountainMax"]:.1f}f;')
    print(f'    tcfg.trenchMax       = {best_config["trenchMax"]:.1f}f;')
    print(f'    tcfg.seaLevel        = {best_config["seaLevel"]:.3f}f;')
    print(f'    tcfg.continentalBase = {best_config["continentalBase"]:.2f}f;')
    print(f'    tcfg.iterations      = {best_config["iterations"]};')
    print(f'    tcfg.coastalPlainW   = {best_config["coastalPlainW"]:.0f}f;')


if __name__ == '__main__':
    main()
