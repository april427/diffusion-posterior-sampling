#!/usr/bin/env python3
"""
Visualize a single training-distribution sample to verify the dataset is correct.
Generates one sample using the same parameters and building-placement distribution
as the training config (configs/aoa_amp_building_config.yaml), then plots AoA,
amplitude, and LoS maps via RayTracingAoAMapGPU.
"""

import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aoa_amp_building_gpu import RayTracingAoAMapGPU


# ---------------------------------------------------------------------------
# Training-config defaults (must match configs/aoa_amp_building_config.yaml)
# ---------------------------------------------------------------------------
MAP_SIZE = (128, 128)
GRID_SPACING = 1.0
BS_GRID_SPACING = 4.0
BUILDING_SIZE_RANGE = ((15, 35), (10, 25))  # (min_w, max_w), (min_h, max_h)
MIN_BUILDING_DISTANCE = 5.0
MARGIN = 5  # keep buildings away from map edges


def _random_building_config(num_buildings: int) -> list:
    """Sample a random building layout following the training distribution."""
    buildings = []
    max_attempts = 1000
    for _ in range(num_buildings):
        for _ in range(max_attempts):
            w = random.randint(*BUILDING_SIZE_RANGE[0])
            h = random.randint(*BUILDING_SIZE_RANGE[1])
            x = random.uniform(MARGIN, MAP_SIZE[0] - w - MARGIN)
            y = random.uniform(MARGIN, MAP_SIZE[1] - h - MARGIN)
            candidate = {'x': x, 'y': y, 'width': w, 'height': h}
            # Check no overlap (with min-distance buffer) against placed buildings
            if all(_no_overlap(candidate, b) for b in buildings):
                buildings.append(candidate)
                break
    return buildings


def _no_overlap(b1: dict, b2: dict) -> bool:
    d = MIN_BUILDING_DISTANCE / 2
    return (b1['x'] + b1['width'] + d <= b2['x'] - d
            or b2['x'] + b2['width'] + d <= b1['x'] - d
            or b1['y'] + b1['height'] + d <= b2['y'] - d
            or b2['y'] + b2['height'] + d <= b1['y'] - d)


def plot_training_sample():
    """Generate and plot one sample from the training distribution."""

    # Sample number of buildings: 1, 2, or 3 with equal probability
    num_buildings = random.choice([1, 2, 3])
    building_config = _random_building_config(num_buildings)

    print(f"Sampled {num_buildings} building(s):")
    for i, b in enumerate(building_config):
        print(f"  Building {i+1}: pos=({b['x']:.1f}, {b['y']:.1f}), "
              f"size=({b['width']}x{b['height']})")

    # Pick one random BS position from the training grid
    bs_x_positions = np.arange(MARGIN, MAP_SIZE[0] - MARGIN, BS_GRID_SPACING)
    bs_y_positions = np.arange(MARGIN, MAP_SIZE[1] - MARGIN, BS_GRID_SPACING)
    bs_x = float(random.choice(bs_x_positions))
    bs_y = float(random.choice(bs_y_positions))

    print(f"BS position: ({bs_x:.1f}, {bs_y:.1f})")

    # Generate a single sample using RayTracingAoAMapGPU directly
    rt = RayTracingAoAMapGPU(
        map_size=MAP_SIZE,
        grid_spacing=GRID_SPACING,
        device='auto',
        verbose=True,
    )
    rt.set_base_station(bs_x, bs_y)
    for b in building_config:
        rt.add_building(b['x'], b['y'], b['width'], b['height'])

    aoa_maps, los_map = rt.generate_aoa_map_gpu(num_paths=3)
    amplitude_maps = rt.generate_amplitude_map_gpu(num_paths=3)

    print(f"\nSample info:")
    print(f"  AoA maps:        {[m.shape for m in aoa_maps]}")
    print(f"  Amplitude maps:  {[m.shape for m in amplitude_maps]}")
    print(f"  LoS map:         {los_map.shape}, "
          f"LoS%={np.mean(los_map)*100:.1f}%")

    path_names = ['Strongest Path', '2nd Strongest', '3rd Strongest']
    rt.plot_aoa_map(aoa_maps, los_map, path_names)
    rt.plot_amplitude_map(amplitude_maps, path_names)
    print("Done.")


if __name__ == "__main__":
    plot_training_sample()