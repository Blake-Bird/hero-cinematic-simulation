#!/usr/bin/env python3
"""Generate optimized cosmological snapshot schedule for EDE vs LCDM campaign.

Creates ~100-120 epochs sampled evenly in scale factor a = 1/(1+z),
guaranteeing key high-z science epochs (including z = 14.44, 14.6, 14.5, 14.4, 14.3, 14.2).
"""
import argparse
import numpy as np

GUARANTEED_SCIENCE_Z = [
    99.0, 30.0, 25.0, 24.0, 22.0, 20.0, 18.0, 16.0, 15.0,
    14.6, 14.5, 14.44, 14.4, 14.3, 14.2, 14.0, 13.0, 12.0, 11.0, 10.0,
    9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0
]

def build_snapshot_schedule(tier='production_120'):
    """Generate list of ordered redshifts for simulation snapshots."""
    if tier == 'benchmark_steps':
        return [99.0, 95.0, 90.0]
    if tier == 'smoke_32':
        return [99.0, 14.44, 9.0, 3.0, 1.0, 0.5, 0.0]
    
    if tier == 'science_30':
        return sorted(GUARANTEED_SCIENCE_Z, reverse=True)
    
    # 120-snapshot tier: scale-factor spacing matching physical cosmic time
    intervals = [
        (99.0, 30.0, 10),
        (30.0, 20.0, 12),
        (20.0, 10.0, 35),
        (10.0, 5.0, 23),
        (5.0, 2.0, 16),
        (2.0, 0.0, 24)
    ]
    
    all_a = []
    for z_hi, z_lo, count in intervals:
        a_lo = 1.0 / (1.0 + z_hi)
        a_hi = 1.0 / (1.0 + z_lo)
        all_a.extend(np.linspace(a_lo, a_hi, count, endpoint=True).tolist())
    
    for z in GUARANTEED_SCIENCE_Z:
        all_a.append(1.0 / (1.0 + z))
    
    all_a = sorted(set(all_a))
    redshifts = [round(1.0 / a - 1.0, 4) for a in all_a]
    final_z = sorted(set(redshifts + GUARANTEED_SCIENCE_Z), reverse=True)
    
    cleaned = []
    for z in final_z:
        if not cleaned:
            cleaned.append(z)
            continue
        prev = cleaned[-1]
        diff = abs(prev - z)
        # Keep high-z resolution near 14.44 dense
        if 14.0 <= z <= 15.0:
            if diff >= 0.035:
                cleaned.append(z)
        elif diff >= 0.05:
            cleaned.append(z)
            
    # Guarantee 0.0 is the exact end
    if cleaned[-1] != 0.0:
        cleaned.append(0.0)
    return cleaned

def format_schedule_string(redshifts):
    return ','.join(f'{z:.4g}' if not float(z).is_integer() else f'{int(z)}' for z in redshifts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=('smoke_32', 'science_30', 'production_120'), default='production_120')
    parser.add_argument('--csv', action='store_true', help='Output comma-separated string')
    args = parser.parse_args()
    
    schedule = build_snapshot_schedule(args.tier)
    if args.csv:
        print(format_schedule_string(schedule))
    else:
        print(f"Tier: {args.tier} ({len(schedule)} snapshots)")
        print(format_schedule_string(schedule))

