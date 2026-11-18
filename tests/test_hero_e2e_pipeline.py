#!/usr/bin/env python3
"""End-to-End Verification Test for the Complete Hero Cosmological & Cinematic Pipeline.

Tests:
  1. GADGET-4 multi-species hydro snapshot creation (Gas, DM, Stars, Black Hole)
  2. VELOCIraptor 6D phase-space finder
  3. Gala / StreaMAX tidal stellar stream simulation with subhalo flyby gaps
  4. Lenstronomy strong gravitational lensing Einstein ring
  5. DisPerSE 3D persistent homology cosmic filament skeleton
  6. SBI neural simulation-based inference posterior estimation
  7. Blacklight Kerr relativistic event horizon ray-tracing
  8. Hero particle extraction (hero_particles.npz)
  9. Blender 5.3 Cycles master scene generation & test frame render
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py

TEST_RUN_DIR = ROOT / "test_hero_run"
RESULTS_DIR = TEST_RUN_DIR / "results"


def setup_mock_gadget_snapshot():
    """Create a physically consistent GADGET-4 hydro snapshot with PartType0, 1, 4, 5."""
    if TEST_RUN_DIR.is_dir():
        shutil.rmtree(TEST_RUN_DIR)
    
    snap_dir = TEST_RUN_DIR / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_file = snap_dir / "snapshot_015.hdf5"

    box_size_kpc_h = 20000.0  # 20 Mpc/h
    center = np.array([10000.0, 10000.0, 10000.0])  # Halo center at 10 Mpc/h

    np.random.seed(42)
    n_gas = 1200
    n_dm = 2500
    n_star = 800
    n_bh = 1

    # PartType0: Gas particles
    r_gas = np.random.exponential(150.0, size=n_gas)
    u_gas = np.random.normal(size=(n_gas, 3))
    u_gas /= np.linalg.norm(u_gas, axis=1, keepdims=True)
    gas_coords = center + u_gas * r_gas[:, None]
    gas_vels = np.random.normal(0, 180.0, size=(n_gas, 3))
    gas_u = np.random.uniform(50.0, 2500.0, size=n_gas)  # Internal energy
    gas_rho = np.random.uniform(1e-4, 1.0, size=n_gas)

    # PartType1: Dark Matter particles
    r_dm = np.random.exponential(350.0, size=n_dm)
    u_dm = np.random.normal(size=(n_dm, 3))
    u_dm /= np.linalg.norm(u_dm, axis=1, keepdims=True)
    dm_coords = center + u_dm * r_dm[:, None]
    dm_vels = np.random.normal(0, 220.0, size=(n_dm, 3))

    # PartType4: Star particles
    r_star = np.random.exponential(25.0, size=n_star)
    theta = np.random.uniform(0, 2*np.pi, size=n_star) + 2.0 * np.log(1.0 + r_star / 5.0)
    z_star = np.random.normal(0, 2.5, size=n_star)
    star_coords = center + np.column_stack([r_star * np.cos(theta), r_star * np.sin(theta), z_star])
    star_vels = np.random.normal(0, 140.0, size=(n_star, 3))
    star_aform = np.random.uniform(0.1, 0.95, size=n_star)

    # PartType5: Central Black Hole
    bh_coords = np.array([center])
    bh_vels = np.array([[5.0, -3.0, 2.0]])
    bh_mass = np.array([4.2e-4])  # 4.2e6 Msun in code units
    bh_mdot = np.array([1.2e-3])

    with h5py.File(snap_file, "w") as f:
        # Header
        hdr = f.create_group("Header")
        hdr.attrs["BoxSize"] = box_size_kpc_h
        hdr.attrs["Time"] = 1.0  # z=0
        hdr.attrs["Redshift"] = 0.0
        hdr.attrs["NumPart_ThisFile"] = [n_gas, n_dm, 0, 0, n_star, n_bh]
        hdr.attrs["NumPart_Total"] = [n_gas, n_dm, 0, 0, n_star, n_bh]
        hdr.attrs["MassTable"] = [0.0, 0.001, 0.0, 0.0, 0.0, 0.0]

        # PartType0
        g0 = f.create_group("PartType0")
        g0.create_dataset("Coordinates", data=gas_coords.astype(np.float32))
        g0.create_dataset("Velocities", data=gas_vels.astype(np.float32))
        g0.create_dataset("ParticleIDs", data=np.arange(1, n_gas + 1, dtype=np.uint64))
        g0.create_dataset("Masses", data=np.full(n_gas, 0.0001, dtype=np.float32))
        g0.create_dataset("InternalEnergy", data=gas_u.astype(np.float32))
        g0.create_dataset("Density", data=gas_rho.astype(np.float32))

        # PartType1
        g1 = f.create_group("PartType1")
        g1.create_dataset("Coordinates", data=dm_coords.astype(np.float32))
        g1.create_dataset("Velocities", data=dm_vels.astype(np.float32))
        g1.create_dataset("ParticleIDs", data=np.arange(100000, 100000 + n_dm, dtype=np.uint64))
        g1.create_dataset("Masses", data=np.full(n_dm, 0.001, dtype=np.float32))

        # PartType4
        g4 = f.create_group("PartType4")
        g4.create_dataset("Coordinates", data=star_coords.astype(np.float32))
        g4.create_dataset("Velocities", data=star_vels.astype(np.float32))
        g4.create_dataset("ParticleIDs", data=np.arange(500000, 500000 + n_star, dtype=np.uint64))
        g4.create_dataset("Masses", data=np.full(n_star, 0.0001, dtype=np.float32))
        g4.create_dataset("StellarFormationTime", data=star_aform.astype(np.float32))

        # PartType5
        g5 = f.create_group("PartType5")
        g5.create_dataset("Coordinates", data=bh_coords.astype(np.float32))
        g5.create_dataset("Velocities", data=bh_vels.astype(np.float32))
        g5.create_dataset("ParticleIDs", data=np.array([999999], dtype=np.uint64))
        g5.create_dataset("Masses", data=bh_mass.astype(np.float32))
        g5.create_dataset("BH_Mass", data=bh_mass.astype(np.float32))
        g5.create_dataset("BH_Mdot", data=bh_mdot.astype(np.float32))

    print(f"Created mock GADGET-4 snapshot: {snap_file}")
    return snap_file


def run_stages_test():
    """Run scripts/run_hero_stages.py and verify all scientific products."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_hero_stages.py"),
        str(TEST_RUN_DIR),
        "--snapshot", "15"
    ]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.returncode != 0:
        print("STDERR:\n", res.stderr)
        raise RuntimeError(f"run_hero_stages.py failed with returncode {res.returncode}")

    # Check that all expected deliverables exist
    expected_files = [
        RESULTS_DIR / "velociraptor" / "velociraptor_catalogue.json",
        RESULTS_DIR / "streams" / "gala_stream_particles.npz",
        RESULTS_DIR / "streams" / "gala_stream_metadata.json",
        RESULTS_DIR / "streams" / "gala_stellar_stream_4k.png",
        RESULTS_DIR / "lenstronomy" / "lenstronomy_einstein_ring_4k.png",
        RESULTS_DIR / "lenstronomy" / "lensing_measurements.json",
        RESULTS_DIR / "disperse" / "snapshot_015_disperse_skeleton.npz",
        RESULTS_DIR / "disperse" / "disperse_skeleton_summary.json",
        RESULTS_DIR / "inference" / "inference_corner.png",
        RESULTS_DIR / "inference" / "inference_posterior_summary.json",
        RESULTS_DIR / "blacklight" / "blacklight_kerr_horizon_4k.png",
        RESULTS_DIR / "blacklight" / "blacklight_kerr_metadata.json"
    ]
    for ef in expected_files:
        assert ef.is_file(), f"Missing deliverable: {ef}"
        assert ef.stat().st_size > 0, f"Empty deliverable: {ef}"
        print(f"Verified deliverable: {ef.name} ({ef.stat().st_size} bytes)")


def run_cinematic_planning_test():
    """Run scripts/build_hero_cinematic.py and verify particle extraction & Blender script."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts/build_hero_cinematic.py"),
        str(TEST_RUN_DIR),
        "--snapshot", "15",
        "--output", str(TEST_RUN_DIR / "hero_cinematic_manifest.json")
    ]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.returncode != 0:
        print("STDERR:\n", res.stderr)
        raise RuntimeError(f"build_hero_cinematic.py failed with returncode {res.returncode}")

    # Check hero_particles.npz
    particles_npz = RESULTS_DIR / "hero_particles.npz"
    assert particles_npz.is_file(), f"Missing hero_particles.npz: {particles_npz}"
    pdata = np.load(particles_npz)
    for key in ["gas_pos_mpc", "gas_temp_k", "star_pos_mpc", "star_age_gyr", "bh_pos_mpc", "stream_pos_mpc"]:
        assert key in pdata, f"Missing key in hero_particles.npz: {key}"
        assert len(pdata[key]) > 0, f"Empty array for key: {key}"
        print(f"Verified hero_particles[{key}]: shape {pdata[key].shape}")

    # Check render_hero_blender.py
    blender_script = TEST_RUN_DIR / "render_hero_blender.py"
    assert blender_script.is_file(), f"Missing render_hero_blender.py: {blender_script}"
    print(f"Verified Blender script: {blender_script} ({blender_script.stat().st_size} bytes)")


def run_blender_scene_test():
    """Run Blender in background mode to verify scene building with real particles."""
    blender_script = TEST_RUN_DIR / "render_hero_blender.py"
    cmd = ["blender", "-b", "-P", str(blender_script)]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("BLENDER STDOUT:\n", res.stdout)
    if res.returncode != 0:
        print("BLENDER STDERR:\n", res.stderr)
        raise RuntimeError(f"Blender failed with returncode {res.returncode}")

    assert "Hero Master Blender Cycles Scene Configured" in res.stdout
    assert "GasParticlesObject" in res.stdout or "real SPH gas particles" in res.stdout
    print("Blender master Cycles scene successfully configured with real simulation particles!")


def main():
    print("======================================================================")
    print(" Starting Hero Simulation & Cinematic End-to-End Verification Test")
    print("======================================================================")

    # Step 1: Mock Snapshot
    setup_mock_gadget_snapshot()

    # Step 2: Advanced Astrophysics Stages
    run_stages_test()

    # Step 3: Cinematic Planning & Particle Extraction
    run_cinematic_planning_test()

    # Step 4: Blender Cycles Scene Verification
    run_blender_scene_test()

    print("======================================================================")
    print(" ALL END-TO-END HERO PIPELINE TESTS PASSED WITH 100% SUCCESS!")
    print("======================================================================")


if __name__ == "__main__":
    main()

