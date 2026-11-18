# High Point University (HPU) Slurm Cluster Guide: EDE vs. $\Lambda\text{CDM}$

This guide explains how to transfer, set up, benchmark, and run the verified matched-pair cosmological simulations on the HPU Slurm cluster (`WSNS130-L26T-AS` / `hpc.highpoint.edu`).

---

## 1. Transfer Fixed Package to Cluster

From your local machine (Mac / workstation terminal):
```bash
scp /Users/bbird/Downloads/ede-matched-cluster-fixed.zip bbird@hpc.highpoint.edu:~/
```

Then SSH into the cluster login node:
```bash
ssh bbird@hpc.highpoint.edu
```
*(If prompted, change your initial temporary password using the `passwd` command).*

---

## 2. Extract and Set Up Environment

Unpack the archive:
```bash
mkdir -p ~/ede-matched-cluster
unzip -o ~/ede-matched-cluster-fixed.zip -d ~/ede-matched-cluster
cd ~/ede-matched-cluster
```

Submit the environment setup job to a compute node on partition `cs`:
```bash
sbatch setup_cluster.sh
```
*Note: You can monitor progress with `squeue -u $USER` and `tail -f ede-setup-*.log`.*
Once complete, the environment at `$HOME/ede-runtime/environments/default` will be active, and portable Blender LTS will be installed in `$HOME/ede-runtime/tools/blender` if not already present on the cluster modules.

---

## 3. Matched 32³ Validation Test (24 Cores: 12 Cores per Model)

To verify all components (MPI, GADGET-4, FOF, SUBFIND, power spectrum, kinematics, Blender scene generation, and matched pair comparison) on 1 node using 24 cores total (12 cores for LCDM and 12 cores for EDE in parallel):

```bash
sbatch run_pair_32.sh
```

This runs both models simultaneously and executes `scripts/compare_pair.py --n 32`, generating:
- `paired_density.mp4` (synchronized side-by-side evolution movie)
- `heatmap_abundance_ratio.png` (2D mass–redshift abundance ratio heatmap)
- `heatmap_significance.png` (statistical separation $\sigma$ map)
- `heatmap_counts.png` (resolved halo sample size matrix)
- `hmf_multi_epoch_matrix.csv` (complete numerical table across all redshifts)
- `scientific_summary_z14.44.json` & `scientific_summary_z0.json`
- `index.html` (interactive HTML report)

---

## 4. 512³ Benchmark Run (Pre-flight Scaling & Memory Check)

Before launching the full 122-snapshot production run, run the dedicated 512³ benchmark to measure actual peak RAM, MPI scaling across 64 cores, and parallel I/O throughput over the initial 20 timesteps ($z = 99 \to 90$):

```bash
sbatch benchmark_512.sh
```

---

## 5. Full 512³ Production Launch (2 Nodes, 64 Cores Each)

To launch the full production campaign across two dedicated 64-physical-core nodes with automated Slurm dependency chaining:

```bash
bash submit_512_production.sh
```

### What this submits:
1. **`JOB_LCDM`**: Node 1 (`#SBATCH --nodes=1 --ntasks=64 --cpus-per-task=1 --mem=400G --partition=cs`)
2. **`JOB_EDE`**: Node 2 (`#SBATCH --nodes=1 --ntasks=64 --cpus-per-task=1 --mem=400G --partition=cs`)
3. **`JOB_COMP`**: Matched pair cross-comparison (`#SBATCH --dependency=afterok:$JOB_LCDM:$JOB_EDE`)
4. **`JOB_RENDER`**: Optional, non-blocking 3D Blender movie rendering (`render_512.sh`).

---

## 6. Sync Results to Local Machine

When complete, pull down only the lightweight science results, comparison heatmaps, movies, and reports without transferring the ~16 TB of raw snapshot files:

```bash
# Run on your local computer terminal:
bash scripts/sync_results.sh bbird@hpc.highpoint.edu:~/ede-runtime ~/ede-runtime
```

---

## 7. Standalone Analysis & One-Command Studies

### Run Standalone Post-Simulation Analysis
Use the root CLI `run_analysis.py` to analyze any saved snapshot run:
```bash
python run_analysis.py <run_dir> --graphs available --find-halos
```

### Run Unified End-to-End Study Driver
To run an automated study (linear solver -> simulation -> halo finding -> target selection -> streams -> scene export) with a single command:
```bash
# 1. Preview execution plan (read-only dry run)
python commands.py study configs/study.example.json --dry-run

# 2. Execute study
python commands.py study configs/study.example.json
```
