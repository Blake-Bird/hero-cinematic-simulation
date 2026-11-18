#!/usr/bin/env python3
"""
Mission Control Terminal Dashboard for the Hero Cinematic Simulation.
Live-refreshes every 3 seconds with ANSI color formatting.
Monitors both Slurm orchestrator logs and GADGET-4 simulation logs.
"""

import os
import sys
import time
import glob
import re
import math
import subprocess
from pathlib import Path

# ANSI Color Codes
C_RESET   = "\033[0m"
C_BOLD    = "\033[1m"
C_DIM     = "\033[2m"
C_CYAN    = "\033[36m"
C_BCYAN   = "\033[1;36m"
C_BLUE    = "\033[34m"
C_BBLUE   = "\033[1;34m"
C_GREEN   = "\033[32m"
C_BGREEN  = "\033[1;32m"
C_YELLOW  = "\033[33m"
C_BYELLOW = "\033[1;33m"
C_MAGENTA = "\033[35m"
C_BMAG    = "\033[1;35m"
C_RED     = "\033[31m"
C_BRED    = "\033[1;31m"
C_WHITE   = "\033[37m"
C_BWHITE  = "\033[1;37m"

def bar(frac, width=26, color=C_BCYAN):
    frac = max(0.0, min(1.0, frac))
    filled = int(round(width * frac))
    empty = width - filled
    return f"{color}[{'█'*filled}{'░'*empty}]{C_RESET} {frac*100:5.1f}%"

def get_job_info():
    """Query Slurm for the active hero-4k job."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", "bbird"), "-h", "-o", "%i|%P|%j|%t|%M|%R|%m"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        lines = [l for l in out.splitlines() if l.strip()]
        for l in lines:
            parts = l.split("|")
            if len(parts) >= 6 and ("hero" in parts[2].lower() or "setu" in parts[2].lower()):
                return {
                    "id": parts[0],
                    "partition": parts[1],
                    "name": parts[2],
                    "state": parts[3],
                    "time": parts[4],
                    "nodelist": parts[5],
                    "mem": parts[6] if len(parts) > 6 else "400G"
                }
        if lines:
            parts = lines[0].split("|")
            return {
                "id": parts[0], "partition": parts[1], "name": parts[2],
                "state": parts[3], "time": parts[4], "nodelist": parts[5],
                "mem": parts[6] if len(parts) > 6 else "400G"
            }
    except Exception:
        pass
    return None

def find_latest_log():
    """Find the most recently modified hero-cinematic log."""
    candidates = sorted(glob.glob("hero-cinematic-*.log"), key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
    return candidates[-1] if candidates else None

def parse_log(log_path):
    """Extract current stage, redshift, scale factor, and last log lines."""
    state_root = os.environ.get("RESEARCH2_STATE_ROOT", os.path.expanduser("~/ede-runtime"))
    campaign = os.environ.get("CAMPAIGN", "hero_universe")
    run_dir = Path(state_root) / "campaigns" / campaign / "run"
    gadget_log = run_dir / "input/gadget.log"

    info = {
        "stage": 1,
        "stage_name": "Stage 1: AxiCLASS EDE & Monofonic 3LPT",
        "redshift": None,
        "a": None,
        "step": None,
        "lines": [],
        "completed": False,
        "error": None,
        "log_source": "orchestrator",
        "gadget_log_exists": gadget_log.is_file(),
    }

    if log_path and os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            
            info["lines"] = [l.rstrip() for l in all_lines[-7:]]
            full_text = "".join(all_lines)

            # Determine stage from entire file history
            if "--- Stage 8:" in full_text:
                info["stage"] = 8
                info["stage_name"] = "Stage 8: FFmpeg 4K Master Video Encoding"
            elif "--- Stage 7:" in full_text:
                info["stage"] = 7
                info["stage_name"] = "Stage 7: Blender Cycles 4K Photorealistic Render"
            elif "--- Stage 6:" in full_text:
                info["stage"] = 6
                info["stage_name"] = "Stage 6: Multi-Scale Camera Trajectory"
            elif "--- Stage 5:" in full_text:
                info["stage"] = 5
                info["stage_name"] = "Stage 5: Gala Stellar Streams Disruption"
            elif "--- Stage 4:" in full_text:
                info["stage"] = 4
                info["stage_name"] = "Stage 4: Advanced Category-B Astrophysics Suite"
            elif "--- Stage 3:" in full_text:
                info["stage"] = 3
                info["stage_name"] = "Stage 3: Halo Finding (Rockstar / FOF / Subfind)"
            elif "--- Stage 2:" in full_text:
                info["stage"] = 2
                info["stage_name"] = "Stage 2: GADGET-4 Cosmological Evolution (512³)"

            if "The Ultimate Hero Cinematic Run Successfully Finished!" in full_text:
                info["completed"] = True

            # Check for Python tracebacks or errors
            for i, line in enumerate(all_lines[-40:]):
                if "Traceback (most recent call last):" in line:
                    info["error"] = "".join(all_lines[-40+i:]).strip()
                    break
                elif "CalledProcessError" in line:
                    info["error"] = line.strip()

        except Exception as e:
            info["error"] = str(e)

    # If GADGET-4 log exists, parse it for high-resolution simulation telemetry
    if gadget_log.is_file():
        try:
            with open(gadget_log, "r", encoding="utf-8", errors="replace") as gf:
                glines = gf.readlines()
            
            if glines:
                # If Stage 2 is active or gadget.log was modified within 60s, tail gadget.log
                g_mtime = os.path.getmtime(gadget_log)
                if info["stage"] == 2 or (time.time() - g_mtime < 120):
                    info["lines"] = [l.rstrip() for l in glines[-7:]]
                    info["log_source"] = "gadget4"

                gcontent = "".join(glines[-200:])

                # Extract Redshift / Scale factor / Sync point from live GADGET evolution logs
                # Format: "Sync-Point 12, Time: 0.02345, Redshift: 41.64, Systemstep: ..."
                sync_matches = re.findall(r'Sync-Point\s+(\d+),\s*Time:\s*([0-9.eE+-]+),\s*Redshift:\s*([0-9.eE+-]+)', gcontent)
                if sync_matches:
                    last_step, last_time, last_z = sync_matches[-1]
                    info["step"] = int(last_step)
                    info["a"] = float(last_time)
                    info["redshift"] = float(last_z)
                else:
                    info["step"] = 0
                    info["a"] = 0.01
                    info["redshift"] = 99.0

                # Check for GADGET Terminate or Abort
                for gl in glines[-30:]:
                    if any(kw in gl for kw in ("Terminate", "terminate", "Abort", "abort", "fatal", "Fatal", "We stop because", "errorFlag")):
                        info["error"] = f"[GADGET-4 ABORT] {gl.strip()}"
                        break
        except Exception:
            pass

    return info

def get_snapshots():
    """Scan the runtime directory for written snapshots."""
    state_root = os.environ.get("RESEARCH2_STATE_ROOT", os.path.expanduser("~/ede-runtime"))
    campaign = os.environ.get("CAMPAIGN", "hero_universe")
    run_dir = Path(state_root) / "campaigns" / campaign / "run"
    
    snaps = []
    total_bytes = 0
    if run_dir.is_dir():
        for pat in ("snapshots/snap_*.hdf5", "snapshots/snapshot_*.hdf5", "snap_*.hdf5", "snapshot_*.hdf5"):
            for f in run_dir.glob(pat):
                try:
                    st = f.stat()
                    total_bytes += st.st_size
                    snaps.append((f.name, st.st_size, st.st_mtime))
                except Exception:
                    pass

    snaps.sort(key=lambda x: x[2], reverse=True)
    return snaps, total_bytes

def render_dashboard():
    job = get_job_info()
    log_file = find_latest_log()
    log_info = parse_log(log_file)
    snaps, total_bytes = get_snapshots()

    # Clear screen and move cursor to home
    out = []
    out.append("\033[H\033[J")
    
    # Header Banner
    out.append(f"{C_BCYAN}╭─────────────────────────────────────────────────────────────────────────────╮{C_RESET}")
    out.append(f"{C_BCYAN}│  {C_BWHITE}🌌  HERO UNIVERSE — 4K CINEMATIC COSMOLOGY MISSION CONTROL{C_BCYAN}                │{C_RESET}")
    out.append(f"{C_BCYAN}╰─────────────────────────────────────────────────────────────────────────────╯{C_RESET}")

    # Job Status Bar
    if job:
        st_color = C_BGREEN if job["state"] in ("R", "RUNNING") else C_BYELLOW
        state_str = "RUNNING" if job["state"] in ("R", "RUNNING") else job["state"]
        out.append(f" {C_BWHITE}JOB ID:{C_RESET} {C_BCYAN}{job['id']}{C_RESET}  "
                   f"{C_BWHITE}STATE:{C_RESET} {st_color}{state_str}{C_RESET}  "
                   f"{C_BWHITE}NODE:{C_RESET} {C_BMAG}{job['nodelist']}{C_RESET}  "
                   f"{C_BWHITE}ELAPSED:{C_RESET} {C_BYELLOW}{job['time']}{C_RESET}  "
                   f"{C_BWHITE}PARTITION:{C_RESET} {job['partition']}")
        out.append(f" {C_DIM}ALLOCATION: 64 Cores (16 MPI × 4 OpenMP) | {job['mem']} RAM | InfiniBand mlx5_0{C_RESET}")
    else:
        out.append(f" {C_BWHITE}JOB STATUS:{C_RESET} {C_DIM}No active Slurm job running (ready for next submission){C_RESET}")

    out.append("")

    # Error Alert Box (if active error)
    if log_info["error"] and not job:
        err_msg = log_info["error"].splitlines()[-1][:70]
        out.append(f"{C_BRED}┌─ ATTENTION: JOB EXECUTION HALTED ──────────────────────────────────────────┐{C_RESET}")
        out.append(f"│  {C_RED}Error: {err_msg:<66}{C_RESET} │")
        out.append(f"│  {C_DIM}Check: tail -n 50 ~/ede-runtime/campaigns/hero_universe/run/input/gadget.log  {C_RESET}│")
        out.append(f"{C_BRED}└─────────────────────────────────────────────────────────────────────────────┘{C_RESET}")

    # Pipeline Stages Progress
    stages = [
        (1, "AxiCLASS EDE & Monofonic 3LPT Initial Conditions"),
        (2, "GADGET-4 Cosmological Evolution (512³ Particles)"),
        (3, "Halo Finders (FOF, SUBFIND, Rockstar) & HMF Curves"),
        (4, "Advanced Category-B Suite (VELOCIraptor, DisPerSE, Lensing)"),
        (5, "Gala / StreaMAX Tidal Stellar Disruption Ribbons"),
        (6, "5-Act Multi-Scale Cinematic Camera Trajectory Splines"),
        (7, "Blender Cycles 4K Photorealistic Master Render"),
        (8, "FFmpeg 4K Master UHD Video Encoding (3840x2160 @ 24fps)")
    ]

    out.append(f"{C_BBLUE}┌─ PIPELINE EXECUTION PIPELINE ───────────────────────────────────────────────┐{C_RESET}")
    cur_stage = log_info["stage"]
    for num, name in stages:
        if log_info["completed"] or cur_stage > num:
            icon = f"{C_BGREEN}✓ COMPLETED{C_RESET}"
            line_style = C_WHITE
        elif cur_stage == num:
            icon = f"{C_BYELLOW}▶ ACTIVE   {C_RESET}"
            line_style = C_BWHITE
        else:
            icon = f"{C_DIM}○ QUEUED   {C_RESET}"
            line_style = C_DIM
        out.append(f"│  {icon} {line_style}Stage {num}: {name[:52]:<52}{C_RESET} │")
    out.append(f"{C_BBLUE}└─────────────────────────────────────────────────────────────────────────────┘{C_RESET}")

    # Simulation Telemetry (Stage 2)
    out.append(f"{C_BMAG}┌─ N-BODY EVOLUTION TELEMETRY ────────────────────────────────────────────────┐{C_RESET}")
    a_val = log_info.get("a") or 0.01
    z_val = log_info.get("redshift")
    z_str = f"{z_val:7.3f}" if z_val is not None else " 99.000"
    a_str = f"{a_val:7.4f}"

    # a goes from 0.01 (z=99) to 1.0 (z=0)
    frac_a = max(0.0, min(1.0, (a_val - 0.01) / 0.99))
    step_str = str(log_info.get("step") if log_info.get("step") is not None else "0")
    
    out.append(f"│  {C_BWHITE}Scale Factor (a):{C_RESET} {C_BCYAN}{a_str}{C_RESET} / 1.0000   "
               f"{C_BWHITE}Redshift (z):{C_RESET} {C_BYELLOW}{z_str}{C_RESET}   "
               f"{C_BWHITE}Time-Step:{C_RESET} {C_WHITE}{step_str:<6}{C_RESET} │")
    out.append(f"│  {C_BWHITE}Expansion (z=99 → 0):{C_RESET}  {bar(frac_a, 44, C_BMAG)} │")
    
    n_snaps = len(snaps)
    total_snaps = 122
    frac_snaps = n_snaps / total_snaps
    storage_gib = total_bytes / (1024**3)
    out.append(f"│  {C_BWHITE}Snapshots Saved:{C_RESET}       {bar(frac_snaps, 34, C_BGREEN)} {n_snaps:>3}/{total_snaps} snaps │")
    out.append(f"│  {C_BWHITE}Disk Output:{C_RESET}           {C_BCYAN}{storage_gib:6.1f} GiB{C_RESET} (Estimated: ~488 GiB total)               │")
    out.append(f"{C_BMAG}└─────────────────────────────────────────────────────────────────────────────┘{C_RESET}")

    # Recent Snapshot Files
    out.append(f"{C_BWHITE}┌─ RECENT SNAPSHOT ARTIFACTS ─────────────────────────────────────────────────┐{C_RESET}")
    if snaps:
        for name, size, mtime in snaps[:3]:
            size_mb = size / (1024**2)
            t_str = time.strftime("%H:%M:%S", time.localtime(mtime))
            out.append(f"│  {C_BGREEN}●{C_RESET} {C_WHITE}{name:<25}{C_RESET}  {C_CYAN}{size_mb:6.1f} MB{C_RESET}  {C_DIM}Time: {t_str}{C_RESET}                        │")
    else:
        out.append(f"│  {C_DIM}Initial conditions staged (134,217,728 particles in 8 shards); evolution ready │{C_RESET}")
    out.append(f"{C_BWHITE}└─────────────────────────────────────────────────────────────────────────────┘{C_RESET}")

    # Live Log Snippet
    source_tag = "GADGET-4 ENGINE" if log_info["log_source"] == "gadget4" else "SLURM PIPELINE"
    log_title = f" LIVE LOG: {source_tag} ({log_file or 'waiting for log'}) "
    dash_count = max(0, 75 - len(log_title))
    out.append(f"{C_BCYAN}┌─{log_title}{'─'*dash_count}┐{C_RESET}")
    if log_info["lines"]:
        for l in log_info["lines"][-5:]:
            clean = l[:72]
            out.append(f"│  {C_DIM}{clean:<72}{C_RESET} │")
    else:
        out.append(f"│  {C_DIM}{'Waiting for simulation output stream...':<72}{C_RESET} │")
    out.append(f"{C_BCYAN}└─────────────────────────────────────────────────────────────────────────────┘{C_RESET}")

    out.append(f"{C_DIM} Auto-refreshing every 3s • Press {C_BWHITE}Ctrl+C{C_DIM} to detach (run keeps going in Slurm){C_RESET}")

    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()

def main():
    try:
        while True:
            render_dashboard()
            time.sleep(3)
    except KeyboardInterrupt:
        print(f"\n{C_BCYAN}Mission Control detached. Simulation continues uninterrupted in background.{C_RESET}\n")

if __name__ == "__main__":
    main()
