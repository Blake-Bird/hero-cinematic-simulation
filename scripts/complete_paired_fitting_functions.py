#!/usr/bin/env python3
"""Complete and expose fitting-function comparisons at every highlighted epoch."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import hmf_mass_edges, load_cfg
from core.foundation import atomic_json, file_sha256
from core.hmf_fitting_functions import write_hmf_fit_comparisons, LABELS
from core.paired_atlas import epoch_name, _write_csv
from scripts.prepare_paired_analysis import prepare

EPOCHS = ((21,14.4),(26,12.),(30,10.),(35,8.),(41,6.),(54,3.),(69,1.),(85,0.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    root = parser.parse_args().atlas.resolve()
    sources = {}
    for model in ("EDE", "LCDM"):
        work = prepare(root.parent/f"dmo3_N128_L5_fast128_{model.lower()}",
                       root/"Source/Native_workspaces"/f"{model}_fit_completion")
        cfg = load_cfg(work)
        bundle = sorted((root/"Source/Native_workspaces"/model).glob("results/analysis_products_*/hmf_fitting_functions"))[-1]
        sources[model] = {}
        for snap, z in EPOCHS:
            table = bundle/f"comparison_{snap:03d}.csv"
            if not table.exists():
                measurement = root/"06_Tables_and_Data/Measurements"/model/(epoch_name(model,snap,z)+".npz")
                meta = json.loads(measurement.with_suffix(".json").read_text())
                h = meta["summary"]["h"]
                if file_sha256(measurement) != meta["numeric_sha256"]:
                    raise ValueError(f"Changed measured epoch {measurement}")
                with np.load(measurement, allow_pickle=False) as data:
                    masses = data["fof_mass_msun"][data["fof_particle_count"]>=300]*h
                edges = hmf_mass_edges(cfg)
                counts = np.histogram(masses, edges)[0]
                volume = float(cfg["simulation"]["box_mpc_h"])**3
                measured = (np.sqrt(edges[:-1]*edges[1:]), counts/volume/np.diff(np.log10(edges)),
                            counts/volume/np.diff(edges), counts, edges)
                existing = work/f"results/hmf_fitting_functions/comparison_{snap:03d}.csv"
                if not existing.exists():
                    result = write_hmf_fit_comparisons(work,cfg,snap,model.lower(),z,measured)
                    if result["status"] == "SKIPPED":
                        raise ValueError(result["reason"])
                table = existing
                atomic_json(existing.with_name(f"measured_selection_{snap:03d}.json"), {
                    "source":str(measurement), "sha256":meta["numeric_sha256"],
                    "minimum_particles":300,"selected_host_count":int(len(masses)),
                    "empty_bin_policy":"68.27% central Garwood upper limits",
                    "scope":"Explicit sparse-sample reference diagnostic. Native raw counts retained even when the standard suite minimum host count is unmet."})
            sources[model][snap] = table
            print("FIT_TABLE",model,snap,table,flush=True)

    colors = ["#c16e26","#167c86","#9d6bab","#8b9a39","#ce5865","#568bcd"]
    fits = list(LABELS)
    for snap,z in EPOCHS:
        label = epoch_name("EDE_vs_LCDM",snap,z)
        folder = root/"01_Comparisons"/label
        stem = folder/(label+"_Fitting_function_comparison")
        fig, axes = plt.subplots(1,2,figsize=(13,6.6),sharey=True,layout="constrained")
        fig.get_layout_engine().set(rect=(0,.16,1,.84))
        fig.suptitle(f"Measured halo abundance and reference fits · z = {z:g}",fontsize=19)
        tables = []
        records = {}
        for model,h,ax in zip(("EDE","LCDM"),(.7483,.6781),axes):
            path = sources[model][snap]
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            mass = np.array([float(row["mass_msun_h"])/h for row in rows])
            count = np.array([int(row["count"]) for row in rows])
            measured = np.array([float(row["measured_dn_dlog10M_h3_mpc3"])*h**3 for row in rows])
            low = np.array([float(row["poisson_68_lower"])*h**3 for row in rows])
            high = np.array([float(row["poisson_68_upper"])*h**3 for row in rows])
            present = count>0
            ax.errorbar(mass[present],measured[present],yerr=[(measured-low)[present],(high-measured)[present]],
                        fmt="o",ms=4,c="black",capsize=2,label="Measured · 68% counting interval",zorder=4)
            ax.scatter(mass[~present],high[~present],marker="v",s=16,c="#454545",
                       label="Zero-count upper limit",zorder=4)
            for fit,color in zip(fits,colors):
                y = np.array([float(row[fit+"_dn_dlog10M_h3_mpc3"])*h**3 for row in rows])
                valid = np.array([row["k_covered"]=="1" for row in rows])
                calibration = np.array([row[fit+"_table_range"]=="1" for row in rows])
                ax.plot(mass,np.where(valid&calibration,y,np.nan),color=color,label=LABELS[fit],lw=1.5)
                ax.plot(mass,np.where(valid&~calibration,y,np.nan),color=color,ls="--",lw=1.2)
                for i,row in enumerate(rows):
                    tables.append({"model":model,"snapshot":snap,"redshift":z,"fit":fit,"mass_msun":mass[i],
                                   "count":int(count[i]),"measured_dn_dlog10M_mpc3":measured[i],
                                   "counting_68_lower":low[i],"counting_68_upper":high[i],
                                   "fit_dn_dlog10M_mpc3":y[i],"input_k_supported":bool(valid[i]),
                                   "fit_calibration_supported":bool(calibration[i])})
            ax.set(xscale="log",yscale="log",xlabel="FOF mass [Msun]",
                   title=f"{model} · {count.sum()} hosts within bins at ≥300 particles")
            ax.set_ylim(max(float(high.min())*.025,1.e-8),max(float(high.max())*5,1.))
            ax.grid(alpha=.18)
            if not present.any():
                ax.text(.97,.95,"Zero selected hosts\nMarkers show counting upper limits",transform=ax.transAxes,
                        ha="right",va="top",fontsize=10)
            records[model]={"table":str(path.relative_to(root)),"sha256":file_sha256(path)}
        axes[0].set_ylabel(r"$dn / d\log_{10} M$ [Mpc$^{-3}$]")
        handles,labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.042),ncol=4,frameon=False,fontsize=9)
        fig.text(.5,.018,"Dashed curves extend beyond fit calibration ranges. EDE uses CDM reference extrapolations. Counting intervals exclude box sample variance.",ha="center",fontsize=9)
        for suffix in (".png",".pdf",".svg"):
            fig.savefig(stem.with_suffix(suffix),dpi=180)
        plt.close(fig)
        _write_csv(stem.with_suffix(".csv"),tables)
        atomic_json(stem.with_suffix(".json"),{"sources":records,"snapshot":snap,"redshift":z,"fits":LABELS,
                     "scope":"Native FOF bins and a 300-particle selection; sparse and empty bins remain visible."})
        manifest_path = folder/"comparison_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["figures"] = sorted(set(manifest["figures"])|{stem.with_suffix(".png").name})
        manifest["tables"] = sorted(set(manifest["tables"])|{stem.with_suffix(".csv").name})
        atomic_json(manifest_path,manifest)
    atomic_json(root/"07_Status_and_Provenance/highlighted_fitting_function_coverage.json", {
        model:{str(snap):{"table":str(path.relative_to(root)),"sha256":file_sha256(path)}
               for snap,path in entries.items()} for model,entries in sources.items()})


if __name__=="__main__":
    main()
