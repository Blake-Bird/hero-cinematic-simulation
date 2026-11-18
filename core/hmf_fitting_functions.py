"""Additional, analysis-only HMF comparisons; never changes a simulation or old plot.

Equations: Murray, Power & Robotham 2013, arXiv:1306.6721v1, Table 1.
Reed07 coefficients follow Reed et al. 2007 eqs. 11-12 (ca=0.764,
c=1.08); see https://hmf.readthedocs.io/en/latest/_modules/hmf/mass_function/fitting_functions.html.
All fits here use FOF b=0.2. No SO mass conversion is attempted.
"""
from pathlib import Path
import csv
import json
import math

import numpy as np
from scipy.integrate import simpson

LABELS = {
    "press_schechter": "Press-Schechter (1974)",
    "sheth_tormen": "Sheth-Tormen (1999/2001)",
    "reed_2007": "Reed et al. (2007)",
    "watson_fof": "Watson FOF (2013)",
    "jenkins": "Jenkins et al. (2001)",
    "warren": "Warren et al. (2006)",
}
SOURCES = ["https://arxiv.org/abs/1306.6721",
           "https://hmf.readthedocs.io/en/latest/_modules/hmf/mass_function/fitting_functions.html"]


def multiplicity_functions(sigma, slope, delta_c=1.686):
    """Return dimensionless f(sigma); slope=d ln(sigma^-1)/d ln(M).

    Reed's n_eff=6*slope-3, NOT a derivative of the tabulated P(k).
    No fitted normalization or particle-count mass correction is applied.
    """
    from core.cosmology import f_sheth_tormen
    sigma, slope = np.broadcast_arrays(np.asarray(sigma, float), np.asarray(slope, float))
    if (np.any(~np.isfinite(sigma)) or np.any(sigma <= 0)
            or np.any(~np.isfinite(slope)) or np.any(slope <= 0)
            or not np.isfinite(delta_c) or delta_c <= 0):
        raise ValueError("Positive finite sigma, variance slope and delta_c are required")
    nu = delta_c / sigma
    x = -np.log(sigma)
    a, c = .764 / 1.08, 1.08
    g1 = np.exp(-(x - .4)**2 / .72)
    g2 = np.exp(-(x - .75)**2 / .08)
    return {
        "press_schechter": np.sqrt(2 / np.pi)*nu*np.exp(-nu**2 / 2),
        "sheth_tormen": f_sheth_tormen(sigma, delta_c=delta_c),
        "reed_2007": (.3222*np.sqrt(2*a/np.pi)*nu
                      *(1 + (a*nu**2)**(-.3) + .6*g1 + .4*g2)
                      *np.exp(-c*a*nu**2/2 - .03*nu**.6/(6*slope)**2)),
        "watson_fof": .282*((2.163/sigma)**1.406 + 1)*np.exp(-1.21/sigma**2),
        "jenkins": .315*np.exp(-np.abs(x + .61)**3.8),
        "warren": .7234*(sigma**(-1.625) + .2538)*np.exp(-1.1982/sigma**2),
    }


def calibration_masks(sigma, mass, z):
    """Table-1 coordinate ranges only, not a claim of cosmology calibration.

    Reed's v1 catalogue range is retained explicitly, even though later hmf
    implementations use a different cutmask. PS/ST have no table range.
    Warren's quoted mass range is recorded as an approximate h-scaled range.
    """
    x = -np.log(sigma)
    return {
        "press_schechter": np.ones_like(x, bool),
        "sheth_tormen": np.ones_like(x, bool),
        "reed_2007": (x > -1.7) & (x < .9) & (0 <= z <= 30),
        "watson_fof": (x > -.55) & (x < 1.31) & (0 <= z <= 30),
        "jenkins": (x > -1.2) & (x < 1.05) & (0 <= z <= 5),
        "warren": (mass > 1e10) & (mass < 1e15) & np.isclose(z, 0, atol=1e-6, rtol=0),
    }


def _linear_inputs(run_dir, cfg, model, z):
    """Validate saved solver inputs, then evolve P using its background D(a)."""
    from core.analysis import _manifest_artifact_path
    from core.cosmology import class_background_table, pk_output_redshift
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if str(manifest.get("model", "")).lower() != model:
        raise ValueError("run model disagrees with its manifest")
    artifacts = manifest["input_artifacts"]
    paths = {}
    for key in ("linear_power_spectrum", "boltzmann_background"):
        paths[key], error = _manifest_artifact_path(root, artifacts.get(key))
        if error:
            raise ValueError(error)
    pk = paths["linear_power_spectrum"]
    bg = paths["boltzmann_background"]
    meta = json.loads((root / "input" / (model + "_cosmology.json")).read_text())
    if (str(meta.get("model", "")).lower() != model
            or str(meta.get("engine", "")).lower() not in {"class", "axiclass"}
            or meta.get("pk_sha256") != artifacts["linear_power_spectrum"]["sha256"]
            or Path(meta.get("pk_file", "")).name != pk.name
            or Path(meta.get("background_file", "")).name != bg.name):
        raise ValueError("solver metadata disagrees with checksummed inputs")
    z_input = float(pk_output_redshift(pk))
    for recorded in (meta["output_redshift"], artifacts["linear_power_spectrum"]["redshift"]):
        if not np.isclose(z_input, float(recorded), rtol=0, atol=5.1e-4):
            raise ValueError("linear-spectrum redshift metadata disagrees")
    table = np.loadtxt(pk, comments="#", ndmin=2)
    if table.shape[1] < 2:
        raise ValueError("linear spectrum needs k and P(k) columns")
    k, power = table[:, 0], table[:, 1]
    if (len(k) < 8 or np.any(~np.isfinite(table[:, :2]))
            or np.any(k <= 0) or np.any(power <= 0) or np.any(np.diff(k) <= 0)):
        raise ValueError("linear k and P(k) must be finite, positive and ordered")
    columns, background = class_background_table(bg)
    a = 1/(1 + background[:, columns["z"]])
    growth = background[:, columns["growth_D"]]
    order = np.argsort(a)
    a, growth = a[order], growth[order]
    if (np.any(~np.isfinite(a)) or np.any(a <= 0) or np.any(a > 1)
            or np.any(np.diff(a) <= 0) or np.any(~np.isfinite(growth)) or np.any(growth <= 0)):
        raise ValueError("invalid or duplicate background growth samples")
    targets = np.array([1/(1+z_input), 1/(1+z)])
    if np.any(targets < a[0]) or np.any(targets > a[-1]):
        raise ValueError("requested epoch lies outside saved growth table")
    d_input, d_target = np.interp(np.log(targets), np.log(a), growth)
    scale = float((d_target/d_input)**2)
    cosmo = cfg["cosmology"]
    h = float(cosmo["H0"])/100
    omega = (float(cosmo["omega_b"]) + float(cosmo["omega_cdm"]))/h**2
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("invalid mean cold+baryon density")
    return k, power*scale, omega, {
        "input_power_redshift": z_input, "growth_rescaling": scale,
        "input_artifacts": {key: {"name": paths[key].name, "sha256": artifacts[key]["sha256"]}
                            for key in paths},
        "growth_method": "saved background D(a), log(a) interpolation; scale-independent approximation",
    }


def binned_predictions(k, power, omega, edges, z):
    """Integrate dn/dlog10M across each measured bin, without k extrapolation."""
    from core.cosmology import compute_sigma_and_dlnsigma, lagrangian_radius_from_mass
    from core.foundation import RHO_CRIT_MSUN_H_MPC_H3
    edges = np.asarray(edges, float)
    if (edges.ndim != 1 or len(edges) < 3 or np.any(~np.isfinite(edges))
            or np.any(edges <= 0) or np.any(np.diff(edges) <= 0)):
        raise ValueError("at least two positive ordered mass bins are required")
    # 32 Simpson panels per bin; centers occur at sample 16.
    log_grid = np.linspace(np.log10(edges[:-1]), np.log10(edges[1:]), 33, axis=1)
    mass = 10**log_grid
    radius = lagrangian_radius_from_mass(mass, omega)
    covered = (k[0]*radius <= .05) & (k[-1]*radius >= 20)
    valid_bins = np.all(covered, axis=1)
    if np.count_nonzero(valid_bins) < 2:
        raise ValueError("fewer than two complete mass bins have k_min R <= .05 and k_max R >= 20")
    selected_mass = mass[valid_bins].ravel()
    sigma, slope = compute_sigma_and_dlnsigma(k, power, selected_mass, omega)
    fits = multiplicity_functions(sigma, slope)
    masks = calibration_masks(sigma, selected_mass, z)
    shape = (np.count_nonzero(valid_bins), 33)
    result = {}
    for name, f in fits.items():
        rate = math.log(10)*omega*RHO_CRIT_MSUN_H_MPC_H3/selected_mass*f*slope
        averaged = np.full(len(edges)-1, np.nan)
        averaged[valid_bins] = simpson(rate.reshape(shape), x=log_grid[valid_bins], axis=1)/np.diff(np.log10(edges))[valid_bins]
        in_range = np.zeros(len(averaged), bool)
        in_range[valid_bins] = np.all(masks[name].reshape(shape), axis=1)
        f_center = np.full_like(averaged, np.nan)
        f_center[valid_bins] = f.reshape(shape)[:, 16]
        result[name] = {"dn_dlog10M": averaged, "table_range_mask": in_range,
                        "multiplicity_at_center": f_center}
    sigma_center, slope_center = np.full(len(edges)-1, np.nan), np.full(len(edges)-1, np.nan)
    sigma_center[valid_bins] = sigma.reshape(shape)[:, 16]
    slope_center[valid_bins] = slope.reshape(shape)[:, 16]
    return result, sigma_center, slope_center, valid_bins


def write_hmf_fit_comparisons(run_dir, cfg, snap, model, z, measured_hmf):
    """Add diagnostic figures and separate tables; preserve existing outputs."""
    from core.foundation import atomic_json, file_sha256
    output = Path(run_dir) / "results" / "hmf_fitting_functions"
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "hmf-fitting-function-comparisons-v1", "status": "SKIPPED",
              "science_acceptance": "NOT_ACCEPTED", "snapshot": int(snap),
              "model": str(model), "sources": SOURCES, "fits": LABELS,
              "implementation_sha256": file_sha256(Path(__file__)),
              "physics": str(cfg.get("simulation", {}).get("physics", "unknown")),
              "mass_definition": "FOF b=0.2", "delta_c": 1.686,
              "calibration_policy": "All curves diagnostic; dashed segments outside Table-1 coordinate ranges. PS/ST have no listed range. Warren mass range is approximate in Msun/h.",
              "limitations": "Fits are CDM DMO references, not calibrated for EDE, WDM, SIDM or hydro. Model spread is not uncertainty. No FOF particle-mass correction, finite-volume correction or precision validation is implied."}
    destination = output / ("comparison_%03d.json" % snap)
    model = str(model).lower()
    try:
        z = float(z)
        if not np.isfinite(z) or z < 0:
            raise ValueError("invalid snapshot redshift")
        if measured_hmf is None:
            raise ValueError("measured HMF is unavailable")
        if str(cfg["analysis"]["hmf_mass_definition"]).lower() != "fof":
            raise ValueError("FOF fits cannot be compared to SO masses without a calibrated conversion")
        if not np.isclose(float(cfg.get("halos", {}).get("fof_linking_length", .2)), .2, atol=1e-12, rtol=0):
            raise ValueError("these fits require FOF linking length b=0.2")
        if model not in {"lcdm", "ede", "wdm"}:
            raise ValueError("unknown cosmology model")
        if int(cfg["cosmology"].get("N_ncdm", 0) or 0) != 0:
            raise ValueError("massive-neutrino growth and matter conventions need a separate reference")
        k, power, omega, provenance = _linear_inputs(run_dir, cfg, model, z)
        fits, sigma, slope, covered = binned_predictions(k, power, omega, measured_hmf[4], z)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        report["reason"] = str(exc)
        atomic_json(destination, report)
        return report
    report.update(provenance)
    report.update(status="DIAGNOSTIC_ONLY", reason=None, redshift=z, omega_m0=omega,
                  supported_bin_count=int(covered.sum()), bin_integration="32-panel Simpson in log10(M)",
                  cosmology_scope="CDM reference" if model == "lcdm" else "EXTRAPOLATION: " + model.upper())
    artifacts = _write_plots_and_table(output, snap, measured_hmf, fits, sigma, slope,
                                     covered, cfg, report)
    report["artifacts"] = artifacts
    atomic_json(destination, report)
    return report


def count_diagnostics(counts, expected, edges, volume):
    """Finite mass-window statistics; missing prediction bins propagate to tails."""
    counts, expected, edges = (np.asarray(x, dtype=float) for x in (counts, expected, edges))
    if (counts.ndim != 1 or expected.shape != counts.shape or edges.shape != (counts.size+1,)
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.isfinite(edges).all() or np.any(edges <= 0) or np.any(np.diff(edges) <= 0)
            or not np.isfinite(volume) or volume <= 0):
        raise ValueError('Invalid HMF counts, edges or volume')
    valid = np.isfinite(expected) & (expected > 0)
    deviance = np.full(counts.shape, np.nan)
    n, mu = counts[valid], expected[valid]
    term = mu-n
    positive = n > 0
    term[positive] += n[positive]*np.log(n[positive]/mu[positive])
    deviance[valid] = np.sign(n-mu)*np.sqrt(2*np.maximum(term, 0))
    return {'measured_cumulative_number_density': np.cumsum(counts[::-1])[::-1]/volume,
            'fit_cumulative_number_density': np.cumsum(np.where(valid, expected, np.nan)[::-1])[::-1]/volume,
            'signed_poisson_deviance': deviance}


def _write_plots_and_table(output, snap, measured, fits, sigma, slope, covered, cfg, report):
    import matplotlib.pyplot as plt
    from core.analysis import poisson_rate_interval
    from core.figures import export_figure
    from core.foundation import RHO_CRIT_MSUN_H_MPC_H3
    mass, density, _, counts, edges = (np.asarray(x) for x in measured)
    volume = float(cfg["simulation"]["box_mpc_h"])**3
    widths = np.diff(np.log10(edges))
    lower, upper = poisson_rate_interval(counts, volume, widths)
    populated = counts > 0
    colors = dict(zip(LABELS, plt.get_cmap("tab10").colors))
    title = f"{report['model'].upper()} | z={report['redshift']:.3g} | FOF b=0.2 | diagnostic"
    artifacts = []

    def save(fig, suffix):
        name = f"HMF_fits_{suffix}_{snap:03d}.png"
        fig.tight_layout(rect=(0, .16, 1, .92), h_pad=3.2, w_pad=2.0)
        fig.suptitle(title, fontsize=11)
        fig.text(.5, .015, "Dashed: outside table range; all fits are CDM references.\n"
                 "EDE/WDM/hydro/SIDM comparisons are extrapolations; spread is not uncertainty.",
                 ha="center", fontsize=8)
        export_figure(fig, output/name, metadata=report)
        artifacts.append(name)

    def model_line(ax, name, values, x=mass):
        mask = fits[name]["table_range_mask"]
        ax.plot(x, np.where(~mask, values, np.nan), color=colors[name], ls="--", lw=1.3)
        ax.plot(x, np.where(mask, values, np.nan), color=colors[name], lw=1.6, label=LABELS[name])

    def measurement(ax, scale=1):
        scale = np.broadcast_to(scale, density.shape)
        use = populated & np.isfinite(scale) & (scale > 0)
        ax.errorbar(mass[use], (density*scale)[use],
                    yerr=[((density-lower)*scale)[use], ((upper-density)*scale)[use]],
                    fmt="o", color="black", ms=3, capsize=2, label="Measured (68.27% Poisson)")
        empty = (~populated) & np.isfinite(scale) & (scale > 0)
        ax.scatter(mass[empty], (upper*scale)[empty], marker="v", color="black", s=16,
                   label="Zero-count upper limit" if empty.any() else None)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in fits:
        model_line(ax, name, fits[name]["dn_dlog10M"])
    measurement(ax)
    ax.set(xscale="log", yscale="log", xlabel=r"$M\ [M_\odot/h]$",
           ylabel=r"$dn/d\log_{10}M\ [(h/{\rm Mpc})^3]$")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "overlay")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, name in zip(axes.flat, fits):
        prediction = fits[name]["dn_dlog10M"]
        ok = covered & (prediction > 0)
        use = ok & populated
        ax.errorbar(mass[use], density[use]/prediction[use]-1,
                    yerr=[(density-lower)[use]/prediction[use], (upper-density)[use]/prediction[use]],
                    fmt="o", color=colors[name], ms=3, capsize=2)
        empty = ok & ~populated
        ax.scatter(mass[empty], upper[empty]/prediction[empty]-1, marker="v", color=colors[name])
        outside = use & ~fits[name]["table_range_mask"]
        ax.scatter(mass[outside], density[outside]/prediction[outside]-1,
                   facecolors="none", edgecolors="black", s=38, label="Outside table range")
        ax.axhline(0, color="gray", lw=.8)
        ax.set(xscale="log", title=LABELS[name], xlabel=r"$M\ [M_\odot/h]$", ylabel="Measured / fit - 1")
        if outside.any():
            ax.legend(fontsize=7)
    save(fig, "residuals")

    fig, ax = plt.subplots(figsize=(9, 5))
    reference = fits["sheth_tormen"]["dn_dlog10M"]
    for name in fits:
        with np.errstate(invalid="ignore", divide="ignore"):
            model_line(ax, name, fits[name]["dn_dlog10M"]/reference)
    ax.axhline(1, color="gray", lw=.8)
    ax.set(xscale="log", xlabel=r"$M\ [M_\odot/h]$", ylabel="Fit / Sheth-Tormen (bin averaged)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "ratios")

    fig, ax = plt.subplots(figsize=(9, 6))
    inverse_jacobian = mass/(math.log(10)*report["omega_m0"]*RHO_CRIT_MSUN_H_MPC_H3*slope)
    for name in fits:
        model_line(ax, name, fits[name]["multiplicity_at_center"])
    measurement(ax, inverse_jacobian)
    ax.set(xscale="log", yscale="log", xlabel=r"$M\ [M_\odot/h]$", ylabel=r"$f(\sigma)$ (binned estimate for data)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "multiplicity")

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in fits:
        model_line(ax, name, fits[name]["dn_dlog10M"]*volume*widths)
    measurement(ax, volume*widths)
    ax.axhline(1, color="gray", lw=.8, ls=":")
    ax.set(xscale="log", yscale="log", xlabel=r"$M\ [M_\odot/h]$", ylabel="Halos per measured mass bin")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "counts")

    # Every cumulative curve stops at the highest exported edge. These are
    # correlated tail counts, never an extrapolation to infinite halo mass.
    diagnostics = {name: count_diagnostics(counts, fit['dn_dlog10M']*volume*widths, edges, volume)
                   for name, fit in fits.items()}
    cumulative = np.cumsum(counts[::-1])[::-1]
    cum_low, cum_high = poisson_rate_interval(cumulative, volume, np.ones_like(widths))
    fig, ax = plt.subplots(figsize=(9, 6))
    measured_tail = cumulative/volume
    ax.errorbar(edges[:-1], measured_tail, yerr=[measured_tail-cum_low, cum_high-measured_tail],
                fmt='o', ms=3, color='black', label='Measured finite-window tail (correlated)')
    for name in fits:
        tail_mask=np.logical_and.accumulate(fits[name]['table_range_mask'][::-1])[::-1]
        values=diagnostics[name]['fit_cumulative_number_density']
        ax.plot(edges[:-1], np.where(tail_mask,values,np.nan), color=colors[name], label=LABELS[name])
        ax.plot(edges[:-1], np.where(~tail_mask,values,np.nan), color=colors[name], ls='--')
    ax.set(xscale='log', yscale='log', xlabel=r'Lower mass edge [$M_\odot/h$]',
           ylabel=r'$n(M_\mathrm{low} \leq M < M_\mathrm{max})\ [(h/\mathrm{Mpc})^3]$')
    ax.legend(fontsize=7, ncol=2); save(fig, 'cumulative')

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in fits:
        model_line(ax, name, diagnostics[name]['signed_poisson_deviance'])
    ax.axhline(0, color='gray', lw=.8)
    ax.set(xscale='log', xlabel=r'$M\ [M_\odot/h]$', ylabel='Signed Poisson deviance residual (diagnostic)')
    ax.legend(fontsize=8, ncol=2); save(fig, 'deviance')

    fig, ax = plt.subplots(figsize=(9, 6))
    # Bin-center approximation is explicit, not an integrated mass density.
    for name in fits:
        model_line(ax, name, mass*fits[name]['dn_dlog10M'])
    measurement(ax, mass)
    ax.set(xscale='log', yscale='log', xlabel=r'$M\ [M_\odot/h]$',
           ylabel=r'$M_\mathrm{center}\,dn/d\log_{10}M\ [M_\odot h^2/\mathrm{Mpc}^3]$')
    ax.legend(fontsize=8, ncol=2); save(fig, 'mass_contribution')

    fig, ax = plt.subplots(figsize=(9, 5))
    stack = np.array([fit['dn_dlog10M'] for fit in fits.values()])
    supported = np.all(np.isfinite(stack) & (stack > 0), axis=0)
    spread = np.full(mass.shape, np.nan)
    spread[supported] = stack[:, supported].max(axis=0)/stack[:, supported].min(axis=0)
    ax.plot(mass, spread, color='purple', marker='o', ms=3)
    ax.set(xscale='log', xlabel=r'$M\ [M_\odot/h]$', ylabel='Largest / smallest fit (model spread, not uncertainty)')
    save(fig, 'spread')

    fig, ax = plt.subplots(figsize=(10, 5))
    support = np.array([np.where(covered, np.where(fit['table_range_mask'], 2, 1), 0) for fit in fits.values()])
    from matplotlib.colors import ListedColormap, BoundaryNorm
    image = ax.pcolormesh(edges, np.arange(len(fits)+1), support,
                         cmap=ListedColormap(['#bbbbbb', '#e69f00', '#009e73']),
                         norm=BoundaryNorm([-.5,.5,1.5,2.5], 3), shading='flat')
    ax.set_yticks(np.arange(len(fits))+.5, [LABELS[name] for name in fits])
    ax.set(xscale='log', xlabel=r'$M\ [M_\odot/h]$')
    bar=fig.colorbar(image, ax=ax, ticks=[0,1,2]);bar.ax.set_yticklabels(['Missing k support','Outside listed range','Listed range / no limit listed'])
    save(fig, 'coverage')

    diagnostic_table = output/f'diagnostics_{snap:03d}.csv'
    with diagnostic_table.open('w', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['mass_msun_h','lower_edge_msun_h','upper_window_edge_msun_h',
                         'measured_cumulative_number_density_h3_mpc3','cumulative_poisson_68_lower',
                         'cumulative_poisson_68_upper','measured_mass_center_dn_dlog10M', 'fit_max_over_min']
                        +[f'{name}_{quantity}' for name in fits for quantity in
                          ('cumulative_number_density_h3_mpc3','signed_poisson_deviance','mass_center_dn_dlog10M')])
        for i in range(len(mass)):
            row=[mass[i],edges[i],edges[-1],measured_tail[i],cum_low[i],cum_high[i],mass[i]*density[i],spread[i]]
            for name in fits:
                row.extend([diagnostics[name]['fit_cumulative_number_density'][i],diagnostics[name]['signed_poisson_deviance'][i],mass[i]*fits[name]['dn_dlog10M'][i]])
            writer.writerow(row)
    artifacts.append(diagnostic_table.name)

    table = output / f"comparison_{snap:03d}.csv"
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mass_msun_h", "bin_left_msun_h", "bin_right_msun_h", "count", "measured_dn_dlog10M_h3_mpc3",
                         "poisson_68_lower", "poisson_68_upper", "sigma", "dln_sigma_inverse_dlnM", "k_covered"]
                        + [f"{name}_{quantity}" for name in fits for quantity in
                           ("dn_dlog10M_h3_mpc3", "expected_count", "measurement_over_fit", "table_range")])
        for i in range(len(mass)):
            row = [mass[i], edges[i], edges[i+1], counts[i], density[i], lower[i], upper[i], sigma[i], slope[i], int(covered[i])]
            for name, fit in fits.items():
                prediction = fit["dn_dlog10M"][i]
                row.extend([prediction, prediction*volume*widths[i], density[i]/prediction if prediction > 0 else np.nan,
                            int(fit["table_range_mask"][i])])
            writer.writerow(row)
    return artifacts + [table.name]
