"""Portable, read-only source/figure inventory for the optional teaching lab.

AST inspection never imports backend software or launches a solver. The graph
registry remains the authority for recipe names and required producer routes.
"""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

EQUATIONS = {
 'flyby_events': {'title':'Known flyby and event-time reconstruction',
  'formula':'r(t)² = v²(t−t_peri)² + b²; t_infall = t_peri − sqrt(R²−b²)/v',
  'units':'Proper separation, impact parameter b and boundary R: kpc; time: Myr; proper speed: kpc/Myr (converted to/from peculiar km/s with H x).',
  'read':'The 32-particle authored flyby has three saved epochs, a fixed 1.5-kpc boundary, b=0.5 kpc and v=0.4 kpc/Myr. The original graph/image/clock pipeline recovers candidate infall and pericentre times. The saved brackets and interpolation-uncertified status remain; this is not a physical merger or finder detection.',
  'implementation':'core/assembly.py','function':'orbital_assembly_events'},
 'shape_tensor': {'title':'Known-shape recovery from mass moments',
  'formula':'I_ij = Σ m x_i x_j / Σ m; b/a = sqrt(λ1/λ0); c/a = sqrt(λ2/λ0); θ = acos(|e_inner · e_outer|)',
  'units':'Offsets: proper kpc; tensor: kpc²; mass: Msun; ratios: dimensionless; unsigned twist: degrees.',
  'read':'The 6000-point authored triaxial shells have known ratios and rotations. Iterative ellipsoidal selection and bootstrap orientation support must pass before displaying the twist. A fixed centre and complete aperture are retained. This is geometric recovery, not halo convergence.',
  'implementation':'core/halos.py','function':'radial_density_shapes'},
 'splashback_slope': {'title':'Known density valley and correlated derivative uncertainty',
  'formula':'s = D lnρ; C_s = D C_lnρ Dᵀ; width = ln(r_outer/r_inner) at half prominence',
  'units':'Radius: proper kpc; shell density: Msun/kpc³; slope, prominence and log-radius width: dimensionless.',
  'read':'The 32768-particle authored profile has a known valley. The primary seven-bin GLS derivative keeps missing bins and its full propagated covariance. Correlated Gaussian covariance draws measure conditional feature support; they are not independent particle simulations. Five- and nine-bin alternatives stay separately recorded.',
  'implementation':'core/analysis.py','function':'uncertainty_aware_log_density_derivative'},
 'cooling_source': {'title':'Native gas energy and implicit cooling',
  'formula':'u_next = u_before + Δt L_net(T_next, composition_next) / ρ; u_cgs = u_saved × velocity_unit²',
  'units':'u: erg/g; net volumetric heating/cooling L_net: erg/(cm³ s); ρ: g/cm³; source Δt: s.',
  'read':'The gas control evolves hydrodynamics and native atomic cooling. Its mass-weighted specific energy includes pressure work as well as the cooling source. Read the original energy arrays and unit attributes. This is a short 64-gas execution control; it does not establish cooling convergence.',
  'implementation':'core/evolution.py','function':'gadget_param_text'},
 'stellar_birth': {'title':'Native SH03 stellar birth opportunities',
  'formula':'t_star(ρ) = t0 sqrt(ρ_threshold / ρ); dotρ_locked = (1 − β) x ρ / t_star',
  'units':'ρ: proper gas density; t_star: proper time; β: prompt returned fraction; x: cold-cloud mass fraction.',
  'read':'The native source samples stellar births with exact particle IDs, counters and StellarRandomSeed. This control starts with no stars. New SSPBirthMode=1 records native SH03 births; seeded mode 2 is distinct. Gas energy on the effective multiphase EOS is not a single-phase temperature. The gas/star census and source journal are reconciled by the production history audit. Delayed stellar returns are disabled.',
  'implementation':'core/evolution.py','function':'hydro_star_formation_parameters'},
 'bh_accretion': {'title':'Native seeded black-hole accretion',
  'formula':'dotM_in = min[α f_rot 4πG² M_BH² ρ/(c_s²+v_rel²)^(3/2), dotM_Edd]; dotM_BH = (1−ε_r) dotM_in; ΔE_feedback = ε_f ε_r dotM_in c² Δt',
  'units':'Physical BH mass: Msun; inflow: Msun/year; density: g/cm³; velocities: cm/s in the native source. Saved comoving Header/Time is scale factor a.',
  'read':'One 10-Msun BH is explicitly supplied with 64 gas particles. The analytic gas IC is compressed before evolution, and the original sealed grid remains retained. Native fields show physical mass and inflow separately from gravitational particle mass. Positive growth/inflow checks execution; full source-budget reconciliation and resolved BH accuracy are separate requirements.',
  'implementation':'core/evolution.py','function':'gadget_param_text'},
 'linear_bundle': {'title':'CLASS power and dimensionless particle handoff',
  'formula':'Δ²(k,z) = k³ P_CLASS(k,z) / (2π²); handoff columns = log10(k), log10(Δ²)',
  'units':'k: h/Mpc; P_CLASS: (Mpc/h)³; Δ²: dimensionless. The handoff uses the saved start redshift.',
  'read':'These arrays come from actual linear Boltzmann solver outputs, not from the quick lab particle FFT. CLASS and AxiCLASS run separately at z=49 and z=0. Inspect the original model, density/velocity transfer gauge, scalar-field scope, source hashes and background conventions before preparing particles. A completed spectrum is not a nonlinear simulation or a calibrated cosmological constraint.',
  'implementation':'core/linear.py','function':'export_power'},
 'local_response': {'title':'Calibrated parameter-response modes',
  'formula':'Jᵢⱼ = ∂yᵢ/∂θⱼ; C = L Lᵀ; A = L⁻¹ J diag(s); A = U diag(σ) Vᵀ',
  'units':'J has observable units divided by parameter units. C has squared observable units. Each positive s has its parameter units; A, singular values σ and right-mode loadings are dimensionless.',
  'read':'Independent held-out forward runs must first check the local response. Large singular values identify locally informative scaled parameter combinations; zero values retain unresolved combinations. The 32-particle BH control measures final BH mass, uses scales 10⁻²⁰ cm²/g and 10,000 Msun, and an explicit covariance floor of 10,000 Msun². Its singular values are 100 and 0: initial mass is visible, scattering is unconstrained. This is not a feedback measurement or a calibrated posterior. Mode signs are arbitrary; read parameter order and scales from the original product.',
  'implementation':'core/sidm.py','function':'forward_model_degeneracy'},
 'density': {'title':'Density from actual mass in shells', 'formula':'ρⱼ = Σᵢ∈shell(j) mᵢ / [4π(rⱼ₊₁³ − rⱼ³)/3]',
  'units':'m: Msun; r: proper kpc; ρ: Msun / proper kpc³',
  'read':'Shell averages are not pointwise density. Empty shells have measured zero mass; log plots mask zero. At 32 particles the sampling noise is large.',
  'implementation':'optional_tests/lab.py', 'function':'measure_density'},
 'power': {'title':'Periodic mesh power', 'formula':'δ = Ncell/⟨Ncell⟩ − 1; P(k) = V |FFT(δ)|² / Nmesh⁶',
  'units':'In this lab: k in 1/proper kpc; P in proper kpc³. Cosmological registered products use their saved Mpc and h conventions.',
  'read':'The lab uses NGP assignment and includes sampling/assignment effects. It is not the native GADGET folded, CIC-deconvolved, shot-noise-subtracted spectrum.',
  'implementation':'core/analysis.py','function':'fft_power_spectrum'},
 'orbit': {'title':'Physical orbit integration', 'formula':'dx/dt = v; dv/dt = −∇Φ; E = |v|²/2 + Φ',
  'units':'x: proper kpc; v: km/s; t: Myr; Φ and E: (km/s)²; code converts km/s to kpc/Myr.',
  'read':'The quick lab integrates tracers in a declared static isochrone potential. Tracers have no self-gravity and are not a native galaxy simulation.',
  'implementation':'core/streams.py','function':'integrate_stream'},
 'circular': {'title':'Spherical-equivalent circular speed', 'formula':'v_c(r) = √[G M(<r)/r]',
  'units':'G = 4.3009… × 10⁻⁶ proper kpc (km/s)² / Msun',
  'read':'This is a spherical-equivalent diagnostic. It is not necessarily the rotation speed of a nonspherical system.',
  'implementation':'core/halo_products.py','function':'export_native_structure_figures'},
 'covariance': {'title':'Weighted population velocity covariance', 'formula':'pᵢ = mᵢ/Σm; μ = Σpᵢvᵢ; C = Σpᵢ(vᵢ−μ)(vᵢ−μ)ᵀ',
  'units':'μ: km/s; C: (km/s)²; effective count: 1/Σpᵢ²',
  'read':'This describes the finite measured population, not an unbiased covariance estimate of an unknown population or the covariance of the mean.',
  'implementation':'core/stream_common.py','function':'weighted_covariance'},
 'units': {'title':'Native to physical units', 'formula':'physical = saved × to_cgs × a^(a_scaling) × h^(h_scaling)',
  'units':'The synthetic lab uses isolated proper coordinates, h=1, zero a/h exponents. Cosmological data require their original conventions.',
  'read':'Mass, position, velocity and clock conversions are separate. Header/Time is elapsed native time for an isolated file, not a scale factor.',
  'implementation':'core/data.py','function':'read_species_phase_space'},
 'sbc': {'title':'Simulation-based calibration ranks',
  'formula':'rⱼ = count(θⱼ,draw < θⱼ,true) + randomized tie count; PITⱼ = (rⱼ + U)/(Ndraw + 1)',
  'units':'Rank is an integer from 0 through Ndraw. PIT and its empirical CDF are dimensionless; U is uniform on [0,1).',
  'read':'The CDF uses one selected test case per independent parent/lineage component. A diagonal-looking curve alone is not acceptance: original finite-draw KS, coverage and multiple-comparison decisions remain attached.',
  'implementation':'core/stream/sbi.py','function':'calibrate_frozen_model'},
 'coverage': {'title':'Independent credible-interval coverage',
  'formula':'Ĉⱼ = Σg 1[θⱼ,true,g ∈ CIⱼ,g] / Nindependent; accept only if the binomial CI lies inside [q − ε, q + ε]',
  'units':'Coverage, nominal credible mass q, tolerance ε and the binomial bounds are dimensionless probabilities.',
  'read':'The posterior is frozen before reading held-out truth. The interval quantifies finite independent test cases; multiple draws from one parent do not increase that count. The short harmonic example fails calibration and remains disabled for deployment.',
  'implementation':'core/stream/sbi.py','function':'calibrate_frozen_model'},
 'ood': {'title':'Training-domain support',
  'formula':'d² = (s − μ)ᵀ C⁻¹(s − μ); foutside = count(sⱼ outside training rangeⱼ)/Nsummary',
  'units':'Summaries use their recorded normalization. Squared Mahalanobis distance and outside fraction are dimensionless.',
  'read':'Mean, covariance regularization and ranges come only from training observations. Both original thresholds must pass. This is a support screen, not a calibrated probability of model correctness; posterior deployment also requires accepted independent calibration.',
  'implementation':'core/stream/sbi.py','function':'evaluate_ood'},
 'graph_gain': {'title':'Paired graph-versus-summary log-score gain',
  'formula':'Δg = log pgraph(θtrue,g | xg) − log psummary(θtrue,g | xg); gain probability = fraction(meanbootstrap(Δ) > declared minimum)',
  'units':'Log-score differences are dimensionless when both densities use the same recorded parameter coordinates and units.',
  'read':'Both frozen models must independently pass calibration on the same held-out lineage cases. Bootstrap resamples paired independent groups. A rejected gain cannot authorize graph deployment, and this conditional evaluation does not establish performance on other physical domains.',
  'implementation':'core/stream/sbi.py','function':'graph_baseline_gain'},
}

FAMILY_NOTES = {
 'cosmology':'Follow the CLASS bundle, native estimator policy, exact saved epoch, k units, support masks and covariance. A ratio is not a new simulation.',
 'abundance':'Counts are normalized by the declared volume and log-bin widths. Particle thresholds do not establish physical completeness.',
 'structure':'Inspect radius, density/mass units, component selection, fit residuals and shell support. A fitted parameter is conditional on its model.',
 'assembly':'Use the same identity graph across saved epochs. Cadence limits derivative, infall and merger timing.',
 'streams':'Check proper positions, peculiar velocities, tracer weights, tail/wrap selection, track coordinates and supported bins.',
 'encounters':'Use common frames, actual trajectories, interpolation certificates and measured impact geometry. Close passage is not proof of a causal gap.',
 'black_holes':'Separate gravity mass, subgrid BH mass, accretion, feedback and committed source events. A marker is not an event horizon.',
 'baryons':'Distinguish thermal gas, effective EOS, seeded SSPs, actual birth events and spectral synthesis. Missing composition must remain missing.',
 'sidm':'Read the interaction model, exact event/participant IDs, exposure clock, kernel support and independent controls.',
 'hubble':'Keep fitted observables, model calibration, covariance, mock/published data and privileged post-fit truth distinct.',
 'inference':'Read actual chain/model diagnostics, priors, heldout lineage and OOD decisions. A rendered posterior can still be unusable.',
 'convergence':'Vary one declared control on matched targets. Retain unsupported bins and paired cross covariance; one smoke contrast is not general convergence.',
}


def sha(path):
    """Return the SHA-256 of a regular local file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    """Describe first-party files with source docs or implementation-bound API notes.

    Reading notes validates syntax-tree bindings; it never executes their targets.
    Verification and optional tests remain an inventory, not execution evidence.
    """
    from scripts.build_reference import callable_notes
    files = {*ROOT.glob('*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py'),
             *(ROOT/'optional_tests').glob('*.py'), *(p for p in (ROOT/'core').rglob('*.py') if 'software' not in p.parts)}
    payloads={path:path.read_bytes() for path in files}
    texts={path:payload.decode("utf-8") for path,payload in payloads.items()}
    trees={path:ast.parse(text) for path,text in texts.items()}
    notes=callable_notes(trees)
    rows=[]
    for path in sorted(files):
        text=texts[path]; tree=trees[path]
        doc=ast.get_docstring(tree) or 'No module docstring; inspect the linked implementation and callable list.'
        calls=[]
        for n in tree.body:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and not n.name.startswith('_'):
                docstring=ast.get_docstring(n)
                reviewed=notes.get((path,n.name))
                calls.append({'name':n.name,'line':n.lineno,
                    'description':docstring or reviewed or 'No reviewed description yet; inspect the implementation.',
                    'description_source':'source_docstring' if docstring else 'reviewed_note' if reviewed else 'undocumented'})
        rows.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(payloads[path]).hexdigest(),'lines':len(text.splitlines()),
                     'description':doc,'functions':calls,'role':'verification' if 'tests' in path.parts or path.name=='verification.py' else 'application',
                     'status':'NOT_EXECUTED_BY_THIS_LAB'})
    return rows


def graph_inventory():
    """Join all 102 declared graph routes with portable retained examples, if present."""
    registry=json.loads((ROOT/'configs/research/graph_producers.json').read_text())
    entries=registry.get('entries',registry.get('producers',[]))
    if not entries:
        entries=next(value for value in registry.values() if isinstance(value,list))
    reference_path=HERE/'reference_gallery/catalogue.json'
    reference=json.loads(reference_path.read_text()) if reference_path.exists() else {'figures':[]}
    saved={(r['family'],r['variant']):r for r in reference['figures']}
    rows=[]
    for entry in entries:
        row=dict(entry);key=(row['family'],row['variant'])
        row['reference']=saved.get(key)
        row['reading_guide']=FAMILY_NOTES.get(row['family'],'Read the original numeric contract, units and masks.')
        rows.append(row)
    return rows


def build_catalogue():
    """Build UI data; inspection and archived evidence never become fresh test passes."""
    sources=source_inventory();graphs=graph_inventory()
    archived_trace = HERE / 'reference_gallery/regression-calls.json'
    if archived_trace.is_file():
        trace = json.loads(archived_trace.read_text())
        for row in sources:
            observed = trace['files'].get(row['path'])
            if observed:
                row['archived_regression'] = {
                    'calls': observed['calls'],
                    'tests': observed['tests'],
                    'scope': trace['scope'],
                    'original_sha256': trace['original_sha256'],
                    'execution_source_sha256': observed.get('execution_source_sha256'),
                    'current_source_matches': observed.get('execution_source_sha256') == row['sha256'] if observed.get('execution_source_sha256') else None,
                }
    indexed={row['path']:row for row in sources}
    layout=json.loads((ROOT/'configs/core_source_layout.json').read_text())['modules']
    for graph in graphs:
        graph['implementation_route']=[]
        producers = [(node, 'producer') for node in graph['producer_route']]
        for node, role in [*producers, (graph['renderer'], 'renderer')]:
            logical=node['module'].removeprefix('core.')
            path=layout[logical]['source'] if logical in layout else node['module'].replace('.','/')+'.py'
            source=indexed.get(path)
            if source:
                function=next((f for f in source['functions'] if f['name']==node['function']),None)
                graph['implementation_route'].append({'name':node['module']+'.'+node['function'],
                    'path':path,'line':function['line'] if function else 1,'role':role})
    return {'schema':'research2-optional-lab-catalogue-v1','sources':sources,'graphs':graphs,
            'equations':EQUATIONS,'counts':{'python_files':len(sources),'graph_recipes':len(graphs)},
            'software_reference':json.loads((HERE/'reference_gallery/software.json').read_text()) if (HERE/'reference_gallery/software.json').exists() else [],
            'reference_scope':'Archived calculations have their own producer scope. Run the lab to obtain fresh results on this machine.'}
