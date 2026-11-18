"""Small public command handlers, consolidated from root launch scripts.

Each handler imports only its selected domain. Scientific calculations stay
with their domain modules; this module owns argument parsing and forwarding.
"""
import argparse
import json
from pathlib import Path


def hmf_main():
    from core.hmf_campaign import hmf_main as run
    return run()


def hmf_learning_main():
    from core.hmf_learning import learning_main
    return learning_main()


def run_pipeline_main():
    """Run a fresh user-configured linear cosmology, then hand it to GADGET."""
    import json
    import os
    import sys
    from core import linear
    from core.resources import runtime_paths
    from core.native import native_engine_directory
    parser = argparse.ArgumentParser(description=run_pipeline_main.__doc__)
    parser.add_argument('--linear-config', type=Path, help='Your CLASS/AxiCLASS answers JSON; otherwise prompt')
    parser.add_argument('--answers', type=Path, help='Your GADGET answers JSON; cosmology must match the new bundle')
    parser.add_argument('--state-root', type=Path, help='External runtime root for every stage')
    parser.add_argument('--linear-output', type=Path, help='Parent for new named cosmology; defaults to runtime/runs/linear')
    parser.add_argument('--run-dir', type=Path, help='Explicit new particle-run directory')
    parser.add_argument('--binary', type=Path, help='Explicit provenance-bound compiled linear solver')
    parser.add_argument('--draft', type=Path, help='Saved terminal questionnaire; resume with the same path')
    parser.add_argument('--count', type=int, help='Number of simulations to configure and run sequentially')
    args = parser.parse_args()
    previous_root = os.environ.get('RESEARCH2_STATE_ROOT')
    previous_argv = sys.argv
    try:
        paths = runtime_paths(state_root=args.state_root)
        os.environ['RESEARCH2_STATE_ROOT'] = str(paths.state_root)
        count=args.count
        if count is None and not any((args.linear_config,args.answers,args.run_dir)):
            while True:
                answer=input('How many simulations? [1 for one run]: ').strip() or '1'
                if answer.isdigit() and int(answer)>0:
                    count=int(answer);break
                print('Enter a positive whole number.')
        count=1 if count is None else count
        if count<1:parser.error('--count must be positive')
        if count>1:
            if any((args.linear_config,args.answers,args.run_dir)):
                parser.error('Multiple runs are configured interactively; omit single-run config, answers and run-dir flags')
            from core.batch_simulation import configure_and_run_queue
            return configure_and_run_queue(count,paths,linear_output=args.linear_output,binary=args.binary)
        if args.linear_config:
            cfg = json.loads(args.linear_config.read_text())
        else:
            from core.setup_walkthrough import Questionnaire, draft_path
            print('\n[1/7] Cosmology and linear spectrum')
            cfg = Questionnaire(draft_path(args.draft), 'linear').run(linear.prompt_config)
        linear.validate(cfg)
        solver = args.binary.expanduser().resolve() if args.binary else (
            native_engine_directory('axiclass' if cfg['model']=='ede' else 'class')/'class')
        if not solver.is_file():
            raise FileNotFoundError('Compiled solver missing; run bash scripts/setup.sh first: ' + str(solver))
        output_root = args.linear_output.expanduser().resolve() if args.linear_output else paths.runs_root/'linear'
        from core.setup_walkthrough import draft_path
        from core.foundation import atomic_json, canonical_digest
        draft = draft_path(args.draft)
        saved_draft = json.loads(draft.read_text()) if draft.exists() and not args.linear_config else None
        binding = saved_draft.get('linear_bundle', {}) if saved_draft else {}
        expected_binding = {'configuration': canonical_digest(cfg), 'solver_sha256': linear.checksum(solver),
                            'output_root': str(output_root.resolve())}
        target_dir = output_root / cfg.get('name', '')
        if binding.get('inputs') == expected_binding:
            output = Path(binding['directory'])
            linear.verify_spectrum_directory(output)
            print('Reusing verified linear bundle from setup draft: '+str(output))
        elif target_dir.is_dir() and (target_dir / 'spectrum.json').is_file():
            try:
                linear.verify_spectrum_directory(target_dir)
                output = target_dir
                print('Reusing verified completed linear bundle: ' + str(output))
            except Exception:
                output = linear.run(cfg, output_root, solver)
        else:
            output = linear.run(cfg, output_root, solver)
            if saved_draft is not None:
                saved_draft['linear_bundle'] = {'inputs': expected_binding, 'directory': str(output.resolve())}
                atomic_json(draft, saved_draft)
        print('Linear bundle complete: ' + str(output), flush=True)
        from commands import run_gadget_main, load_json, spectrum_path
        if args.answers:
            saved = load_json(args.answers)
            if saved.get('cosmology') == 'REPLACED_BY_RUN_COMMAND':
                saved['cosmology'] = str((output/'spectrum.json').resolve())
                atomic_json(args.answers, saved)
            # Never silently attach particle settings authored for another cosmology.
            if 'cosmology' not in saved or spectrum_path(saved['cosmology']) != (output/'spectrum.json').resolve():
                raise ValueError('GADGET answers cosmology must name the newly generated spectrum.json: ' + str(output))
        sys.argv = ['commands.py gadget', '--cosmology', str(output)]
        if args.draft:
            sys.argv.extend(['--draft', str(args.draft)])
        if args.answers:
            sys.argv.extend(['--answers', str(args.answers)])
        if args.run_dir:
            sys.argv.extend(['--run-dir', str(args.run_dir)])
        return run_gadget_main()
    finally:
        sys.argv = previous_argv
        if previous_root is None:
            os.environ.pop('RESEARCH2_STATE_ROOT', None)
        else:
            os.environ['RESEARCH2_STATE_ROOT'] = previous_root

def audit_graph_producers_main():
    """Audit exact102 graph producer routes without running scientific producers."""
    parser = argparse.ArgumentParser(description=audit_graph_producers_main.__doc__)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from core.graph_producer_registry import audit_graph_producers
    from core.foundation import atomic_json
    result = audit_graph_producers(args.registry)
    atomic_json(args.output, result)
    print(json.dumps({'status': result['status'], 'variants': result['variant_count'], 'output': str(args.output.resolve())}))

def combine_graph_galleries_main():
    """Combine original measured graph galleries with all102 producer/status entries."""
    parser = argparse.ArgumentParser(description=combine_graph_galleries_main.__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from core.research.display import combine_graph_galleries
    result = combine_graph_galleries(json.loads(args.request.read_text()), args.output)
    print(json.dumps({'variants': result['variant_count'], 'measured_views': len(result['views']), 'output': str(args.output.resolve())}))

def export_hubble_figures_main():
    """Export registered Hubble figures from original measured/frozen products."""
    parser = argparse.ArgumentParser(description=export_hubble_figures_main.__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-gib', type=float, default=1.0)
    args = parser.parse_args()
    from core.expansion_products import export_hubble_figures
    print(json.dumps(export_hubble_figures(json.loads(args.request.read_text()), args.output, memory_gib=args.memory_gib), indent=2))

def export_inference_figures_main():
    """Export registered inference graphs from completed original producer products."""
    parser = argparse.ArgumentParser(description=export_inference_figures_main.__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-gib', type=float, default=1.0)
    args = parser.parse_args()
    from core.flagship_inference import export_inference_figures
    print(json.dumps(export_inference_figures(json.loads(args.request.read_text()), args.output, memory_gib=args.memory_gib), indent=2))

def export_abundance_figures_main():
    """Export native stellar-mass and Vmax abundance from a frozen input request."""
    p = argparse.ArgumentParser(description=export_abundance_figures_main.__doc__)
    p.add_argument('request', type=Path)
    p.add_argument('output_directory')
    p.add_argument('--kind', choices=('subhalo', 'host', 'matched'), default='subhalo')
    p.add_argument('--memory-gib', type=float, default=1.0)
    a = p.parse_args()
    from core.flagship_abundance import export_native_abundance, export_native_host_abundance, export_matched_host_abundance
    exporter = {'subhalo': export_native_abundance, 'host': export_native_host_abundance, 'matched': export_matched_host_abundance}[a.kind]
    r = exporter(json.loads(a.request.read_text()), a.output_directory, memory_gib=a.memory_gib)
    print('Rendered native abundance figures:', len(r['figures']))
    return 0

def export_assembly_figures_main():
    """Export registered assembly figures from a source-bound native ancestry stage."""
    p = argparse.ArgumentParser(description=export_assembly_figures_main.__doc__)
    p.add_argument('product')
    p.add_argument('identity_graph_directory')
    p.add_argument('output_directory')
    p.add_argument('--memory-gib', type=float, default=1.0)
    a = p.parse_args()
    from core.assembly import export_native_assembly_figures
    r = export_native_assembly_figures(a.product, a.identity_graph_directory, a.output_directory, memory_gib=a.memory_gib)
    print('Rendered native assembly figures:', len(r['figures']))
    return 0

def export_bh_figures_main():
    """Export registered growth, feedback and accretion-ratio figures from native BH histories."""
    p = argparse.ArgumentParser(description=export_bh_figures_main.__doc__)
    p.add_argument('native_bh_directory')
    p.add_argument('destination')
    p.add_argument('--run-directory')
    p.add_argument('--memory-gib', type=float, default=1.0)
    a = p.parse_args()
    from core.flagship_bh import export_native_bh_figures
    r = export_native_bh_figures(a.native_bh_directory, a.destination, run_directory=a.run_directory, memory_gib=a.memory_gib)
    print('Registered native BH figures:', len(r['figures']))
    return 0

def export_convergence_figures_main():
    """Export registered convergence figures from pinned native measurements."""
    parser = argparse.ArgumentParser(description=export_convergence_figures_main.__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--memory-gib', type=float, default=2.0)
    args = parser.parse_args()
    from core.flagship_products import export_convergence_figures
    report = export_convergence_figures(json.loads(args.request.read_text()), args.destination, request_base=args.request.resolve().parent, memory_gib=args.memory_gib)
    print('Registered native convergence figures:', len(report['figures']))

def export_splashback_figures_main():
    """Export registered splashback diagnostics from a verified native environment."""
    p = argparse.ArgumentParser(description=export_splashback_figures_main.__doc__)
    p.add_argument('product')
    p.add_argument('output_directory')
    p.add_argument('--memory-gib', type=float, default=1.0)
    a = p.parse_args()
    from core.halo_products import export_native_splashback_figures
    r = export_native_splashback_figures(a.product, a.output_directory, memory_gib=a.memory_gib)
    print('Rendered native splashback figures:', len(r['figures']))
    return 0

def export_stream_figures_main():
    """Export registered stream graphs from a sealed measured coordinate product."""
    parser = argparse.ArgumentParser(description=export_stream_figures_main.__doc__)
    parser.add_argument('product', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--memory-gib', type=float, default=2.0)
    args = parser.parse_args()
    from core.flagship_products import export_registered_stream_figures
    report = export_registered_stream_figures(args.product, args.destination, memory_gib=args.memory_gib)
    print('Registered measured stream figures:', len(report['figures']))

def export_structure_figures_main():
    """Export registered profiles from a verified native-halo-structure-v1 product."""
    p = argparse.ArgumentParser(description=export_structure_figures_main.__doc__)
    p.add_argument('product')
    p.add_argument('output_directory')
    p.add_argument('--memory-gib', type=float, default=1.0)
    a = p.parse_args()
    from core.halo_products import export_native_structure_figures
    result = export_native_structure_figures(a.product, a.output_directory, memory_gib=a.memory_gib)
    print('Rendered native structure figures:', len(result['figures']))
    return 0

def flagship_main():
    """Unified preparation, execution, recovery, analysis and rendering entrypoint."""
    from core.flagship_launch import main
    return main()

def ingest_cosmography_main():
    """Validate measured ladder/full-lens inputs without executing an inference fit."""
    parser = argparse.ArgumentParser(description=ingest_cosmography_main.__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    from core.cosmography import ingest_cosmography_manifest
    print(json.dumps(ingest_cosmography_manifest(json.loads(args.request.read_text()), args.destination), indent=2))

def prepare_analysis_figures_main():
    """Prepare original observational/inference campaign and registered graph stages."""
    parser = argparse.ArgumentParser(description=prepare_analysis_figures_main.__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    from core.analysis_figure_campaign import prepare_analysis_figure_campaign
    print(json.dumps(prepare_analysis_figure_campaign(json.loads(args.request.read_text()), args.destination), indent=2))

def bao_model_main():
    """Evaluate one frozen model against the original compressed BAO covariance."""
    parser = argparse.ArgumentParser(description=bao_model_main.__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-inference', action='store_true', help='Validate the model, explicit sampler settings and installed backend without sampling.')
    args = parser.parse_args()
    from core.bao import publish_conditional_bao, prepare_bao_posterior
    from core.foundation import atomic_json
    request = json.loads(args.request.read_text())
    if args.prepare_inference:
        _, record, sources = prepare_bao_posterior(request['model_request'], request['inference_settings'])
        if args.output.exists():
            raise FileExistsError('Keep previous inference preparation')
        atomic_json(args.output, {**record, 'source_hashes': sources})
    else:
        publish_conditional_bao(request, args.output)
    print(str(args.output.resolve()))

def flagship_products_main():
    """Native stream measurements, registered figures, Blender timeline and yields."""
    from core.flagship_products import flagship_products_main
    return flagship_products_main()

def music_boundary_main():
    """Build the native MUSIC boundary response adapter without running ICs."""
    parser=argparse.ArgumentParser(description=music_boundary_main.__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compiler')
    parser.add_argument('--include',type=Path,action='append',default=[])
    args=parser.parse_args()
    from core.music.boundary import prepare_boundary_adapter
    result=prepare_boundary_adapter(args.output,compiler=args.compiler,include_directories=args.include)
    print(json.dumps(result,indent=2))
    return 0

def music_capacity_main():
    """Plan native generation, response, coupled-solver and covariance resources."""
    parser=argparse.ArgumentParser(description=music_capacity_main.__doc__)
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from core.music.resources import publish_generation_capacity
    result=publish_generation_capacity(json.loads(args.request.read_text()),args.output)
    print(json.dumps(result,indent=2))
    return 2 if result['status']=='REQUIRES_LARGER_ALLOCATION' else 0

def music_flow_prepare_main():
    """Freeze native window launch settings in complete public evolution answers."""
    parser=argparse.ArgumentParser(description=music_flow_prepare_main.__doc__)
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from core.music.flow import prepare_original_flow_answers
    result=prepare_original_flow_answers(json.loads(args.request.read_text()),args.output)
    print(json.dumps(result,indent=2))
    return 0

def music_frozen_model_main():
    """Verify retained complete MUSIC fields against their original seed streams."""
    parser=argparse.ArgumentParser(description=music_frozen_model_main.__doc__)
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.music.phases import publish_frozen_noise_model
    result=publish_frozen_noise_model(json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 2

def music_phase_plan_main():
    """Preflight native tiled phase memory and cache storage on retained geometry."""
    from core.music.resources import phase_resources_main
    return phase_resources_main()

def music_seeded_model_main():
    """Verify fixed-prefix local MUSIC caches against their declared native seed streams."""
    parser=argparse.ArgumentParser(description=music_seeded_model_main.__doc__)
    parser.add_argument('request',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--memory-gib',type=float,default=1.)
    args=parser.parse_args()
    from core.music.phases import publish_seeded_noise_model
    result=publish_seeded_noise_model(json.loads(args.request.read_text()),args.output,memory_gib=args.memory_gib)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 2

def ingest_bao_main():
    """Publish a pinned measured BAO input without fitting cosmological parameters."""
    parser = argparse.ArgumentParser(description=ingest_bao_main.__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from core.bao import ingest_compressed_bao
    ingest_compressed_bao(json.loads(args.request.read_text()), args.output)
    print(str(args.output.resolve()))

def prepare_flagship_campaign_main():
    """Prepare selected native campaign recipes using the existing workflow engine."""
    parser = argparse.ArgumentParser(description=prepare_flagship_campaign_main.__doc__)
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    mode.add_argument('--matched', action='store_true', help='Combine prepared CDM/SIDM branches with a shared IC producer per target.')
    mode.add_argument('--production', action='store_true', help='Connect retained production bridge staging and prepared evolution settings.')
    mode.add_argument('--history', action='store_true', help='Extend a prepared native campaign with material recovery and actual epoch coverage.')
    args = parser.parse_args()
    from core.flagship_campaigns import prepare_native_campaign, prepare_matched_campaign
    from core.flagship_production_campaigns import prepare_production_campaign
    from core.flagship_history import prepare_history_campaign
    if args.history:
        prepare = prepare_history_campaign
    elif args.production:
        prepare = prepare_production_campaign
    elif args.matched:
        prepare = prepare_matched_campaign
    else:
        prepare = prepare_native_campaign
    print(json.dumps(prepare(json.loads(args.request.read_text()), args.output), indent=2))

def prepare_flagship_profile_main():
    """Freeze selected flagship profile settings after actual resource preflight."""
    parser = argparse.ArgumentParser(description=prepare_flagship_profile_main.__doc__)
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument('profile', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', dest='cases', help='Select a case ID; repeat for any number of cases. Default: all supplied cases.')
    parser.add_argument('--sidm-settings', type=Path, help='Prepare paired CDM/SIDM profiles using this explicit native SIDM configuration.')
    mode.add_argument('--freeze-matched', action='store_true', help='Read a prepared matched-pair manifest and freeze both branches with current resource checks.')
    mode.add_argument('--geometry-only', action='store_true', help='Estimate a pinned future production geometry; does not freeze launch settings or generate ICs.')
    mode.add_argument('--hardware-requirements', action='store_true', help='Prepare larger-hardware requirements while retaining the selected production targets and resolution.')
    mode.add_argument('--stage-production', action='store_true', help='Stage production inputs from the retained parent phase prefix; never generate production ICs.')
    mode.add_argument('--plan-production', action='store_true', help='Run current-generator mesh-only planning on completed staged production inputs.')
    mode.add_argument('--production-answers', action='store_true', help='Materialize complete native production evolution settings without launching.')
    parser.add_argument('--headroom-fraction', type=float, default=0.25, help='Provisioning headroom assumption for --hardware-requirements (default: 0.25).')
    args = parser.parse_args()
    selected_mode = any((args.freeze_matched, args.geometry_only, args.hardware_requirements, args.stage_production, args.plan_production, args.production_answers))
    if selected_mode and (args.cases is not None or args.sidm_settings is not None):
        parser.error('This mode consumes its own request; omit --case and --sidm-settings')
    if not args.hardware_requirements and args.headroom_fraction != 0.25:
        parser.error('--headroom-fraction applies only to --hardware-requirements')
    from core.flagship_profiles import freeze_profile, prepare_matched_sidm_profiles, freeze_matched_profiles, preflight_planned_geometry, publish_production_requirements, stage_production_inputs, plan_staged_production, prepare_production_evolution_settings
    if args.production_answers:
        print(json.dumps(prepare_production_evolution_settings(json.loads(args.profile.read_text()), args.output), indent=2))
        return
    if args.plan_production:
        result = plan_staged_production(json.loads(args.profile.read_text()), args.output)
        print(json.dumps(result, indent=2))
        if result['resource_status'] == 'FAIL':
            raise SystemExit(2)
        return
    if args.stage_production:
        print(json.dumps(stage_production_inputs(json.loads(args.profile.read_text()), args.output), indent=2))
        return
    if args.hardware_requirements:
        request = json.loads(args.profile.read_text())
        print(json.dumps(publish_production_requirements(request, args.output, headroom_fraction=args.headroom_fraction), indent=2))
        return
    if args.geometry_only:
        request = json.loads(args.profile.read_text())
        print(json.dumps(preflight_planned_geometry(request, args.output, output_directory=args.output.parent), indent=2))
        return
    if args.freeze_matched:
        print(json.dumps(freeze_matched_profiles(args.profile, args.output), indent=2))
        return
    request = json.loads(args.profile.read_text())
    if args.cases is not None:
        if len(set(args.cases)) != len(args.cases):
            parser.error('Selected case IDs must be distinct')
        available = {row['id']: row for row in request['zoom_cases']}
        if set(args.cases) - set(available):
            parser.error('Selected case is absent from the supplied profile')
        request['zoom_cases'] = [available[name] for name in args.cases]
    result = prepare_matched_sidm_profiles(request, args.sidm_settings, args.output) if args.sidm_settings else freeze_profile(request, args.output)
    print(json.dumps(result, indent=2))

def export_power_figures_main():
    """Export source-bound linear, evolved or matched matter-power products.

Default positional usage still consumes a sealed native analysis bundle.
Use --kind linear with spectrum.json, or --kind matched with a frozen
flagship-matched-power-v1 request containing measured joint covariance."""
    parser = argparse.ArgumentParser(description=export_power_figures_main.__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('--kind', choices=('linear', 'evolved', 'matched', 'ensemble'), default='evolved')
    parser.add_argument('--memory-gib', type=float, default=1.0)
    args = parser.parse_args()
    from core.power import export_linear_power, export_evolved_power, export_matched_power
    if args.kind == 'linear':
        result = export_linear_power(args.input, args.output_directory, memory_gib=args.memory_gib)
    elif args.kind == 'ensemble':
        from core.power import export_power_ensemble
        result = export_power_ensemble(json.loads(args.input.read_text()), args.output_directory, memory_gib=args.memory_gib)
    elif args.kind == 'matched':
        result = export_matched_power(json.loads(args.input.read_text()), args.output_directory, memory_gib=args.memory_gib)
    else:
        result = export_evolved_power(args.input, args.output_directory, memory_gib=args.memory_gib)
    print(json.dumps({'rendered_figures': len(result['figures']), 'science_acceptance': 'NOT_ACCEPTED'}))
    return 0

def export_sidm_figures_main():
    """Export native SIDM history, fixed-cohort rates and source-bound extra figures."""
    parser = argparse.ArgumentParser(description=export_sidm_figures_main.__doc__)
    parser.add_argument('operation', choices=['history', 'scattering', 'centroids', 'extended', 'matched'])
    parser.add_argument('input', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--run-directory', type=Path)
    parser.add_argument('--memory-gib', type=float, default=2.0)
    args = parser.parse_args()
    if args.operation in {'history', 'scattering'}:
        if args.run_directory is None:
            parser.error('--run-directory is required for native histories/rates')
        if args.operation == 'history':
            from core.sidm_products import export_native_sidm_history as function
        else:
            from core.sidm_products import export_native_scattering_rates as function
        result = function(args.input, args.destination, run_directory=args.run_directory, memory_gib=args.memory_gib)
    else:
        if args.operation == 'centroids':
            from core.sidm_products import export_native_sidm_centroids as function
        elif args.operation == 'extended':
            from core.sidm_products import export_native_sidm_extended as function
        else:
            from core.sidm_products import export_native_sidm_matched as function
        result = function(json.loads(args.input.read_text()), args.destination, memory_gib=args.memory_gib)
    print(json.dumps({'figure_count': len(result['figures']), 'variants': sorted({r['variant'] for r in result['figures']})}, indent=2))
    return 0

def render_scene_main():
    from core.scene_render import prepare_render_main
    return prepare_render_main()


def conditional_h0_main():
    """Fit conditional H0 to measured BAO with fixed expansion shape and drag scale."""
    parser=argparse.ArgumentParser(description=conditional_h0_main.__doc__)
    parser.add_argument('--bao-output',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--settings',required=True,type=Path,help='Explicit JSON: h0_bounds_km_s_mpc, sampler, histogram_bins, memory_gib')
    args=parser.parse_args()
    from core.bao import publish_conditional_h0
    settings=json.loads(args.settings.read_text())
    result=publish_conditional_h0(args.bao_output,args.output,settings=settings)
    print(json.dumps({'output':str(args.output.resolve()),'conditional_H0_median':result['median'],
        'equal_tail_68':result['equal_tail_68'],'units':result['parameter_units'],
        'conditioning':result['conditioning'],'science_acceptance':'NOT_ACCEPTED'},indent=2))


def export_analysis_figures_main():
    """Connect sealed native analysis datasets to registered scientific figures.

Each recipe names existing datasets and their units. Missing measurements remain
pending. Rendering does not establish physical resolution or convergence."""
    p = argparse.ArgumentParser(description=export_analysis_figures_main.__doc__)
    p.add_argument('analysis_bundle', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--memory-gib', type=float, default=1.0)
    p.add_argument('--dpi', type=int, default=160)
    a = p.parse_args()
    from core.figures import export_native_analysis_figures
    r = export_native_analysis_figures(a.analysis_bundle, a.destination, memory_gib=a.memory_gib, dpi=a.dpi)
    print('Rendered native registered figures:', len(r['figures']))


def artwork_main():
    """Process measured outputs from an existing saved scene campaign."""
    parser = argparse.ArgumentParser(description=artwork_main.__doc__)
    parser.add_argument('operation', choices=['parent-export', 'diagnostics', 'encode'])
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.operation == 'parent-export':
        from core.flagship_runtime_display import export_artwork_parent
        export_artwork_parent(run)
    elif args.operation == 'diagnostics':
        from core.flagship_orbits import native_orbit_diagnostics
        source = run/'native-diagnostics-settings.json'
        settings = json.loads(source.read_text())
        settings['memory_gib'] = json.loads((run/'artwork-profile.json').read_text())['template']['resources']['memory_gib']*.50
        native_orbit_diagnostics(run/'simulation', run/'native-orbit-diagnostics', settings)
    else:
        from core.flagship_runtime_display import encode_artwork_movies
        encode_artwork_movies(run)


def inspect_poisson_failure_main():
    """Inspect a saved native failed Poisson field without changing its source."""
    parser=argparse.ArgumentParser(description=inspect_poisson_failure_main.__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--order',type=int,choices=(2,4,6),required=True)
    parser.add_argument('--memory-gib',type=float,default=.25)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous Poisson failure inspection')
    from core.music.poisson import inspect_failure_snapshot
    from core.foundation import atomic_json
    result=inspect_failure_snapshot(args.snapshot,order=args.order,memory_gib=args.memory_gib)
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))
    return 0


def stream_products_main():
    import json
    from core.stream_common import json_safe
    parser=argparse.ArgumentParser(description='Export measured stream figures or Blender inspector assets.')
    parser.add_argument('kind',choices=('figures','inspector'))
    parser.add_argument('product',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--memory-gib',type=float,required=True)
    args=parser.parse_args()
    if args.kind=='figures':
        from core.figures import export_stream_figures as export
    else:
        from core.blender import export_stream_inspector as export
    result=export(args.product,args.output,memory_gib=args.memory_gib)
    print(json.dumps(json_safe(result),indent=2))
