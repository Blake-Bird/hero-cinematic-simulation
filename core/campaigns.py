"""Prepare native campaign recipes through one command, without evolving runs.

Usage: python commands.py native-prepare DOMAIN REQUEST DESTINATION
The module entry point, python -m core.campaigns, accepts the same arguments.
Only the selected domain's implementation is imported.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    domains = parser.add_subparsers(dest='domain', required=True)
    for name in (
        'assembly', 'baryons', 'black-holes', 'convergence', 'encounters',
        'halos', 'power', 'response-branch', 'sbi', 'sidm', 'web',
        'recovered-sidm', 'joint-ics',
    ):
        command = domains.add_parser(name, help='prepare ' + name + ' recipes')
        command.add_argument('request', type=Path, help='JSON campaign request')
        command.add_argument('destination', type=Path, help='new recipe directory')

    responses = domains.add_parser('responses', help='connect intervention designs to response estimators')
    responses.add_argument('design', type=Path)
    responses.add_argument('template', type=Path)
    responses.add_argument('destination', type=Path)

    joint = domains.add_parser('joint-response', help='bind completed IC generations to a response design')
    joint.add_argument('design', type=Path)
    joint.add_argument('--ic-manifest', type=Path, required=True)
    joint.add_argument('--ic-execution', type=Path, required=True)
    joint.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    # These two routes consume paths rather than the common request mapping.
    if args.domain == 'responses':
        from .response_campaign import prepare_native_response_campaign
        from .foundation import file_sha256
        result = prepare_native_response_campaign(
            args.design,
            {'path': str(args.template.resolve()), 'sha256': file_sha256(args.template)},
            args.destination,
        )
    elif args.domain == 'joint-response':
        from core.response_campaign import prepare_response_from_generated_ics
        design = args.design.resolve()
        result = prepare_response_from_generated_ics(
            json.loads(design.read_text()), args.ic_manifest, args.ic_execution,
            args.output, request_base=design.parent,
        )
    else:
        request = json.loads(args.request.read_text())
        match args.domain:
            case 'assembly':
                from .halo_campaign import prepare_native_assembly_campaign as prepare
            case 'baryons':
                from .native_baryon_campaign import prepare_native_baryon_campaign as prepare
            case 'black-holes':
                from .native_bh_campaign import prepare_native_bh_campaign as prepare
            case 'convergence':
                from .native_convergence_campaign import prepare_native_convergence as prepare
            case 'encounters':
                from .stream.native_encounters import prepare_native_encounter_campaign as prepare
            case 'halos':
                from .halo_campaign import prepare_native_halo_campaign as prepare
            case 'power':
                from .power import prepare_power_campaign as prepare
            case 'response-branch':
                from .response_campaign import prepare_native_response_branch as prepare
            case 'sbi':
                from core.stream.sbi import prepare_native_sbi_branch as prepare
            case 'sidm':
                from .sidm_campaign import prepare_native_sidm_campaign as prepare
            case 'web':
                from .halo_campaign import prepare_native_web_campaign as prepare
            case 'recovered-sidm':
                from .sidm_recovery import prepare_recovered_sidm_campaign as prepare
            case 'joint-ics':
                from core.response_campaign import prepare_joint_ic_generation as prepare
        if args.domain == 'sidm':
            # SIDM requests resolve relative assets beside the original request,
            # not relative to the shell's current working directory.
            result = prepare(request, args.destination, request_base=args.request.resolve().parent)
        else:
            result = prepare(request, args.destination)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
