"""Human-readable recovery commands derived from a workflow's explicit requests."""
import shlex


def attach_retry_commands(status):
    """Attach commands without promising reuse of immutable published outputs."""
    from core.setup_walkthrough import REQUESTS
    selected = status.get('products', [])
    unavailable = status.get('unavailable', {})
    commands = []
    for product in selected:
        if product not in unavailable and 'scientific_analysis' not in unavailable:
            continue
        cmd = 'python commands.py analysis '+shlex.quote(status['run'])+' --graphs '+shlex.quote(product)
        if product in REQUESTS:
            key, flag = REQUESTS[product]
            cmd += ' '+flag+' YOUR_REQUEST.json'
        commands.append({'product': product, 'command': cmd,
                         'note': 'Resolve the reported cause first; choose fresh destinations for immutable request outputs.'})
    for product, reason in status.get('applicability', {}).get('missing_inputs', {}).items():
        if product not in selected:
            cmd = 'python commands.py analysis '+shlex.quote(status['run'])+' --graphs '+shlex.quote(product)
            if product in REQUESTS:
                cmd += ' '+REQUESTS[product][1]+' YOUR_REQUEST.json'
            commands.append({'product': product, 'command': cmd, 'note': str(reason)})
    for product in status.get('deferred_outputs', []):
        if product in unavailable:
            flag = REQUESTS[product][1]
            commands.append({'product': product, 'command': 'python commands.py analysis '+shlex.quote(status['run'])+' --graphs '+product+' '+flag+' YOUR_REQUEST.json',
                             'note': unavailable[product]})
    status['retry_commands'] = commands
    states = {}
    for product, decision in status.get('applicability', {}).get('decisions', {}).items():
        states[product] = {'status': decision.get('status', 'UNKNOWN'), 'reason': decision.get('reason')}
    for product in selected:
        failed = product in unavailable or ('scientific_analysis' in unavailable and product not in status.get('artifacts', {}))
        states[product] = {'status': 'FAILED' if failed else ('COMPLETE' if status.get('status') in ('COMPLETE', 'INCOMPLETE') else 'UNKNOWN'),
                           'reason': unavailable.get(product) or (unavailable.get('scientific_analysis') if failed else None)}
    # A core suite can contain unavailable measurements at only some epochs.
    from pathlib import Path
    import json
    directory = status.get('artifacts', {}).get('analysis')
    if isinstance(directory, (str, Path)):
        summary = Path(directory)/'analysis_products_summary.json'
        if summary.is_file():
            try:
                payload = json.loads(summary.read_text())
                for product in payload.get('products', []):
                    outcomes = [row.get('products', {}).get(product, {}).get('status') for row in payload.get('snapshots', {}).values()]
                    outcomes = [value for value in outcomes if value]
                    if outcomes:
                        states[product] = {'status': 'COMPLETE' if all(value == 'COMPLETE' for value in outcomes) else 'INCOMPLETE',
                                           'epoch_statuses': outcomes}
            except (OSError, ValueError, TypeError):
                pass
    status['product_states'] = states
