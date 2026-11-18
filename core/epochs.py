"""Saved output schedules and display highlights; neither changes native names."""
from decimal import Decimal, InvalidOperation
import math


def parse_schedule(spec, start_z):
    """Return descending epochs and starred subset, preserving exact range endpoints."""
    start = Decimal(str(start_z))
    if not start.is_finite() or start < 0:
        raise ValueError('Starting redshift must be finite and nonnegative')
    values, highlights = [], []
    for raw in str(spec).split(','):
        token = raw.strip()
        starred = token.endswith('*')
        if starred:
            token = token[:-1].strip()
        fields = token.split(':')
        try:
            if len(fields) == 1:
                value = Decimal(fields[0])
                if not value.is_finite():
                    raise ValueError('Redshifts must be finite')
                values.append(value)
                if starred:
                    highlights.append(value)
            elif len(fields) == 3 and not starred:
                first, last, step = map(Decimal, fields)
                if not all(v.is_finite() for v in (first, last, step)):
                    raise ValueError('Range endpoints and steps must be finite')
                if not first >= last >= 0 or step <= 0:
                    raise ValueError('Range must have first >= last >= 0 and step > 0')
                count = (first-last)/step
                if count != count.to_integral_value() or count > 1099:
                    raise ValueError('Range must exactly reach its endpoint within 1100 snapshots')
                values.extend(first-i*step for i in range(int(count)+1))
            else:
                raise ValueError('Use z, z*, or first:last:step; star individual epochs')
        except InvalidOperation as error:
            raise ValueError('Invalid redshift entry: '+raw) from error
    values = sorted(set(values), reverse=True)
    if not values or len(values) > 1100 or values[0] > start or values[-1] < 0:
        raise ValueError('Snapshot redshifts must lie between z_start and z=0, with at most 1100 outputs')
    return [float(v) for v in values], [float(v) for v in sorted(set(highlights), reverse=True)]


def highlight_redshifts(spec, start_z, extra=()):
    """Validate that every explicit highlight belongs to the requested schedule."""
    schedule, starred = parse_schedule(spec, start_z)
    if isinstance(extra, str):
        extra = [] if extra.strip().lower() in ('', 'none') else parse_schedule(extra, start_z)[0]
    if not isinstance(extra, (list, tuple)):
        raise ValueError('highlight_redshifts must be a list of saved redshifts')
    selected = set(starred)
    for value in extra:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value not in schedule:
            raise ValueError('Each highlighted redshift must belong to the saved schedule')
        selected.add(float(value))
    return sorted(selected, reverse=True)


def epoch_label(snapshot, redshift):
    """Snapshot ID makes rounded display labels unique; metadata retains precision."""
    if type(snapshot) is not int or snapshot < 0 or not math.isfinite(float(redshift)):
        raise ValueError('Finite redshift and nonnegative snapshot ID required')
    return f'snap_{snapshot:03d}_z_{float(redshift):.3f}'
