"""Explicit grouped numerical prompts; no recipes or inferred parameter values."""
import math
from decimal import Decimal, InvalidOperation


def required_values(title, fields):
    """Read one comma-separated row, preserving integer precision and bounds.

    Fields: (key, description including units, int/float, minimum, maximum).
    Bounds may refer to earlier values through a callable of the parsed row.
    A bad row is retried in full; no partially accepted state escapes.
    """
    print("\n" + title)
    for key, label, kind, low, high in fields:
        print("  %s: %s%s" % (key, label, " (integer)" if kind is int else ""))
    keys = ", ".join(field[0] for field in fields)
    while True:
        raw = input(keys + " [comma-separated, in this order]: ").strip().split(',')
        result = {}
        try:
            if len(raw) != len(fields):
                raise ValueError("Enter exactly %d comma-separated values" % len(fields))
            for token, (key, label, kind, low, high) in zip(raw, fields):
                token = token.strip()
                lower = low(result) if callable(low) else low
                upper = high(result) if callable(high) else high
                if kind is int:
                    number = Decimal(token)
                    if not number.is_finite() or number != number.to_integral_value() or not lower <= number <= upper:
                        raise ValueError("%s must be an exact integer from %s to %s" % (key, lower, upper))
                    value = int(number)
                else:
                    value = float(token)
                    if not math.isfinite(value) or not lower <= value <= upper:
                        raise ValueError("%s must be finite and between %s and %s" % (key, lower, upper))
                result[key] = value
            print("  Accepted: " + ", ".join("%s=%s" % pair for pair in result.items()))
            return result
        except (ValueError, InvalidOperation, OverflowError) as error:
            print(str(error) + "; re-enter this row.")
