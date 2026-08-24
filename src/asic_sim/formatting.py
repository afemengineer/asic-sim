from __future__ import annotations


def fmt_bytes(value: float) -> str:
    units = ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB"))
    for threshold, suffix in units:
        if abs(value) >= threshold:
            return f"{value / threshold:,.3f} {suffix}"
    return f"{value:,.0f} B"


def fmt_time_s(value: float | None) -> str:
    if value is None:
        return "n/a"
    units = ((1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns"))
    for threshold, suffix in units:
        if value >= threshold:
            return f"{value / threshold:,.3f} {suffix}"
    return f"{value * 1e12:,.3f} ps"


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.1f} tok/s"
