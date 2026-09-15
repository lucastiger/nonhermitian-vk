"""nhvk -- reproducibility package for the non-Hermitian Vakhitov--Kolokolov paper.

*The Vakhitov--Kolokolov criterion beyond Hermiticity: a biorthogonal,
conditioning-corrected stability theory for non-Hermitian solitons.*

The package exposes the shared machinery (:mod:`nhvk.core`) and the three
potential/gain-loss profiles used in the paper (:mod:`nhvk.profiles`).  The
executable entry points live in ``scripts/``.
"""

from __future__ import annotations

from .core import (
    Model,
    S_eigenvalue,
    S_map,
    adj_kernel,
    gaugefix,
    ip,
    ip_cp,
    make_grid,
    pt_gauge,
    real_ray,
    reflect,
    transport,
)
from .profiles import pt_barrier, pt_well, wadati_gaussian

__version__ = "1.0.0"

__all__ = [
    "Model",
    "S_eigenvalue",
    "S_map",
    "adj_kernel",
    "gaugefix",
    "ip",
    "ip_cp",
    "make_grid",
    "pt_gauge",
    "real_ray",
    "reflect",
    "transport",
    "wadati_gaussian",
    "pt_well",
    "pt_barrier",
    "__version__",
]
