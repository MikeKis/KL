# Spec: 2026-09-13_layerwise-from-colanet.md
"""Parse ArNIGPU ``-Pt`` text protocols (``spikes.<id>.txt``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ann_graph import ConverterError


def spike_counts_from_protocol(
    path: Path | str,
    *,
    n_images: int,
    period: int,
    n_neurons: int,
) -> np.ndarray:
    """Sum ``@`` per neuron over each presentation.

    One line is one tact: ``.`` silent, ``@`` spike, one character per neuron in
    general-directory order. A step-1 probe net has only the new TinyfromANN
    section, created in ``spatial_index`` order, so column ``i`` is that index.
    """
    n_images = int(n_images)
    period = int(period)
    n_neurons = int(n_neurons)
    if n_images <= 0 or period <= 0 or n_neurons <= 0:
        raise ConverterError("spike protocol shape must be positive")
    counts = np.zeros((n_images, n_neurons), dtype=np.int32)
    need = n_images * period
    got = 0
    at = ord("@")
    dot = ord(".")
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            if got >= need:
                break
            if line.endswith("\n"):
                line = line[:-1]
            if line.endswith("\r"):
                line = line[:-1]
            if len(line) != n_neurons:
                raise ConverterError(
                    f"spike protocol line {got} has {len(line)} neurons, expected {n_neurons}"
                )
            raw = np.frombuffer(line.encode("ascii"), dtype=np.uint8)
            if not np.all((raw == at) | (raw == dot)):
                raise ConverterError(f"spike protocol line {got} has a character other than '.' or '@'")
            counts[got // period] += (raw == at).astype(np.int32)
            got += 1
    if got != need:
        raise ConverterError(f"spike protocol has {got} tacts, expected {need}")
    return counts
