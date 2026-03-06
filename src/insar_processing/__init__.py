"""
InSAR processing package.

This subpackage contains:

- Low-level I/O utilities for SAR, interferograms, coherence maps, and DEMs.
- Baseline InSAR processing helpers (wrappers around external tools or existing products).
- Dataset preparation utilities for training learning-based models.
"""

# Submodules are imported lazily to avoid hard dependencies at package load time.
# Import them explicitly when needed, e.g.:
#   from src.insar_processing import io, baseline, pair_graph

