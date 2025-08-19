"""Utility modules for precision background removal."""

from .mac_optimization import (
    detect_mac_architecture,
    initialize_mac_optimizations,
    get_optimal_device,
    optimize_for_mac_processing,
)

__all__ = [
    'detect_mac_architecture',
    'initialize_mac_optimizations',
    'get_optimal_device',
    'optimize_for_mac_processing',
]