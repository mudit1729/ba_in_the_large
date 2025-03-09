from .ba_solver import solve_bundle_adjustment
from .utils import read_bal_data, prettylist
from .visualizer import plot_residuals, display_optimization_results, visualize_reconstruction

# Import Plotly visualizer if available
try:
    from .plotly_visualizer import visualize_reconstruction_plotly, plot_residuals_plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

__all__ = [
    'solve_bundle_adjustment',
    'read_bal_data',
    'prettylist',
    'plot_residuals',
    'display_optimization_results',
    'visualize_reconstruction'
]

# Add Plotly visualizers to exports if available
if PLOTLY_AVAILABLE:
    __all__.extend([
        'visualize_reconstruction_plotly',
        'plot_residuals_plotly',
        'PLOTLY_AVAILABLE'
    ])
else:
    __all__.append('PLOTLY_AVAILABLE')