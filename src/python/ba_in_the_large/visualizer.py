import matplotlib.pyplot as plt
import numpy as np

def plot_residuals(initial_residuals, final_residuals):
    """Plot initial and final residuals."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(initial_residuals)
    plt.title('Initial Residuals')
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(final_residuals)
    plt.title('Final Residuals')
    plt.grid(True)
    
    plt.tight_layout()
    return plt
    
def display_optimization_results(initial_params, final_params, elapsed_time, utils):
    """Display optimization results."""
    print("Optimization took {0:.0f} seconds".format(elapsed_time))
    
    print('Before:')
    print('cam0: {}'.format(utils.prettylist(initial_params[0:9])))
    print('cam1: {}'.format(utils.prettylist(initial_params[9:18])))
    
    print('After:')
    print('cam0: {}'.format(utils.prettylist(final_params[0:9])))
    print('cam1: {}'.format(utils.prettylist(final_params[9:18])))