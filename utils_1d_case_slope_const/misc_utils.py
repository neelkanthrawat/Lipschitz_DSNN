# misc utility function
# list of function
# 1. Laplace to normal
# 2. plot distributions
# 3. plot the transformation  which maps laplace to standard normal
#
#
##
#

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt



def transformation_laplace_to_std_normal(mu=0, b=1, x_range=(-10,10), num_points=41):
    # Define x values
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Compute Laplace CDF
    F_X = np.where(x < mu, 0.5 * np.exp((x - mu) / b), 1 - 0.5 * np.exp(-(x - mu) / b))

    # Compute the transformation function T(x)
    T_x = norm.ppf(F_X)

    return x,T_x

def plot_transformation(mu=0, b=1, x_range=(-5, 5), num_points=1000):
    """
    Plots the transformation function T(x) = Φ⁻¹(F_X(x)) that maps a 
    Laplace(mu, b) distributed variable to a standard normal.
    
    Args: 
    - mu: Mean of the Laplace distribution.
    - b: Scale parameter of the Laplace distribution.
    - x_range: Tuple defining the range of x values for the plot.
    - num_points: Number of points in the plot.
    
    Returns:
    - A plot of T(x) = Φ⁻¹(F_X(x)).
    """
    x,T_x = transformation_laplace_to_std_normal(mu=mu, b=b, x_range=x_range, num_points=num_points)

    # Plot T(x)
    plt.figure(figsize=(7, 5))
    plt.plot(x, T_x, label=r'$T(x) = \Phi^{-1}(F_X(x))$', color='b')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("$x$")
    plt.ylabel("$T(x)$")
    plt.title(f"Transformation Function from Laplace({mu}, {b}) to Standard Normal")
    plt.legend()
    plt.grid(which='both')  # Optional: Add a grid
    plt.minorticks_on()

    # Bold major grid lines
    plt.grid(which='major', linestyle='-', linewidth=1.5, alpha=0.9, color='black')

    # Light minor grid lines
    plt.grid(which='minor', linestyle=':', linewidth=0.7, alpha=0.5, color='gray')
    plt.show()