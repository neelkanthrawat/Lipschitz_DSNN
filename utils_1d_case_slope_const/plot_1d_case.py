# code to visualise 1d cases
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, laplace

# list of functions
# 1. plot multiple histograms
# 2. plot splines (with possibility of along annotation)
# 3. 
#

# 1. plot multiple histogram
def plot_multiple_histogram(list_of_inputs, labels_list):
    # add a code line here: if the inputs are not numpy, convert them to numpy
    plt.figure(figsize=(5, 5))

    for i in range(0,len(list_of_inputs)):
        # Histogram for the code samples
        plt.hist(list_of_inputs[i], bins=50, alpha=0.4, 
                density=True, label=labels_list[i])
        plt.legend()

# 2. plot splines with annotations
def plot_with_annotations(x, y,label="with const coeffs",
                title="Plot with (x, y) Annotations", 
            xlabel="X-axis", ylabel="Y-axis",
            txt_color="red", annotate=1, style="-o",
            **kwargs):
    """
    Plot x and y values with (x, y) annotations.

    Args:
    - x (array-like): Array of x-values.
    - y (array-like): Array of y-values.
    - title (str): Title of the plot.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.

    Returns:
    - None
    """
    plt.plot(x, y, style,label=label)  # Plot the points with markers and lines

    # Annotate each point with its (x, y) coordinates
    if annotate:
        for x_val, y_val in zip(x, y):
            plt.annotate(
                f"({x_val:.2f}, {y_val:.2f})",  # Format to 2 decimal places
                (x_val, y_val),  # The point to annotate
                textcoords="offset points",  # Offset the text slightly
                xytext=(5, 5),  # Offset (5, 5) pixels
                fontsize=8,  # Font size of the annotation
                color=txt_color # Optional: text color
            )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(which='both')  # Optional: Add a grid
    plt.minorticks_on()

    # # Define x values
    ## Note FOR ME: I NEED TO FIGURE OUT WHAT IS MU AND B IN THE NOTEBOOK. 
    # I HAD DEFINED MU AND B AS GLOBAL PARAMTERS, = MEAN AND SCALE
    x = np.linspace(-40, 40, 1000)

    # Check if both mu and b are provided in kwargs
    if 'mu' in kwargs and 'b' in kwargs:
        # Extract mu and b from kwargs with default values
        mu = kwargs.get('mu', 0)  # Default mu = 0
        b = kwargs.get('b', 1)    # Default b = 1

        # Compute Laplace CDF
        F_X = np.where(x < mu, 0.5 * np.exp((x - mu) / b), 1 - 0.5 * np.exp(-(x - mu) / b))

        # Compute the transformation function T(x)
        T_x = norm.ppf(F_X)

        # Plot T(x)
        plt.plot(x, T_x, label=r'$T(x) = \Phi^{-1}(F_X(x))$', color='b')

    # Bold major grid lines
    plt.grid(which='major', linestyle='-', linewidth=1.5, alpha=0.9, color='black')

    # Light minor grid lines
    plt.grid(which='minor', linestyle=':', linewidth=0.7, alpha=0.5, color='gray')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()

