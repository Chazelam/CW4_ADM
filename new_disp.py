import numpy as np
import matplotlib.pyplot as plt

# Constants
SOURCE_EMISSION_RATE = 10  # source emission rate (kg/s)
WIND_SPEED = 3  # wind speed (m/s)
RELEASE_HEIGHT = 0  # release height (m)
MIN_CONCENTRATION = 7.72850010233e-6  # minimum concentration threshold
SOURCE_POSITION = (500, 0, 0)  # (x0, y0, z0)

# Grid parameters
DOMAIN_SIZE_X = 4000  # size of domain in x (m)
DOMAIN_SIZE_Y = 2000  # size of domain in y (m)
NUM_POINTS = 2000  # number of plotting points

# Dispersion coefficients for atmospheric stability classes
DISPERSION_COEFFICIENTS = {
    "B": {"sigma_y": (0.16, 0.0001), "sigma_z": (0.12, 0.0)},
    "C": {"sigma_y": (0.11, 0.0001), "sigma_z": (0.08, 0.0002)},
    "D": {"sigma_y": (0.08, 0.0001), "sigma_z": (0.06, 0.0015)},
}

def determine_atmospheric_stability(wind_speed: float) -> str:
    """
    Determine the atmospheric stability class based on wind speed.
    
    :param wind_speed: Wind speed in m/s.
    :return: Atmospheric stability class ("B", "C", or "D").
    """
    if wind_speed < 2:
        return "B"
    elif wind_speed < 5:
        return "C"
    else:
        return "D"

def calculate_plume_dispersion(distance: float, stability_class: str) -> tuple[float, float]:
    """
    Calculate horizontal (sigma_y) and vertical (sigma_z) dispersion coefficients.
    
    :param distance: Distance from the source in meters.
    :param stability_class: Atmospheric stability class ("B", "C", or "D").
    :return: Tuple of (sigma_y, sigma_z) dispersion coefficients.
    """
    coeff = DISPERSION_COEFFICIENTS.get(stability_class)
    if not coeff:
        raise ValueError(f"Unknown stability class: {stability_class}")
    
    a_y, b_y = coeff["sigma_y"]
    a_z, b_z = coeff["sigma_z"]
    
    sigma_y = a_y * distance / np.sqrt(1 + b_y * distance)
    sigma_z = a_z * distance / np.sqrt(1 + b_z * distance)
    
    return sigma_y, sigma_z

def calculate_concentration(x: float, y: float, z: float, stability_class: str) -> float:
    """
    Calculate the concentration of the pollutant using the Gaussian plume model.
    
    :param x: Distance downwind from the source (m).
    :param y: Crosswind distance from the centerline (m).
    :param z: Vertical distance from the ground (m).
    :param stability_class: Atmospheric stability class ("B", "C", or "D").
    :return: Concentration of the pollutant.
    """
    sigma_y, sigma_z = calculate_plume_dispersion(x, stability_class)
    
    term1 = SOURCE_EMISSION_RATE / (2 * np.pi * WIND_SPEED * sigma_y * sigma_z)
    term2 = np.exp(-((y - SOURCE_POSITION[1]) ** 2) / (2 * sigma_y ** 2))
    term3 = np.exp(-((z - RELEASE_HEIGHT) ** 2) / (2 * sigma_z ** 2)) + np.exp(-((z + RELEASE_HEIGHT) ** 2) / (2 * sigma_z ** 2))
    
    return term1 * term2 * term3

# Main script
stability_class = determine_atmospheric_stability(WIND_SPEED)

# Create grid
x_grid, y_grid = np.meshgrid(
    np.linspace(0.005, DOMAIN_SIZE_X, NUM_POINTS),
    np.linspace(-DOMAIN_SIZE_Y, DOMAIN_SIZE_Y, 2 * NUM_POINTS)
)

# Calculate concentration
concentration = calculate_concentration(x_grid, y_grid, 1, stability_class)

# Plotting
levels = list(np.geomspace(MIN_CONCENTRATION, 1, 20))
plt.contourf(x_grid, y_grid, concentration, levels, colors=['#808080', '#A0A0A0', '#C0C0C0'])
plt.axis('off')
plt.savefig('plume_dispersion.jpg', dpi=300, bbox_inches='tight', facecolor="black")
plt.axis('on')
cbar = plt.colorbar()
cbar.ax.set_yscale('log')
plt.show()

# Debugging output
print("Concentration shape:", concentration.shape)
print("Concentration at (2000, 1750):", concentration[2000][1750])
print("Concentration at (1000, 0):", concentration[1000][0])