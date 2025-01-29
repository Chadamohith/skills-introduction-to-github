import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Expanded sample data points based on visual estimation from the graph
tractor_age = np.array([5, 8, 10, 12, 15, 18, 20, 23, 25])
maintenance_cost = np.array([800, 1100, 1400, 1550, 1600, 1450, 1300, 900, 700])

# Perform polynomial regression (degree 2)
coefficients = Polynomial.fit(tractor_age, maintenance_cost, deg=2).convert().coef

# Generate the regression line
x_fit = np.linspace(5, 25, 100)
y_fit = coefficients[0] + coefficients[1] * x_fit + coefficients[2] * x_fit**2

# Create the scatter plot and regression line
plt.figure(figsize=(8, 6))
plt.scatter(tractor_age, maintenance_cost, color='black', label='Data Points')
plt.plot(x_fit, y_fit, color='blue', label='Regression Line', linewidth=2)

# Add labels, title, and legend
plt.xlabel("Tractor Age (years)", fontsize=12)
plt.ylabel("Maintenance Cost (USD)", fontsize=12)
plt.title("Tractor Age vs Maintenance Cost", fontsize=14)
plt.legend(fontsize=12)

# Customize grid and ticks
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()
