# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import typing

# Use Seaborn data 
data = sns.load_dataset("tips")


def error_calcs(x_points: pd.Series, y_points: pd.Series, m: float, b: float, num_points: int) -> float:
    """
	Calculates Error (Least Squares Method)
	:param x_points: Vector of independent variable
	:param y_points: Vector of dependent variable
	:param m: Slope
	:param b: Y-Intercept
	:param num_points: Total number of points
	:return: Sum of squared error
	"""

    error_sum = 0

    for i in range(num_points):
        error_sum += (y_points[i] - m * x_points[i] - b) ** 2

    return float(error_sum / num_points)


def gradient_step(x_points: pd.Series, y_points: pd.Series, m_current: float, b_current: float,
                  num_points: int, learn_rate: float) -> typing.Tuple[float, float]:
    """
	Calculates the step needed to take for gradient descent
	:param x_points: Vector of independent variable
	:param y_points: Vector of dependent variable
	:param m_current: Current Slope
	:param b_current: Current Y-Intercept
	:param num_points: Total number of points
	:param learn_rate: Learning rate of the function
	:return: New slope (m) and intercept (b) in direction of steepest descent
	"""

    grad_m = 0  # Initialize gradient of m

    grad_b = 0  # Initialize gradient of b

    # Calculate and sum up gradients
    for i in range(num_points):
        grad_m += -(2 / num_points) * x_points[i] * (y_points[i] - m_current * x_points[i] - b_current)
        grad_b += -(2 / num_points) * (y_points[i] - m_current * x_points[i] - b_current)

    # Calculate new m and b
    m_new = m_current - (learn_rate * grad_m)
    b_new = b_current - (learn_rate * grad_b)

    return m_new, b_new


def grad_descent()