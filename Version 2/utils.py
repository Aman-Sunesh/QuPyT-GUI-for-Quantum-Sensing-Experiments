# utils.py
import numpy as np

def lorentzian(x, x0, gamma, A, y0):
    """Basic Lorentzian lineshape."""
    return y0 + A * (gamma**2 / ((x - x0)**2 + gamma**2))

def gaussian(x, mu, sigma, A, y0):
    """Basic Gaussian lineshape."""
    return y0 + A * np.exp(-((x - mu)**2) / (2*sigma**2))