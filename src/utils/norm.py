import numpy as np
import math

class Norm:
    def pdf(self, x):
        """Probability density function of a standard normal distribution."""
        return (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)

    def cdf(self, x):
        """Cumulative distribution function of a standard normal distribution."""
        # Use the error function (erf) to calculate the CDF
        return 0.5 * (1 + self.erf(x / math.sqrt(2)))

    def erf(self, x):
        """Approximation of the error function (erf) using a numerical approximation."""
        # Approximation constants
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        
        # Save the sign of x
        sign = np.where(x >= 0, 1, -1)
        x = np.abs(x)

        # Calculate the approximation of erf
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        
        return sign * y
