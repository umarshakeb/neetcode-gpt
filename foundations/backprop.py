import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x,w) + b
        y_hat = (1/(1+np.exp(-z)))
        dL_dy_hat = y_hat - y_true

        # Derivative of sigmoid: dy_hat/dz = y_hat * (1 - y_hat)
        dy_hat_dz = y_hat * (1.0 - y_hat)

        # Delta = dL/dz = dL/dy_hat * dy_hat/dz
        delta = dL_dy_hat * dy_hat_dz

        # Gradients: dL/dw = delta * x, dL/db = delta
        dL_dw = np.round(delta * x, 5)
        dL_db = round(float(delta), 5)
        dL_db = round(float(delta), 5)
        dL_db = round(float(delta), 5)
        return np.round(dL_dw,5), np.round(dL_db,5)

