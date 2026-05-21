class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        func = init
        while iterations!=0:
            der_func = 2*func
            result = func - learning_rate*der_func
            func = result
            iterations -= 1
        return round(func,5)
