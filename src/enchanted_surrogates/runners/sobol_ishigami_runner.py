import os
import sys
import numbers
from collections.abc import Iterable
from datetime import datetime
from time import sleep
import numpy as np
from math import sin, pi

# --- MOCK DEPENDENCIES for standalone execution ---
class Runner:
    """Mock Runner class for demonstration."""
    def __init__(self, *args, **kwargs):
        pass
    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        raise NotImplementedError

def is_package_available(pkg):
    return False

# --- END MOCK DEPENDENCIES ---


def is_number(x):
    return isinstance(x, numbers.Number)


def is_iterable(x, *, treat_strings_as_iterable=True):
    if not isinstance(x, Iterable):
        return False
    if not treat_strings_as_iterable and isinstance(x, (str, bytes, bytearray)):
        return False
    return True


# --- Sobol G-function and Ishigami Function Implementations (Unchanged) ---

def sobol_g_function(x, a):
    """
    Computes the Sobol G-function: $f(\mathbf{x}) = \prod_{i=1}^d \frac{|4x_i - 2| + a_i}{1 + a_i}$
    where $x_i \in [0, 1]$.
    ...
    """
    x = np.asarray(x)
    a = np.asarray(a)

    if x.shape != a.shape:
        raise ValueError(f"Input vector 'x' (dim={x.size}) and parameter vector 'a' (dim={a.size}) must have the same dimension.")
    if not np.all((x >= 0) & (x <= 1)):
        raise ValueError("All input variables $x_i$ must be in [0, 1].")

    result = np.prod((np.abs(4 * x - 2) + a) / (1 + a))
    return float(result)


def ishigami_function(x, a=7.0, b=0.1):
    """
    Computes the 3-dimensional Ishigami function: $f(\mathbf{x}) = \sin(x_1) + a \sin^2(x_2) + b x_3^4 \sin(x_1)$
    ...
    """
    x = np.asarray(x)
    if x.size != 3:
        raise ValueError(f"Ishigami function requires exactly 3 input variables, but got {x.size}.")
    if not np.all((x >= -pi) & (x <= pi)):
        raise ValueError(f"All input variables $x_i$ must be in [$-\pi$, $\pi$].")

    x1, x2, x3 = x
    result = sin(x1) + a * (sin(x2) ** 2) + b * (x3 ** 4) * sin(x1)
    return float(result)


# --- SobolIshigamiRunner Implementation ---

class SobolIshigamiRunner(Runner):
    """
    SobolIshigamiRunner: a Runner implementation for evaluating common sensitivity analysis
    test functions: the **Sobol G-function** and the **Ishigami function**.
    ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function_name = kwargs.get('function_name', 'sobol_g').lower()
        self._analytical_sobol_indices = None
        self._analytical_total_variance = None


        # Function-specific parameters
        if self.function_name == 'ishigami':
            self.a = kwargs.get('a', 7.0)
            self.b = kwargs.get('b', 0.1)
        elif self.function_name == 'sobol_g':
            # **1. Get the 'a' parameters from kwargs**
            # This allows the config file to pass any dimension 'a' vector.
            self.sobol_a_params = kwargs.get('sobol_a_params', [0, 1, 4.5, 9, 99, 99, 99, 99])
            
            # **2. Calculate and store analytical Sobol indices during initialization**
            try:
                self._calculate_sobol_g_indices()
            except ValueError as e:
                # Re-raise error with context if parameters are invalid
                raise ValueError(f"Invalid Sobol G-function parameters provided: {e}") from e
                
        else:
            raise ValueError(f"Unsupported function_name: '{self.function_name}'. Must be 'sobol_g' or 'ishigami'.")

       
    # ----------------------------------------------------------------------
    # --- ANALYTICAL CALCULATION METHODS ---
    # ----------------------------------------------------------------------
    def _calculate_sobol_g_indices(self):
        """
        Calculates and stores the analytical first-order Sobol indices and Total Variance.
        """
        a_params = self.sobol_a_params
        D = len(a_params)
        
        if any(a < 0 for a in a_params):
             raise ValueError("All 'a' parameters must be non-negative.")
        
        # 1. Calculate Partial Variances (V_i) for each input
        # V_i = 1 / (3 * (1 + a_i)^2)
        partial_variances = [1.0 / (3.0 * (1.0 + a) ** 2) for a in a_params]

        # 2. Calculate Total Variance (V)
        # V = product(1 + V_i) - 1
        product_term = np.prod([1.0 + V_i for V_i in partial_variances])
        total_variance = product_term - 1.0

        # 3. Calculate First-Order Sobol Indices (S_i)
        # S_i = V_i / V
        S1_indices = [V_i / total_variance for V_i in partial_variances]
        
        # Store results as instance attributes
        self._analytical_sobol_indices = S1_indices
        self._analytical_total_variance = total_variance

    def get_analytical_sobol_g_indices(self):
        """
        Returns a dictionary containing the analytical First-Order Sobol Indices (S1_indices)
        and the Total Variance for the configured Sobol G-function.
        """
        if self.function_name != 'sobol_g':
            return {"Error": "Analytical indices are only available for the Sobol G-function."}

        # Format output as requested, including the print statement for a complete answer
        D = len(self.sobol_a_params)
        a_params = self.sobol_a_params
        S1_indices = self._analytical_sobol_indices
        total_variance = self._analytical_total_variance
        sum_S1 = sum(S1_indices)
        interaction_effects = 1.0 - sum_S1

        print("\n--- Analytical Sobol Indices (Configured in init) ---")
        print(f"Dimensionality (D): {D}")
        print(f"Parameters (a_i): {a_params}")
        print(f"Total Variance (V): {total_variance:.6f}")
        print("-" * 50)
        
        for i, S1 in enumerate(S1_indices):
            print(f"S1 Index for x{i+1} (a={a_params[i]}): {S1:.6f}")

        print("-" * 50)
        print(f"Sum of S1 Indices: {sum_S1:.6f}")
        print(f"Contribution from Higher-Order Interactions: {interaction_effects:.6f}")
        print("--------------------------------------------------")

        return {
            "S1_indices": S1_indices,
            "Total_Variance": total_variance,
            "Interaction_Contribution": interaction_effects
        }
    

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Executes the Sobol G-function or Ishigami function using the input parameters
        and returns the result.
        ...
        """
        params = params if params is not None else {}
        output = None
        success = False
        diagnostic_msg = ""
        
        try:
            if self.function_name == 'sobol_g':
                # The dimensionality D is derived dynamically from the configured 'a' parameters
                D = len(self.sobol_a_params)
                x_keys = [f'x{i}' for i in range(1, D + 1)]
                
                if not all(key in params for key in x_keys):
                    # This check now works for any D, based on the 'a' vector size
                    raise ValueError(f"Sobol G-function requires all {D} parameters: {', '.join(x_keys)}")
                
                x_values = [params[k] for k in x_keys]
                print(f'ANALYTICAL SOBOL INDICIES:\n {self.get_analytical_sobol_g_indices()}')
                output = sobol_g_function(x_values, self.sobol_a_params)

            elif self.function_name == 'ishigami':
                x_keys = [f'x{i}' for i in range(1, 4)]
                if not all(key in params for key in x_keys):
                    raise ValueError(f"Ishigami function requires all 3 parameters: {', '.join(x_keys)}")
                x_values = [params[k] for k in x_keys]
                output = ishigami_function(x_values, a=self.a, b=self.b)

            success = True

        except Exception as e:
            success = False
            diagnostic_msg = f"Run failed due to: {type(e).__name__}: {str(e)}"

        final_output = output if is_number(output) else (float('nan') if not success else 0.0)

        result = {
            "output": final_output,
            "success": success,
            "function_used": self.function_name,
        }
        if diagnostic_msg:
            result["diagnostic_msg"] = diagnostic_msg

        return result


# --- DEMONSTRATION OF USAGE ---
if __name__ == '__main__':
    
    print("--- Demonstration 1: 12-Dimensional Sobol G-function ---")
    
    # Simulating the config file passing 12 parameters (via kwargs)
    D12_params = [0, 1, 4.5, 9, 15, 20, 30, 40, 50, 60, 70, 80]
    
    runner_12d = SobolIshigamiRunner(sobol_a_params=D12_params)
    
    # This call now runs the analytical calculation that was requested
    runner_12d.get_analytical_sobol_g_indices() 

    # --- Demonstration 2: Standard 8-Dimensional Sobol G-function ---
    print("--- Demonstration 2: Standard 8-Dimensional Sobol G-function ---")
    
    runner_8d = SobolIshigamiRunner() # Uses the default 8-dim parameters
    runner_8d.get_analytical_sobol_g_indices()