---
layout: default
title: HELENA
nav_order: 4
parent: Plugins
---

# HELENA plugin

The HELENA plugin provides a parser and runner for the HELENA MHD equilibrium code. You need a compiled HELENA executable to use this plugin. The plugin supports different input parameter types and can handle beta iteration to achieve a target normalized beta.
Mandatory fields in the configuration file for using the HELENA plugin:

```yaml
runner:
  type: HelenaRunner
  executable_path: "/path/to/helena/executable"
  other_params:
    namelist_path: "/path/to/fort.10"
    only_generate_files: True
    input_parameter_type: 1

```



## Input parameter types

The HELENA plugin supports multiple different input parameter settings, which determine how the input files are generated based on the sample parameters. The supported types are:

- Type 1: Standard input parameters. The sample parameters directly correspond to the input parameters in the HELENA namelist.
  
```yaml
sampler: 
    type: RandomSampler
    bounds: [[0.02, 0.04], [0.0, 0.45], [1.0, 3.0]]
    parameters: ['pedestal_delta', 'tria', 'beta_N']
    budget: 10

```

- Type 2:

- Type 3:

- Type 4: Europed parameters. The sample parameters correspond to the Europed parameterization, which includes parameters such as the pedestal temperature gradient, core temperature slope, and fast ion pressure scaling factor. These parameters are then converted to the standard HELENA input parameters internally. The KBM constaint is applied.
  
```yaml
sampler: 
    type: RandomSampler
    bounds: [[0.02, 0.04], [0.0, 0.45], [1.0, 3.0]]
    parameters: ['pedestal_delta', 'tria', 'beta_N']
    budget: 10

```

- Type 5: Europed-type parameters without KBM constraint. Similar to Type 4, but the KBM constraint is not applied.

## Example configuration


## Beta itertaion

If a specific normalized beta is required, the HELENA plugin contains multple methods to achieve this. Firstly, the target beta_N has to be specified in the parameters section of the configuration file:

```yaml
sampler: 
    type: RandomSampler
    bounds: [[0.02, 0.04], [0.0, 0.45], [1.0, 3.0]]
    parameters: ['pedestal_delta', 'tria', 'beta_N']
    budget: 10
    
runner:
  type: HelenaRunner
  ...
  other_params:
    ...
    beta_iteration: 1
    beta_tolerance: 0.01
    max_beta_iterations: 4
```
---
### 1. Iterative adustment of the core temperature slope 

Linear + secant method approach.


#### Step 1: Initialization

Run HELENA at two initial values of the core temperature slope $a_{T}$:

$$
a_{T,0} = 0, \quad a_{T,1} = 1,
$$

with corresponding outputs

$$
\beta_{N,0} = \beta_N(a_{T,0}), \quad \beta_{N,1} = \beta_N(a_{T,1}).
$$

#### Step 2: Update

#### Step 3: Convergence


---
### 2. Calculation of an approximate core temperature slope

From Europed. Newton's method.
This method reconstructs the pressure profile and iteratively adjusts the core temperature slope to reach the target beta_N, but without running HELENA multiple times. The procedure is as follows:

---
### 3. Iterative adjustment of the fast ion pressure profile

Secant method.
This method will iteratively adjust the amplitude of the fast ion pressure profile to reach the target beta_N. 
We assume that the normalized beta, $\beta_N$, is a function of the fast-ion pressure scaling factor $apf$. Since the explicit functional form $\beta_N(apf)$ is not known, the secant method is used to iteratively approximate the root of

$$
f(apf) = \beta_N(apf) - \beta_N^{\text{target}} = 0.
$$

The iteration procedure is as follows:


#### Step 1: Initialization

Run HELENA at two initial values of the fast-ion pressure scaling factor:

$$
apf_{0} = 0, \quad apf_{1} = 1,
$$

with corresponding outputs

$$
\beta_{N,0} = \beta_N(apf_{0}), \quad \beta_{N,1} = \beta_N(apf_{1}).
$$



#### Step 2: Secant update

For iteration $k \geq 2$, given the last two approximations $(apf_{k-2}, \beta_{N,k-2})$ and $(apf_{k-1}, \beta_{N,k-1})$, we compute the slope:

$$
s = \frac{\beta_{N,k-1} - \beta_{N,k-2}}{apf_{k-1} - apf_{k-2}}.
$$

The new guess is obtained using the secant method:

$$
apf_{k} = apf_{k-1} + \frac{\beta_N^{\text{target}} - \beta_{N,k-1}}{s}.
$$



#### (Optional) Step 3: Stabilization

To improve robustness, two modifications can be applied:

1. **Damping**:  
   A fraction of the update is mixed with the previous iterate,  

   $$
   apf_{k} \; \leftarrow \; (1-\lambda)\, apf_{k} + \lambda\, apf_{k-1}, \quad 0 < \lambda < 1.
   $$

   (In the code, $\lambda = 0.3$. TODO: make this user-defined in the config file.)

2. **Step-size limit**:  
   To avoid divergence (HELENA crashed easily on too large apf values), the step  

   $$
   \Delta apf = apf_{k} - apf_{k-1}
   $$

   is restricted to $\|\Delta apf \| \leq \Delta apf^{\max}$ (with $\Delta apf^{\max} = 2.$ in the code.  TODO: make this user-defined in the config file).



#### Step 4: Iteration and Convergence

HELENA is rerun at the new $apf_{k}$ to compute $\beta_{N,k} = \beta_N(apf_{k})$.  

The iteration continues until the stopping criterion is met:

$$
\big| \beta_{N,k} - \beta_N^{\text{target}} \big| < \varepsilon \, \beta_N^{\text{target}},
$$

where $\varepsilon$ is the user-defined tolerance (`beta_tolerance`) or until the maximum number of iterations are met (`max_beta_iterations`).

---

