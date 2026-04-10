# Tutorial: Bayesian Optimization

This tutorial explains the example Bayesian optimization workflow in Enchanted Surrogates. The Bayesian optimization example is a synthetic test case that uses the [Example BO Runner](runners/example_bayesian_optimization_runner.md), the [Example BO Parser](parsers/example_bayesian_optimization_parser.md), and an example configuration file  (`configs/example_bayesian_optimization_local.yaml`).



Make sure that when you installed `enchanted-surrogates`, you included the optional dependencies needed for the [Bayesian Optimization Sampler](samplers/bayesian_optimization_sampler.md). 


Run the example:

```bash
cd enchanted-surrogates/
python src/run.py -cf configs/example_bayesian_optimization_local.yaml
```


This Bayesian optimization example aims to solve the problem "find $x$ that makes $f(x)$ closest to the target value $y_{target}$". In this example case, $y_{target} = 2.0$.

The example runner implements a **synthetic 1D objective function** defined as a linear term plus two Gaussian bumps. Note that in reality, the objective function is seldom one-dimensional or simple. 
The formula for the objective funtion in this example is:

$$
f(x) = a \cdot x + g_{11} \cdot \exp\left(-\frac{(x - g_{13})^{2}}{g_{12}}\right) + g_{21} \cdot \exp\left(-\frac{(x - g_{23})^{2}}{g_{22}}\right)
$$


The parameter values for the runner can be defined in the config file. This example uses the default parameters:

- $a = 0.2$ (linear slope)
- $g_{11} = 1.0, g_{12} = 0.001, g_{13} = 0.2$ (first Gaussian)
- $g_{21} = 0.6, g_{22} = 0.01, g_{23} = 0.7$ (second Gaussian)




The search space (bounds) for $x$, the number of samples (budget) and the aquisition function are defined in the sampler section of the config file.

```yaml
samplers:
  s1:
    type: BayesianOptimizationSampler
    budget: 50
    bounds: [[0.001, 1.0]]
    parameters: ['x']
    initial_samples: 3
    acquisition_function: LEI
    random_fraction: 0.5

```

In this example, the sampler starts by generating three `initial_samples`. These are three random points in the search space that serves are starting points for the optimization task. 

Each sample is passed to the runner, which evaluates the objective function $f(x_{sample})$. The samples and objective function evaluations (`output`) are passed to the parser. The parser scores each sample by how close `output` is to the target `2.0`.
In this case, the parser computes a simple distance metric:

$$
d = y_{target} - f(x_{sample})
$$


The combined run inputs and outputs are saved into `enchanted_datapoint.csv` after
each run. Having some initial random samples, the Bayesian Optimization phase can start. Firstly, the parser reads the existing `enchanted_datapoint.csv` files so the sampler can reconstruct the objective from past runs. Secondly, a Gaussian Process model is trained on the data and the acquisition function (e.g., LEI) suggests new promising points to evaluate. These points (samples) are evaluated and saved into `enchanted_datapoint.csv`.
The sampling and parsing is repeated until the budget is exhausted.

The figure below shows the objective function, the initial random points and the accuired samples. It also displays the targed $y_{target}=2.0$ as a line. In this synthetic 1D example, it is clear that $f(x=0.2)$ is the value closest to the target. 


![fig](img/bo_output.png)

Load and analyze the results:

```python
import pandas as pd
df = pd.read_csv('data_dir/bayesian/enchanted_dataset.csv')
```

## Extending the example

You can adapt this example for a real simulation by:

- replacing `ExampleBayesianOptimizationRunner` with a runner that invokes your code
- implementing a parser that reads the real output files and defining your objective in `collect_sample_information`
- updating `bounds` and `parameters` to match your input space

