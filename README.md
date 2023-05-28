# First Order Reliability Method - FORM
First Order Reliability Method implementation using HLRF-BFGS to solve the reliability problem as described in [1-3].
As such, it is possible to analytically solve a reliability problem, taking in account diverse types of correlated random variables, analytical and numerical limit state functions and system definitions.

## Installation

Just copy/git clone the FORM repository into your project root or someplace discoverable by your python environment (PYTHONPATH, sys.path, site-packages, etc).
Creating a symbolic link on those places pointing to the FORM repo is also another option.

## Requirements

This repo makes use of the following libraries:

- numpy
- scipy

Please install them beforehand on your python environment.

## Usage

### Random Variables

The random variables to supply as input must be random variable objects from scipy.stats module.
In this manner, the user is responsible for the random variables specific parameters and fitting process.
The function `generate_RV(...)` is also supplied to help finding a random variable with desired mean $\mu$ and standard deviation $\sigma$.

The input description of said function is:
```python
generate_RV(rv, mean, std, fixed_params: dict[str:float], search_params: list[str], x0=None, method="lm", tol=1e-4):

# rv: Random variable object from scipy.stats.

# mean: Mean value of the resulting random variable.

# std: Standard deviation value of the resulting random variable.

# fixed_params: Dictionary defining the values of the random variable parameters that will be adopted throught the root finding process to fit the scipy.stats random variable object.

# search_params: List of parameter names of the scipy.stats random variable object that will be fitted by the root finding algorithm, resulting in an object with the desired mean and/or standard deviation value(s).

# x0: A initial values list of [mean] or [mean, standard deviation]. Those are used only as starting value for the root finding algorithm. They can, and should be, changed if there is some difficulty on convergence.

# method: Method for root finding (see "root" function from scipy.optimize).

# tol: Tolerance value. After the root function invocation, there is a verification to check if the mean and standard deviation of the resulting random variable object has the desired mean and std. That is, abs(rv.mean() - mean) < tol and abs(rv.std() - std) < tol.
```

#### Examples

```python
X1 = scipy.stats.norm()
X2 = scipy.stats.norm(9.9, 0.8)
X3 = generate_RV(
    scipy.stats.gumbel_l,
    5.8,
    1.9,
    fixed_params={},
    search_params=["loc", "scale"]
)

Xi = [X1, X2, X3]

Xd = [
    generate_RV(
        scipy.stats.lognorm,
        5.0,
        1.2,
        fixed_params={"loc": 0},
        search_params=["s", "scale"]
    )
]
```

### Limit State Functions

Limit state functions define the modes of failure of interest.
Those are a function of the random and deterministic variables of the problem.
The vector of random variables is denoted (in index notation) as $X_m$, whereas the vector of deterministic variables is described as $d_n$.
A distinction is made for the random variables, resulting in 2 groups of random variables:

- $Xi_m$ is a vector of *independent* random variables.
- $Xd_m$ is a vector of *design* random variables.

$d_n$ is also described as a vector of *design* deterministic variables.

The *design* variables, both random and deterministic, are employed in the context of numerical optimization.
The use of this library for solving optimization problems with reliability constraints won't be discussed in this repo.

With that said, the description of a limit state function $g$ is given as:
```python
g(Xi, Xd, d):

# Xi: list of independent random variable objects.

# Xd: list of design random variable objects.

# d: list of design deterministic variable.
```

The limit state function can be defined as analytical or numerical (by wrapping the numerical method function).
As input, FORM must be supplied with a list of limit state functions (even if there is only one).

#### Examples

```python
def g1(Xi, Xd, d):
    X1, X2, X3 = Xi
    return X1**3 + X2**3 - X3/10.0

def g2(Xi, Xd, d):
    X1, X2, X3 = Xi
    d1, d2 = d
    return X1*X3*d1 - X2*d2

def g3(Xi, Xd, d):
    X1, X2, X3 = Xi
    any_other_variable = 42
    max_threshold_value = 3.3
    return max_threshold_value - my_awesome_FEM_model_result(X1, X2, X3, any_other_variable)

limit_state_functions = [g1, g2, g3]
```

### System Definitions

A system definition is expressed as a dictionary with key either "serial" or "parallel" and a value representing a list made of integers and/or subsystem definition dictionaries.
The integer values map out to the index position of a limit state function contained in the limit state functions input list.
This way, it's possible to represent either "serial" or "parallel" systems only.
The evaluation of the system's failure probabilities are done according to [4].
It's only possible to compute "serial" and "parallel" systems in this implementation.

#### Examples

```python
system1 = {"serial": [0, 1, 2]}
system2 = {"parallel": range(len(limit_state_functions))}

system_definitions = [system1, system2]
```

### FORM Interface

The main numerical method is contained in the FORM object.
It's methods compose the interface available to the user to solve the reliability problem.
The following methods are exposed to the user:
```python
FORM()
FORM.HLRF(...)
```

The constructor creates the FORM object and doesn't need initial arguments:
```python
FORM():
```

The evaluation of the limit state functions and system definitions are made by calling the HLRF method.
During the evaluation process, a vector (`self.limit_state_trace_data`) that contains the trace data of each one of the limite state functions is internally stored.
The HLRF method returns a result object with failure probabilities of the limit state functions and system definitions.
The signature of the HLRF is as follows:
```python
FORM.HLRF(limit_state_functions, system_definitions=None, Xi=None, Xd=None, d=None, correlation_matrix=None, epsilon=2.0001, delta=1e-4, max_number_iterations=1000, h_numerical_gradient=1e-3):
# limit_state_functions: A list consisting of functions objects defining each one of the modes of failure of the problem.

# system_definitions: A list of system definitions.

# Xi: A list of independent random variable objects.

# Xd: A list of design random variable objects.

# d: A list of design deterministic variable.

# correlation_matrix: A correlation matrix respective to both independent and design random variables. This matrix must be symmetrical and square. If None, it's assumed that all random variables are completely uncorrelated.

# epsilon: Epsilon value from one of the two convergence criteria for the HLRF algorithm.

# delta: Delta value from one of the two convergence criteria for the HLRF algorithm.

# max_number_iterations: Maximum number of iteriations for the HLRF algorithm.

# h_numerical_gradient: Delta "h" for the numerical gradient algorithm for the limit state functions.
```

### Result object

The result object is a namedtuple of type `result(gX_results, system_results)`.
With `gX_results` and `system_results` of types `gX_results(pfs, betas)` and `system_results(pfs, betas)` respectively.
Both `pfs` and `betas` are lists of failure probabilities and $\beta$ indexes of the limit state functions and system definitions.
The positional indexes of those values correspond to the same indexes as the input variables on the `FORM.HLRF(...)` method.

### Numerical example

The following example is the eighth numerical problem on [3].
It's the reliability analysis of a non linear mathematical function, with failure mode given by the limit state function $g_1$.
The problem definition and resolution by the FORM library implemented is:
```python
def g1(Xi, Xd, d):
    Xi1, Xi2 = Xi
    return Xi1**3 + Xi2**3 - 18

Xi = [st.norm(10, 5), st.norm(9.9, 5)]

f = FORM()
res = f.HLRF([g1], Xi=Xi)

print(res.gXs_results)
print(res.systems_results)
print(res.gXs_results.pfs)
print(res.gXs_results.betas)

gx1_trace_data = f.limit_states_trace_data[0]  # This is an internal variable that stores de trace data from each [i] limit state function.
print(gx1_trace_data.beta_k_trace)
print(gx1_trace_data.k)
```

This same problem is described in the `example.py` file, together with, possibly, other problems.

## References

[1]: Hasofer AM, Lind NC. Exact and invariant second-moment code format. Journal of the Engineering Mechanics Division. 1974;100(1):111–121.
 
[2]: Rackwitz R, Flessler B. Structural reliability under combined random load sequences. Computers & structures. 1978;9(5):489–494.

[3]: Periçaro GA, Santos SR, Ribeiro AA, Matioli LC. HLRF–BFGS optimization algorithm for structural reliability. Applied mathematical modelling. 2015;39(7):2025–2035.

[4]: Song J, Kiureghian AD. Bounds on System Reliability by Linear Programming. Jounal of Engineering Mechanics. 2003;129(6):627–636.
