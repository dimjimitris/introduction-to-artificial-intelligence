# EARIN - 3rd mini-project
Goal is to write a Python program to solve a given 2D function using a basic genetic algorithm. The function to optimize is Rastrigin function using Rank Selection:

f(x,y) = 20 + (x^2 − 10cos(2πx)) + (y^2 − 10cos(2πy))

'x' and 'y' must be initialized from range [-5,5].

Gaussian operator must be used for mutation and random interpolation for cros- sover (e.g.: x_o = α * x_p1 + (1 − α) * x_p2, y_o = α * y_p1 + (1 − α) * y_p2, where 'o', 'p1', 'p2' refer to offspring, parent 1 and parent 2, and 'α' is a random number; this version of crossover operator is practically used for real-valued problems). It will be assumed that the current population is always replaced with the new offspring.

## Repo setup

* Create and activate python environment. You can use either [venv](https://docs.python.org/3/library/venv.html)
  or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment).
* Install the required packages:

```bash
pip install -r requirements.txt
```