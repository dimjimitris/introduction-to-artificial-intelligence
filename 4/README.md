# EARIN - 4th mini-project

Goal is to write a model to predict disease progression on the given diabetes dataset (https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset).

## Procedure
1. **Data preparation.** Features in the dataset will be analyzed to decide which ones are useful for the task. Some features might be redundant, for example, a dataset might contain features that depend on each other, such as ”pre-tax price”, ”tax”, and ”post-tax price”, where the third one is just a sum of the previous two. In such cases, it’s better to remove dependant features. If the dataset is not normalized, normalization should also be performed. Thought process will be described in the report.
2. **Data split.** Data will be split into train and test sets.
3. **Model definition.** Two models to fit will be chosen. The choice of the models will be explained in the report.
4. **Model training.** An appropriate metric to evaluate the models will be selected. Different model parameters for both models will be tested, comparing their performance using cross-validation framework with 4 folds. The tested parameters will be documented for both models in separate tables. Performance of both models with the best found parameters will be compared.

## Repo setup

* Create and activate python environment. You can use either [venv](https://docs.python.org/3/library/venv.html)
  or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment).
* Install the required packages:

```bash
pip install -r requirements.txt
```
