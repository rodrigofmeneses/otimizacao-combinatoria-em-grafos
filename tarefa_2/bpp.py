import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from functools import wraps
from itertools import product
import time, os
import pandas as pd


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        time_taken = time.time() - start
        print(f"time taken by {func.__name__} is {time_taken }")

        return result, time_taken

    return wrapper


def build_model(data, strengthening=False, symmetry_breaking=False):
    model = pyo.ConcreteModel()

    # Sets
    model.ITEMS = pyo.Set(initialize=data["ITEMS"])
    model.BINS = pyo.Set(initialize=data["BINS"])

    # Params
    model.Weight = pyo.Param(
        model.ITEMS, domain=pyo.NonNegativeIntegers, initialize=data["Weight"]
    )
    model.Capacity = pyo.Param(
        domain=pyo.NonNegativeIntegers, initialize=data["Capacity"]
    )

    # Vars
    model.x = pyo.Var(model.ITEMS, model.BINS, domain=pyo.Binary)
    model.y = pyo.Var(model.BINS, domain=pyo.Binary)

    # Base Model
    # Constraints
    @model.Constraint(model.BINS)
    def bin_capacity_rule(model, j):
        return (
            sum(model.Weight[i] * model.x[i, j] for i in model.ITEMS)
            <= model.Capacity * model.y[j]
        )

    @model.Constraint(model.ITEMS)
    def assign_rule(model, i):
        return sum(model.x[i, j] for j in model.BINS) == 1

    # Objective Function
    @model.Objective(sense=pyo.minimize)
    def obj_rule(model):
        return sum(model.y[j] for j in model.BINS)

    # Variants
    if strengthening:
        # Strengthening
        @model.Constraint(model.ITEMS, model.BINS)
        def strengthening(model, i, j):
            return model.x[i, j] <= model.y[j]

    if symmetry_breaking:
        # Breaking Symmetry
        @model.Constraint(model.BINS)
        def symmetry_breaking(model, j):
            if j == 0:
                return pyo.Constraint.Skip
            return sum(model.x[i, j - 1] for i in model.ITEMS) >= sum(
                model.x[i, j] for i in model.ITEMS
            )

    return model


@time_it
def solve_model(model, solver, options={}):
    opt = SolverFactory(solver)
    opt.options = options
    results = opt.solve(model)
    return results.solver.status.value


def load_data(filename):
    with open(filename, "r") as file:
        n, bin_capacity, *weights = list(map(int, file))
    return n, bin_capacity, weights


if __name__ == "__main__":
    with open("results.csv", "w") as f:
        f.write(
            "Instance,Strengthening,Symmetry Breaking,Solver Status,Execution Time\n"
        )

        for variants in product([True, False], repeat=2):
            print("Strengthening:", variants[0], "Symmetry Breaking:", variants[1])

            for filename in os.listdir("instances"):
                n, bin_capacity, weights = load_data("instances/" + filename)
                data = {
                    "ITEMS": range(n),
                    "BINS": range(n),
                    "Weight": {key: value for key, value in enumerate(weights)},
                    "Capacity": bin_capacity,
                }

                model = build_model(data)

                solver_status, execution_time = solve_model(
                    model, "glpk", options={"tmlim": 60, "mipgap": 0.0001}
                )
                f.write(
                    f"{filename},{variants[0]},{variants[1]},{solver_status},{execution_time}\n"
                )
