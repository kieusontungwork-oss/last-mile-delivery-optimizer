"""Optimization package: VRP solvers and cost matrix construction."""

from src.optimization.vrp_solver import ORToolsSolver, PyVRPSolver


def get_solver(solver_name: str) -> PyVRPSolver | ORToolsSolver:
    """Return a solver instance by name.

    Args:
        solver_name: Either "pyvrp" or "ortools".

    Returns:
        Solver instance with a .solve() method.
    """
    if solver_name == "pyvrp":
        return PyVRPSolver()
    elif solver_name == "ortools":
        return ORToolsSolver()
    else:
        raise ValueError(f"Unknown solver: {solver_name!r}. Use 'pyvrp' or 'ortools'.")
