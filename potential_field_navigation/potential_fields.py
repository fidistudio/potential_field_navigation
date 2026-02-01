import numpy as np
from typing import Iterable

# -----------------------------
# Configuration constants
# -----------------------------

MIN_DISTANCE = 1e-6
MAX_ATTRACTIVE_DISTANCE = 10.0

REPULSION_RADIUS = 1.5
ATTRACTIVE_GAIN_NEAR = 1.0
ATTRACTIVE_GAIN_FAR = 1.0
REPULSIVE_GAIN = 2.0
STEP_SIZE = 0.3


# -----------------------------
# Force computations
# -----------------------------


def attractive_force(position: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    Computes the attractive force pulling the agent toward the goal.
    """
    displacement = position - goal
    distance = np.linalg.norm(displacement)

    if distance < MIN_DISTANCE:
        return np.zeros(2)

    if distance < MAX_ATTRACTIVE_DISTANCE:
        return ATTRACTIVE_GAIN_NEAR * displacement

    return ATTRACTIVE_GAIN_FAR * displacement / distance


def repulsive_force(position: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """
    Computes the repulsive force pushing the agent away from an obstacle.
    """
    displacement = position - obstacle
    distance = np.linalg.norm(displacement)

    if distance < MIN_DISTANCE:
        return np.zeros(2)

    if distance >= REPULSION_RADIUS:
        return np.zeros(2)

    scaling = (1 / distance - 1 / REPULSION_RADIUS) / (distance**3)
    return -REPULSIVE_GAIN * scaling * displacement


# -----------------------------
# Motion update
# -----------------------------


def compute_next_position(
    position: np.ndarray,
    goal: np.ndarray,
    obstacles: Iterable[np.ndarray],
) -> np.ndarray:
    """
    Computes the next position using normalized potential field forces.
    """
    total_force = attractive_force(position, goal)

    for obstacle in obstacles:
        total_force += repulsive_force(position, obstacle)

    force_magnitude = np.linalg.norm(total_force)

    if force_magnitude < MIN_DISTANCE:
        return position

    direction = total_force / force_magnitude
    return position - STEP_SIZE * direction
