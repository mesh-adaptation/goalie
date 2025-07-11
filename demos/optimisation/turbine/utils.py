import datetime
import os
import subprocess

__all__ = ["get_experiment_id", "get_latest_experiment_id"]


def get_git_hash():
    """Get the current git hash."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError as cpe:
        raise RuntimeError("Could not retrieve git hash.") from cpe


def get_experiment_id():
    """Generate experiment identifier with datetime stamp and git hash"""
    try:
        return get_latest_experiment_id()
    except ValueError:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        git_hash = get_git_hash()
        return f"{timestamp}_{git_hash}"


def get_latest_experiment_id(hash=None):
    """
    Get the latest experiment ID for a given git hash. If the hash is None, use the
    current hash.
    """
    hash = hash or get_git_hash()
    experiments_dir = "outputs"
    experiment_ids = []

    # Iterate through the experiments directory to find matching hashes
    for experiment_id in os.listdir(experiments_dir):
        if experiment_id.endswith(hash):
            experiment_ids.append(experiment_id)

    if not experiment_ids:
        raise ValueError(f"No experiments found for hash: {hash}")

    # Sort the experiment IDs and return the latest one
    experiment_ids.sort()
    return experiment_ids[-1]
