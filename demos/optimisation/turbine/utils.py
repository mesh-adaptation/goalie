import datetime
import subprocess

__all__ = ["get_experiment_id"]


def get_git_hash():
    """Generate experiment identifier"""
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        git_hash = get_git_hash()
        return f"{timestamp}_{git_hash}"
    except Exception as e:
        raise RuntimeError("Could not generate experiment ID.") from e
