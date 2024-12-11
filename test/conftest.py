"""
Global pytest configuration.

**Disclaimer: some functions copied from firedrake/tests/firedrake/conftest.py
"""


def pytest_configure(config):
    """
    Register an additional marker.

    **Disclaimer: copied from firedrake/tests/firedrake/conftest.py
    """
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow to run",
    )
