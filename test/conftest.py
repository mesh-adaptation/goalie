"""
Global pytest configuration.

**Disclaimer: some functions copied from firedrake/src/tests/conftest.py
"""


def pytest_configure(config):
    """
    Register an additional marker.

    **Disclaimer: copied from firedrake/src/tests/conftest.py
    """
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow to run",
    )


def pytest_runtest_teardown(item, nextitem):
    """
    Clear caches after running a test
    """
    from firedrake.tsfc_interface import TSFCKernel
    from pyop2.global_kernel import GlobalKernel

    if hasattr(TSFCKernel, "_cache"):
        TSFCKernel._cache.clear()
    if hasattr(GlobalKernel, "_cache"):
        GlobalKernel._cache.clear()
