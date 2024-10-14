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
    # TODO #222: GlobalKernel no longer has _cache attribute
    from firedrake.tsfc_interface import clear_cache
    # from pyop2.global_kernel import GlobalKernel

    clear_cache()
    # GlobalKernel._cache.clear()
