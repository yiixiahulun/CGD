# STEP 6: Test Execution Log

## Module: `tests/core`

**Execution Timestamp**: 2025-11-19 08:07:59 UTC

**Outcome**: All tests passed successfully after initial debugging of the test code and environment.

### Summary:

The initial test runs failed due to several issues:
1.  Missing project dependencies (`numpy`, `cgd` package itself, `ternary`). This was resolved by creating a `pyproject.toml` and installing the project in editable mode.
2.  Incorrect assumptions in the test code regarding function locations (`log_map_from_origin`) and the existence of a `project_onto_subsimplex` function. This was corrected by inspecting the source code and updating the test files.
3.  Discrepancies between the test code and the source code, such as asserting for a `.K` attribute that didn't exist and expecting English error/warning messages instead of the actual Chinese ones.
4.  Missing registration for custom `pytest` markers.

After systematically fixing these issues in the test code and `pyproject.toml`, the test suite for the `core` module executed successfully. All outcomes are now as expected.

---

### Final Pytest Output:

```
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.0.1, pluggy-1.6.0
rootdir: /app
configfile: pyproject.toml
plugins: mock-3.15.1, cov-7.0.0
collected 19 items

tests/core/test_source.py ........                                       [ 42%]
tests/core/test_universe.py ...........                                  [100%]

============================== 19 passed in 2.48s ==============================
```
