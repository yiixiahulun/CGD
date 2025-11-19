# standalone_consistency_check.py
"""
A standalone script to verify the numerical consistency of the NumPy and JAX
geometry backends, bypassing the pytest runner.
"""
import numpy as np

def run_check():
    """Runs the consistency checks."""
    print("--- Standalone Backend Consistency Check ---")

    try:
        # Import NumPy versions
        from cgd.geometry import simplex as simplex_np
        from cgd.geometry import transformations as trans_np

        # Import JAX versions
        import jax
        import jax.numpy as jnp
        from cgd.geometry import _jax_impl as jax_impl
        print("SUCCESS: JAX and all source modules imported successfully.")
    except ImportError as e:
        print(f"FAILURE: Could not import necessary modules. Error: {e}")
        return

    # --- Test Data (copied from test_backend_consistency.py) ---
    P_VECTORS = [
        np.array([1.0]),
        np.array([0.5, 0.5]),
        np.array([1.0, 0.0]),
        np.array([1/3, 1/3, 1/3]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.8, 0.1, 0.1]),
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.4, 0.3, 0.2, 0.1]),
    ]
    V_VECTORS = [
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.array([0.5, -0.5]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.6, -0.3, -0.3]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.3, -0.1, -0.1, -0.1]),
    ]

    TOLERANCE = 1e-6
    failures = []

    # --- 1. Test distance_FR ---
    print("\nChecking distance_FR consistency...")
    for i, p1_np in enumerate(P_VECTORS):
        for j, p2_np in enumerate(P_VECTORS):
            if p1_np.shape != p2_np.shape:
                continue

            p1_jax, p2_jax = jnp.array(p1_np), jnp.array(p2_np)
            dist_np = simplex_np.distance_FR(p1_np, p2_np)
            dist_jax = jax_impl.distance_FR_jax(p1_jax, p2_jax)
            if not np.isclose(dist_np, dist_jax, atol=TOLERANCE):
                failures.append(f"distance_FR failed for p_vectors {i} and {j}")

    # --- 2. Test log_map_from_origin ---
    print("Checking log_map_from_origin consistency...")
    for i, p_np in enumerate(P_VECTORS):
        p_jax = jnp.array(p_np)
        v_np = trans_np.log_map_from_origin(p_np)
        v_jax = jax_impl.log_map_from_origin_jax(p_jax)
        try:
            np.testing.assert_allclose(v_np, v_jax, atol=TOLERANCE)
        except AssertionError:
            failures.append(f"log_map_from_origin failed for p_vector {i}")

    # --- 3. Test exp_map_from_origin ---
    print("Checking exp_map_from_origin consistency...")
    for i, v_np in enumerate(V_VECTORS):
        v_jax = jnp.array(v_np)
        p_np = trans_np.exp_map_from_origin(v_np)
        p_jax = jax_impl.exp_map_from_origin_jax(v_jax)
        try:
            np.testing.assert_allclose(p_np, p_jax, atol=TOLERANCE)
        except AssertionError:
            failures.append(f"exp_map_from_origin failed for v_vector {i}")

    # --- Final Report ---
    print("\n--- FINAL REPORT ---")
    if not failures:
        print("✅ SUCCESS: All NumPy and JAX geometry backends are numerically consistent.")
    else:
        print(f"❌ FAILURE: Found {len(failures)} inconsistencies:")
        for f in failures:
            print(f"  - {f}")

if __name__ == "__main__":
    run_check()
