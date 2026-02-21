#!/usr/bin/env python3
"""
Injection Prior Registry Validation Test.

Asserts that the code defaults in sample_source_params() and
sample_lens_params() match the single source of truth in
configs/injection_priors.yaml.

This prevents code-to-paper drift: if anyone changes a default in the
code without updating the YAML (or vice versa), this test fails.

NOTE: This test uses AST parsing to read defaults from the source code
directly, avoiding heavy imports (torch, etc.) that may not be available
in all test environments.

Created: 2026-02-13

Usage:
    cd stronglens-realism-gap
    export PYTHONPATH=.
    python -m pytest tests/test_injection_priors.py -v
    # Or standalone:
    python tests/test_injection_priors.py
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "configs" / "injection_priors.yaml"
ENGINE_PATH = Path(__file__).resolve().parent.parent / "dhs" / "injection_engine.py"


def _load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def _extract_function_defaults(source_path: Path, func_name: str) -> dict:
    """Extract default parameter values from a function definition using AST.

    This avoids importing the module (and its heavy dependencies like torch).
    Handles: numbers, tuples of numbers, None.
    """
    source = source_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            defaults = {}
            args = node.args
            # args.defaults correspond to the last N parameters
            n_defaults = len(args.defaults)
            param_names = [a.arg for a in args.args]
            default_params = param_names[len(param_names) - n_defaults:]
            for name, default_node in zip(default_params, args.defaults):
                val = _ast_to_value(default_node)
                if val is _SENTINEL:
                    # Fail loudly instead of silently skipping.
                    # If a default becomes a computed expression (e.g. 0.5 * MAX_RE),
                    # we want to know about it so the test can be updated.
                    raise ValueError(
                        f"Cannot parse default for parameter '{name}' in "
                        f"{func_name}(). The AST parser does not support "
                        f"complex expressions. Please use a simple literal "
                        f"default or update the parser."
                    )
                defaults[name] = val
            return defaults

    raise ValueError(f"Function {func_name} not found in {source_path}")


_SENTINEL = object()


def _ast_to_value(node):
    """Convert an AST node to a Python value. Returns _SENTINEL if unsupported."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        vals = [_ast_to_value(e) for e in node.elts]
        if _SENTINEL in vals:
            return _SENTINEL
        return tuple(vals)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_to_value(node.operand)
        if inner is not _SENTINEL:
            return -inner
    if isinstance(node, ast.Name) and node.id == "None":
        return None
    # For NameConstant (Python 3.7 compat)
    if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant):
        return node.value
    return _SENTINEL


class TestSourceParamsMatchRegistry(unittest.TestCase):
    """Assert sample_source_params() defaults match injection_priors.yaml."""

    def setUp(self):
        self.registry = _load_registry()["source_params"]
        self.defaults = _extract_function_defaults(ENGINE_PATH, "sample_source_params")

    def test_registry_file_exists(self):
        self.assertTrue(REGISTRY_PATH.exists(),
                        f"Registry file not found: {REGISTRY_PATH}")

    def test_engine_file_exists(self):
        self.assertTrue(ENGINE_PATH.exists(),
                        f"Engine file not found: {ENGINE_PATH}")

    def test_r_mag_range(self):
        self.assertEqual(
            list(self.defaults["r_mag_range"]),
            self.registry["r_mag_range"],
        )

    def test_beta_frac_range(self):
        self.assertEqual(
            list(self.defaults["beta_frac_range"]),
            self.registry["beta_frac_range"],
        )

    def test_re_arcsec_range(self):
        self.assertEqual(
            list(self.defaults["re_arcsec_range"]),
            self.registry["re_arcsec_range"],
        )

    def test_n_range(self):
        self.assertEqual(
            list(self.defaults["n_range"]),
            self.registry["n_range"],
        )

    def test_q_range(self):
        self.assertEqual(
            list(self.defaults["q_range"]),
            self.registry["q_range"],
        )

    def test_g_minus_r_mu_sigma(self):
        self.assertEqual(
            list(self.defaults["g_minus_r_mu_sigma"]),
            self.registry["g_minus_r_mu_sigma"],
        )

    def test_r_minus_z_mu_sigma(self):
        self.assertEqual(
            list(self.defaults["r_minus_z_mu_sigma"]),
            self.registry["r_minus_z_mu_sigma"],
        )

    def test_clumps_prob(self):
        self.assertAlmostEqual(
            self.defaults["clumps_prob"],
            self.registry["clumps_prob"],
        )

    def test_clumps_n_range(self):
        """clumps_n_range is now an explicit parameter."""
        self.assertEqual(
            list(self.defaults["clumps_n_range"]),
            self.registry["clumps_n_range"],
        )

    def test_clumps_frac_range(self):
        """clumps_frac_range is now an explicit parameter."""
        self.assertEqual(
            list(self.defaults["clumps_frac_range"]),
            self.registry["clumps_frac_range"],
        )


class TestLensParamsMatchRegistry(unittest.TestCase):
    """Assert sample_lens_params() defaults match injection_priors.yaml."""

    def setUp(self):
        self.registry = _load_registry()["lens_params"]
        self.defaults = _extract_function_defaults(ENGINE_PATH, "sample_lens_params")

    def test_shear_sigma(self):
        self.assertAlmostEqual(
            self.defaults["shear_sigma"],
            self.registry["shear_sigma"],
        )

    def test_center_sigma_arcsec(self):
        self.assertAlmostEqual(
            self.defaults["center_sigma_arcsec"],
            self.registry["center_sigma_arcsec"],
        )

    def test_q_lens_range(self):
        self.assertEqual(
            list(self.defaults["q_lens_range"]),
            self.registry["q_lens_range"],
        )


class TestRegistryCompleteness(unittest.TestCase):
    """Verify the registry covers all relevant defaults."""

    def test_source_params_extracted_keys_match_expected(self):
        """Verify the AST parser extracts ALL expected keys.

        If this fails, either a new parameter was added to the code (update
        this test), or the AST parser is silently skipping a parameter.
        """
        defaults = _extract_function_defaults(ENGINE_PATH, "sample_source_params")
        expected_extracted = {
            "r_mag_range", "beta_frac_range", "re_arcsec_range",
            "n_range", "q_range", "g_minus_r_mu_sigma",
            "r_minus_z_mu_sigma", "clumps_prob",
            "clumps_n_range", "clumps_frac_range",
            "re_scale", "gmr_shift", "rmz_shift",
        }
        self.assertEqual(
            set(defaults.keys()), expected_extracted,
            f"Extracted defaults mismatch. Extra: {set(defaults.keys()) - expected_extracted}, "
            f"Missing: {expected_extracted - set(defaults.keys())}"
        )

    def test_source_params_keys_present(self):
        """All source prior parameters from the registry are checked."""
        registry = _load_registry()["source_params"]
        expected_keys = {
            "r_mag_range", "beta_frac_range", "re_arcsec_range",
            "n_range", "q_range", "g_minus_r_mu_sigma",
            "r_minus_z_mu_sigma", "clumps_prob",
            "clumps_n_range", "clumps_frac_range",
        }
        for key in expected_keys:
            self.assertIn(key, registry,
                          f"Registry missing source_params key: {key}")

    def test_lens_params_keys_present(self):
        """All lens prior parameters from the registry are checked."""
        registry = _load_registry()["lens_params"]
        expected_keys = {"shear_sigma", "center_sigma_arcsec", "q_lens_range"}
        for key in expected_keys:
            self.assertIn(key, registry,
                          f"Registry missing lens_params key: {key}")

    def test_model2_conditioning_keys_present(self):
        """Model 2 conditioning parameters are documented."""
        registry = _load_registry()["model2_conditioning"]
        for key in ["q_scatter", "q_floor"]:
            self.assertIn(key, registry,
                          f"Registry missing model2_conditioning key: {key}")

    def test_preprocessing_keys_present(self):
        """Preprocessing parameters are documented."""
        registry = _load_registry()["preprocessing"]
        for key in ["annulus_r_in", "annulus_r_out", "clip_range"]:
            self.assertIn(key, registry,
                          f"Registry missing preprocessing key: {key}")


if __name__ == "__main__":
    print("=" * 60)
    print("Injection Prior Registry Validation")
    print(f"  Registry: {REGISTRY_PATH}")
    print("=" * 60)
    unittest.main(verbosity=2)
