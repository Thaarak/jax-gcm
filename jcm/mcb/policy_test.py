"""Tests for MCB policy networks (expand_policy_input)."""

import unittest

import jax
import jax.numpy as jnp

from jcm.mcb.policy import MCBPolicyMLP, expand_policy_input


class TestExpandPolicyInput(unittest.TestCase):
    """Tests for zero-padded input-dimension expansion."""

    def setUp(self):
        self.policy = MCBPolicyMLP(
            output_shape=(8, 4), hidden_dims=(16, 16), max_perturbation=0.15
        )
        self.old_dim = 11
        self.new_dim = 13
        self.params = self.policy.init(
            jax.random.PRNGKey(0), jnp.zeros(self.old_dim)
        )

    def test_expanded_kernel_shape(self):
        """hidden_0 kernel gains new_dim - old_dim zero rows."""
        expanded = expand_policy_input(self.params, self.old_dim, self.new_dim)
        kernel = expanded['params']['hidden_0']['kernel']
        self.assertEqual(kernel.shape, (self.new_dim, 16))
        # New rows are exactly zero
        self.assertTrue(jnp.array_equal(
            kernel[self.old_dim:], jnp.zeros((2, 16))
        ))
        # Old rows are untouched
        self.assertTrue(jnp.array_equal(
            kernel[:self.old_dim],
            self.params['params']['hidden_0']['kernel'],
        ))

    def test_exact_equivalence_regardless_of_new_features(self):
        """Expanded policy output is bit-identical to the original for ANY
        values of the appended features (zero rows nullify them).
        """
        expanded = expand_policy_input(self.params, self.old_dim, self.new_dim)
        x = jax.random.normal(jax.random.PRNGKey(1), (self.old_dim,))
        out_old = self.policy.apply(self.params, x)

        for extra in (jnp.zeros(2), jnp.array([3.7, -12.0])):
            x13 = jnp.concatenate([x, extra])
            out_new = self.policy.apply(expanded, x13)
            self.assertTrue(jnp.array_equal(out_old, out_new))

    def test_gradients_reach_new_rows(self):
        """Gradients w.r.t. the new zero rows are nonzero when the new
        features are nonzero, so the policy can learn to use them.
        """
        expanded = expand_policy_input(self.params, self.old_dim, self.new_dim)
        x13 = jnp.concatenate([
            jax.random.normal(jax.random.PRNGKey(2), (self.old_dim,)),
            jnp.array([1.0, -1.0]),
        ])

        def loss(p):
            return jnp.sum(self.policy.apply(p, x13) ** 2)

        grads = jax.grad(loss)(expanded)
        new_row_grads = grads['params']['hidden_0']['kernel'][self.old_dim:]
        self.assertFalse(jnp.allclose(new_row_grads, 0.0))

    def test_rejects_shrinking(self):
        """new_dim < old_dim must raise."""
        with self.assertRaises(ValueError):
            expand_policy_input(self.params, self.old_dim, self.old_dim - 1)

    def test_does_not_mutate_input(self):
        """The original params are left unchanged."""
        before = self.params['params']['hidden_0']['kernel']
        expand_policy_input(self.params, self.old_dim, self.new_dim)
        self.assertEqual(
            self.params['params']['hidden_0']['kernel'].shape,
            before.shape,
        )


if __name__ == "__main__":
    unittest.main()
