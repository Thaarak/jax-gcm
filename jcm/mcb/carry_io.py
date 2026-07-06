"""Serialization for coupled carry states.

Coupled carries contain tree_math structs whose treedefs are fragile to
pickle across code versions. We therefore serialize ONLY the pytree leaves
(as host numpy arrays) and reconstruct the tree at load time from a template
carry with the same structure (e.g. a fresh `coupler.initialize()`).

Example usage:
    from jcm.mcb.carry_io import save_carry, load_carry

    save_carry(carry, "ic_00_carry.pkl")
    template = coupler.initialize()
    carry = load_carry("ic_00_carry.pkl", template)
"""

import pickle

import jax


def save_carry(carry: dict, path: str) -> None:
    """Save a coupled carry's pytree leaves to disk.

    Only the leaves (moved to host memory) are pickled — never treedefs,
    which for tree_math structs are the main pickling risk.

    Args:
        carry: Coupled carry pytree (e.g. from coupler.initialize() or a
            spun-up simulation).
        path: Output pickle path.

    """
    leaves = jax.tree_util.tree_leaves(carry)
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(leaves), f)


def load_carry(path: str, template_carry: dict) -> dict:
    """Load a coupled carry saved by save_carry.

    Args:
        path: Pickle path written by save_carry.
        template_carry: A carry with the SAME pytree structure (e.g. a fresh
            coupler.initialize()); provides the treedef for reconstruction.

    Returns:
        Coupled carry with the saved leaf values and the template's
        structure.

    """
    with open(path, 'rb') as f:
        leaves = pickle.load(f)
    treedef = jax.tree_util.tree_structure(template_carry)
    if treedef.num_leaves != len(leaves):
        raise ValueError(
            f"Leaf count mismatch: file {path} has {len(leaves)} leaves, "
            f"template has {treedef.num_leaves}. The template carry must "
            f"come from the same model configuration used to save."
        )
    return jax.tree_util.tree_unflatten(treedef, leaves)
