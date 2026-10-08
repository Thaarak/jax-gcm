"""Model states as NetCDF: a portable, pickle-free archive format.

The project's starting states are saved as pickles of the coupled model's
carry (a nested pytree of arrays). Pickles are tied to the exact code that
wrote them and are unsafe to open from an untrusted source. For the public
archive (Part 18 step 33, Zenodo), every state is written as NetCDF:
- one variable per pytree leaf, named ``v0000``, ``v0001``, ... in
  flattening order;
- each variable's ``path`` attribute holds its key path (for example
  ``['ocn']['state'].sea_surface_temperature``);
- the file's ``paths`` attribute lists all of them, in order.

Complex leaves are stored as ``<name>_real`` and ``<name>_imag``, and
booleans as ``int8``; the ``kind`` attribute records which.

Loading needs a template carry from the same model, as loading a pickle
does. ``load_state`` refuses a file whose key paths or shapes differ from
the template's, so a structural change can never be read silently.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import numpy as np
import xarray as xr


def flatten_state(state) -> Tuple[List[str], List[np.ndarray]]:
    """Return the key paths and leaves of a pytree, in flattening order."""
    pairs, _ = jax.tree_util.tree_flatten_with_path(state)
    return ([jax.tree_util.keystr(p) for p, _ in pairs],
            [np.asarray(leaf) for _, leaf in pairs])


def save_state(path, state, attrs: Dict = None, compress: bool = True):
    """Write a pytree of arrays to NetCDF (one variable per leaf)."""
    paths, leaves = flatten_state(state)
    data, encoding = {}, {}
    for i, (key, leaf) in enumerate(zip(paths, leaves)):
        name = f"v{i:04d}"
        dims = [f"{name}_d{j}" for j in range(leaf.ndim)]
        common = {"path": key, "leaf_dtype": str(leaf.dtype)}
        if np.iscomplexobj(leaf):
            for part, values in (("real", leaf.real), ("imag", leaf.imag)):
                data[f"{name}_{part}"] = xr.Variable(
                    dims, values, attrs={**common, "kind": f"complex_{part}"})
        elif leaf.dtype == np.bool_:
            data[name] = xr.Variable(dims, leaf.astype(np.int8),
                                     attrs={**common, "kind": "bool"})
        else:
            data[name] = xr.Variable(dims, leaf, attrs={**common,
                                                        "kind": "plain"})
    if compress:
        encoding = {k: {"zlib": True, "complevel": 4} for k, v in
                    data.items() if v.ndim > 0}
    ds = xr.Dataset(data, attrs={"paths": json.dumps(paths),
                                 "format": "jax-gcm state, one variable per "
                                           "pytree leaf (jcm.mcb.state_netcdf)",
                                 **{k: json.dumps(v) if isinstance(
                                     v, (dict, list)) else v
                                    for k, v in (attrs or {}).items()}})
    ds.to_netcdf(Path(path), encoding=encoding)


def load_leaves(path) -> Tuple[List[str], List[np.ndarray], Dict]:
    """Return the key paths, leaves and attributes stored in a state file."""
    with xr.open_dataset(Path(path), decode_cf=False,
                         mask_and_scale=False) as ds:
        paths = json.loads(ds.attrs["paths"])
        leaves = []
        for i in range(len(paths)):
            name = f"v{i:04d}"
            if f"{name}_real" in ds:
                re_ = ds[f"{name}_real"].values
                im_ = ds[f"{name}_imag"].values
                dtype = ds[f"{name}_real"].attrs["leaf_dtype"]
                leaves.append((re_ + 1j * im_).astype(dtype))
            else:
                var = ds[name]
                values = var.values
                if var.attrs.get("kind") == "bool":
                    values = values.astype(np.bool_)
                leaves.append(values.astype(var.attrs["leaf_dtype"]))
        attrs = {k: v for k, v in ds.attrs.items() if k != "paths"}
    return paths, leaves, attrs


def load_state(path, template):
    """Rebuild a state from NetCDF, using ``template`` for its structure.

    Raises ValueError if the stored key paths or shapes differ from the
    template's. Dtypes come from the file: a freshly initialized template
    can hold a Python int where a saved state holds float32 (the slab land
    clock, for one), and the saved state's own types are the right ones.
    """
    t_paths, t_leaves = flatten_state(template)
    paths, leaves, _ = load_leaves(path)
    if paths != t_paths:
        raise ValueError("the state's key paths differ from the template's")
    for key, leaf, ref in zip(paths, leaves, t_leaves):
        if leaf.shape != ref.shape:
            raise ValueError(f"{key}: stored shape {leaf.shape}, template "
                             f"{ref.shape}")
    treedef = jax.tree_util.tree_structure(template)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def states_equal(a, b) -> bool:
    """Return whether two pytrees have identical structure and leaves."""
    pa, la = flatten_state(a)
    pb, lb = flatten_state(b)
    return pa == pb and all(x.dtype == y.dtype and x.shape == y.shape
                            and np.array_equal(x, y, equal_nan=True)
                            for x, y in zip(la, lb))
