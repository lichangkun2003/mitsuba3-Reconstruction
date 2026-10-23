"""Transfer model/topology state as device buffers; only structure uses IPC."""

from __future__ import annotations


def broadcast_device_state(value, distributed, collective, mi, dr, *, root=0):
    """Broadcast a nested state containing arrays, metadata and Python scalars.

    Arrays remain on device, including refinement parameters and Adam moments.
    NumPy arrays from an explicitly selected CPU implementation are uploaded once.
    Empty arrays and dictionaries are supported. No voxel values travel in IPC.
    """
    if distributed.world_size == 1:
        # No transfer is needed. In particular, preserve the legacy sampler's
        # NumPy image shapes and integer dtypes when the launcher uses one GPU.
        return value
    source = distributed.rank == root
    arrays, uint_keys = {}, []

    def pack(item, path=()):
        if dr.is_array_v(type(item)) or (hasattr(item, "dtype") and hasattr(item, "shape")):
            if not dr.is_array_v(type(item)) and (
                    len(item.shape) != 1 or item.dtype.kind not in "fiu" or item.dtype.itemsize != 4):
                raise ValueError("Replica buffers must be flat float32/uint32 native arrays")
            key = str(len(arrays))
            integer = (path and path[-1] in ("cell_indices", "steps"))
            if dr.is_array_v(type(item)):
                integer = integer or dr.is_integral_v(type(item))
            elif getattr(item.dtype, "kind", "") in "iu":
                integer = True
            cls = mi.UInt32 if integer else mi.Float
            array = cls(item)
            arrays[key] = array
            if integer:
                uint_keys.append(key)
            return ("array", key, dr.width(array), bool(integer))
        if isinstance(item, dict):
            return ("dict", [(key, pack(child, (*path, key))) for key, child in item.items()])
        if isinstance(item, (tuple, list)):
            return ("tuple" if isinstance(item, tuple) else "list",
                    [pack(child, path) for child in item])
        return ("scalar", item)

    # All control broadcasts originate at rank 0. Model and topology ownership
    # use this rank; NCCL's per-view evidence API separately permits any source.
    if root != 0:
        raise ValueError("Replica state is owned by rank 0")
    description = distributed.broadcast(pack(value) if source else None)

    def allocate(node):
        if node[0] == "array":
            _, key, count, integer = node
            arrays[key] = dr.zeros(mi.UInt32 if integer else mi.Float, count)
            if integer:
                uint_keys.append(key)
        elif node[0] == "dict":
            for _, child in node[1]:
                allocate(child)
        elif node[0] in ("tuple", "list"):
            for child in node[1]:
                allocate(child)

    if not source:
        allocate(description)
    if arrays:
        arrays = collective.broadcast_arrays(arrays, root=root, uint_keys=tuple(uint_keys))

    def unpack(node):
        kind = node[0]
        if kind == "array":
            return arrays[node[1]]
        if kind == "dict":
            return {key: unpack(child) for key, child in node[1]}
        if kind == "tuple":
            return tuple(unpack(child) for child in node[1])
        if kind == "list":
            return [unpack(child) for child in node[1]]
        return node[1]

    return unpack(description)
