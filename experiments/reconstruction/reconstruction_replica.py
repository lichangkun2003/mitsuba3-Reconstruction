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
    arrays, uint_keys, cell_keys = {}, [], []
    array_types = {"float32": mi.Float, "uint32": mi.UInt32, "uint64": mi.UInt64}

    def pack(item, path=()):
        if dr.is_array_v(type(item)) or (hasattr(item, "dtype") and hasattr(item, "shape")):
            if dr.is_array_v(type(item)):
                kind = dr.type_v(type(item))
                if kind == dr.VarType.Float32:
                    dtype = "float32"
                elif kind in (dr.VarType.UInt32, dr.VarType.Int32):
                    dtype = "uint32"
                elif kind in (dr.VarType.UInt64, dr.VarType.Int64):
                    dtype = "uint64"
                else:
                    raise ValueError("Replica buffers require float32 or 32/64-bit integer elements")
            else:
                if (len(item.shape) != 1 or
                        not (item.dtype.kind == "f" and item.dtype.itemsize == 4 or
                             item.dtype.kind in "iu" and item.dtype.itemsize in (4, 8))):
                    raise ValueError("Replica buffers must be flat float32/uint32/uint64 native arrays")
                dtype = "float32" if item.dtype.kind == "f" else f"uint{8 * item.dtype.itemsize}"
            # Older CPU checkpoints may still contain 32-bit core IDs. Promote
            # those named addresses while preserving every other integer width.
            if path and path[-1] == "cell_indices":
                if dtype == "float32":
                    raise ValueError("Replica cell_indices must contain integer addresses")
                dtype = "uint64"
            key = str(len(arrays))
            array = array_types[dtype](item)
            arrays[key] = array
            if dtype == "uint64":
                cell_keys.append(key)
            elif dtype == "uint32":
                uint_keys.append(key)
            return ("array", key, dr.width(array), dtype)
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
            _, key, count, dtype = node
            arrays[key] = dr.zeros(array_types[dtype], count)
            if dtype == "uint64":
                cell_keys.append(key)
            elif dtype == "uint32":
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
        arrays = collective.broadcast_arrays(arrays, root=root, uint_keys=tuple(uint_keys),
                                              cell_keys=tuple(cell_keys))

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
