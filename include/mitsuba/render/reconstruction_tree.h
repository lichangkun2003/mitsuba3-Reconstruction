#pragma once

#include <mitsuba/core/fwd.h>
#include <drjit/while_loop.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

namespace reconstruction_tree_detail {

/// Interleave up to 21 bits per coordinate. The 64-bit key also supports a
/// padded 2048 grid (33 Morton bits), without changing 32-bit payload IDs.
template <typename UInt>
dr::uint64_array_t<UInt> morton(const UInt &x, const UInt &y, const UInt &z) {
    using Key = dr::uint64_array_t<UInt>;
    auto spread = [](const UInt &input) {
        Key value(input);
        value &= Key(UINT64_C(0x1fffff));
        value = (value | (value << 32)) & Key(UINT64_C(0x1f00000000ffff));
        value = (value | (value << 16)) & Key(UINT64_C(0x1f0000ff0000ff));
        value = (value | (value << 8)) & Key(UINT64_C(0x100f00f00f00f00f));
        value = (value | (value << 4)) & Key(UINT64_C(0x10c30c30c30c30c3));
        value = (value | (value << 2)) & Key(UINT64_C(0x1249249249249249));
        return value;
    };
    return spread(x) | (spread(y) << 1) | (spread(z) << 2);
}

/// Stable GPU radix permutation. Bucket IDs and permutations are UInt32 even
/// when keys use UInt64; no leaf or permutation data is copied to the host.
template <typename UIntBuffer>
UIntBuffer order(const dr::uint64_array_t<UIntBuffer> &keys, uint32_t key_bits) {
    using KeyBuffer = dr::uint64_array_t<UIntBuffer>;
    const size_t count = keys.size();
    if (count <= 1)
        return dr::arange<UIntBuffer>(count);
    if constexpr (dr::is_jit_v<UIntBuffer>) {
        KeyBuffer ordered = keys;
        UIntBuffer result = dr::arange<UIntBuffer>(count);
        constexpr uint32_t radix_bits = UIntBuffer::Backend == JitBackend::LLVM ? 11u : 8u;
        for (uint32_t shift = 0; shift < key_bits; shift += radix_bits) {
            uint32_t bits = std::min(radix_bits, key_bits - shift), buckets = 1u << bits;
            UIntBuffer digit((ordered >> shift) & KeyBuffer(buckets - 1u));
            dr::eval(digit, ordered, result);
            auto *permutation = static_cast<uint32_t *>(
                jit_malloc(UIntBuffer::Backend, count * sizeof(uint32_t)));
            UIntBuffer perm = UIntBuffer::map_(permutation, count, true);
            jit_block_mkperm(UIntBuffer::Backend, digit.data(), uint32_t(count),
                            uint32_t(count), buckets, permutation, nullptr);
            ordered = dr::gather<KeyBuffer>(ordered, perm);
            result = dr::gather<UIntBuffer>(result, perm);
        }
        return result;
    } else {
        std::vector<uint32_t> result(count);
        std::iota(result.begin(), result.end(), 0u);
        std::stable_sort(result.begin(), result.end(),
            [&](uint32_t a, uint32_t b) { return keys.entry(a) < keys.entry(b); });
        return dr::load<UIntBuffer>(result.data(), result.size());
    }
}

template <typename Buffer, typename UIntBuffer>
Buffer flatten(const std::vector<Buffer> &levels, uint32_t total) {
    Buffer result = dr::empty<Buffer>(total);
    uint32_t offset = 0;
    for (const Buffer &level : levels) {
        UIntBuffer indices = dr::arange<UIntBuffer>(level.size()) + offset;
        dr::scatter(result, level, indices);
        offset += uint32_t(level.size());
    }
    dr::eval(result);
    return result;
}

} // namespace reconstruction_tree_detail

/** Compact occupancy hierarchy. Parameter and Adam row order never changes.
 *
 * Internal levels are concatenated from finest to coarsest. Child segments
 * are contiguous and selected with popcount; the finest segment addresses
 * leaf_ids, while other segments address internal nodes. The low eight mask
 * bits describe normal descendants, the next eight hypothetical descendants.
 * A normal tree and an independently replaceable candidate tree can use the
 * same representation without rebuilding the normal hierarchy on overlay edits.
 */
template <typename UIntBuffer, typename IntBuffer>
struct CompactReconstructionTree {
    using KeyBuffer = dr::uint64_array_t<UIntBuffer>;

    IntBuffer leaf_ids = dr::full<IntBuffer>(-1, 1);
    UIntBuffer masks = dr::zeros<UIntBuffer>(1),
               child_base = dr::zeros<UIntBuffer>(1),
               parent = dr::full<UIntBuffer>(UINT32_MAX, 1);
    uint32_t root = 0, depth = 0, padded_resolution = 1,
             node_count = 0, leaf_count = 0, resolution = 1;
    bool mixed_layout = false;

    DRJIT_STRUCT(CompactReconstructionTree, leaf_ids, masks, child_base, parent)

    void clear(uint32_t grid_resolution) {
        if (!grid_resolution || grid_resolution > (1u << 21))
            throw std::runtime_error("Compact reconstruction tree: invalid grid resolution");
        resolution = grid_resolution;
        padded_resolution = 1;
        depth = 0;
        while (padded_resolution < resolution) {
            padded_resolution <<= 1;
            ++depth;
        }
        root = node_count = leaf_count = 0;
        mixed_layout = false;
        leaf_ids = dr::full<IntBuffer>(-1, 1);
        masks = child_base = dr::zeros<UIntBuffer>(1);
        parent = dr::full<UIntBuffer>(UINT32_MAX, 1);
    }

    uint64_t allocated_bytes() const {
        return (uint64_t(leaf_ids.size()) + masks.size() + child_base.size() + parent.size()) * 4u;
    }

    /// Only topology lengths/validation flags cross the host boundary. Build a
    /// replacement first, so malformed input cannot partially overwrite a tree.
    void build(const UIntBuffer &cells, const IntBuffer &payload, uint32_t grid_resolution) {
        if (!grid_resolution || uint64_t(grid_resolution) * grid_resolution * grid_resolution > UINT32_MAX)
            throw std::runtime_error("Compact reconstruction tree: invalid grid resolution");
        CompactReconstructionTree replacement;
        replacement.clear(grid_resolution);
        if (cells.size() != payload.size() || cells.size() > uint64_t(UINT32_MAX))
            throw std::runtime_error("Compact reconstruction tree: invalid leaf array sizes");
        if (!cells.size()) {
            *this = std::move(replacement);
            return;
        }
        const uint32_t total_cells = uint32_t(uint64_t(grid_resolution) * grid_resolution * grid_resolution);
        if (!dr::slice(dr::all((cells < total_cells) & (payload != -1))))
            throw std::runtime_error("Compact reconstruction tree: cell indices must be unique and in bounds; payloads must be nonempty");
        UIntBuffer x = cells % grid_resolution,
                   y = (cells / grid_resolution) % grid_resolution,
                   z = cells / (grid_resolution * grid_resolution);
        KeyBuffer keys = reconstruction_tree_detail::morton(x, y, z);
        UIntBuffer permutation = reconstruction_tree_detail::order<UIntBuffer>(keys, 3u * replacement.depth);
        keys = dr::gather<KeyBuffer>(keys, permutation);
        UIntBuffer ordinal = dr::arange<UIntBuffer>(cells.size());
        KeyBuffer previous = dr::gather<KeyBuffer>(keys, ordinal - 1u, ordinal > 0u);
        if (!dr::slice(dr::all((ordinal == 0u) | (keys != previous))))
            throw std::runtime_error("Compact reconstruction tree: cell indices must be unique and in bounds");
        replacement.leaf_ids = dr::gather<IntBuffer>(payload, permutation);
        replacement.leaf_count = uint32_t(cells.size());
        dr::eval(replacement.leaf_ids);
        if (!replacement.depth) {
            *this = std::move(replacement);
            return;
        }

        std::vector<UIntBuffer> level_masks, level_bases, level_parents;
        UIntBuffer descendant_masks = dr::select(replacement.leaf_ids >= 0,
                                                  UIntBuffer(1u), UIntBuffer(1u << 8));
        uint32_t total_nodes = 0, previous_offset = 0;
        for (uint32_t level = 1; level <= replacement.depth; ++level) {
            const uint32_t child_count = uint32_t(keys.size());
            UIntBuffer child_rows = dr::arange<UIntBuffer>(child_count);
            KeyBuffer parent_keys = keys >> 3,
                      preceding = dr::gather<KeyBuffer>(parent_keys, child_rows - 1u, child_rows > 0u);
            auto starts = (child_rows == 0u) | (parent_keys != preceding);
            UIntBuffer first = dr::compress(starts);
            const uint32_t count = uint32_t(first.size());
            if (uint64_t(total_nodes) + count > UINT32_MAX)
                throw std::runtime_error("Compact reconstruction tree: too many internal nodes");
            UIntBuffer parent_rows = dr::prefix_sum(UIntBuffer(starts), false) - 1u;
            UIntBuffer child_bit = UIntBuffer(1u) << UIntBuffer(keys & KeyBuffer(7u));
            UIntBuffer values = dr::select((descendant_masks & 0xffu) != 0u, child_bit, UIntBuffer(0u)) |
                                dr::select((descendant_masks & 0xff00u) != 0u, child_bit << 8, UIntBuffer(0u));
            UIntBuffer node_masks = dr::zeros<UIntBuffer>(count);
            dr::scatter_reduce(ReduceOp::Or, node_masks, values, parent_rows);
            if (level > 1)
                level_parents.back() = parent_rows + total_nodes;
            UIntBuffer bases = first + (level == 1 ? 0u : previous_offset);
            KeyBuffer node_prefixes = dr::gather<KeyBuffer>(parent_keys, first);
            dr::eval(node_masks, bases, node_prefixes);
            level_masks.push_back(node_masks);
            level_bases.push_back(bases);
            level_parents.push_back(dr::full<UIntBuffer>(UINT32_MAX, count));
            replacement.root = previous_offset = total_nodes;
            total_nodes += count;
            keys = std::move(node_prefixes);
            descendant_masks = std::move(node_masks);
        }
        replacement.node_count = total_nodes;
        replacement.masks = reconstruction_tree_detail::flatten<UIntBuffer, UIntBuffer>(level_masks, total_nodes);
        replacement.child_base = reconstruction_tree_detail::flatten<UIntBuffer, UIntBuffer>(level_bases, total_nodes);
        replacement.parent = reconstruction_tree_detail::flatten<UIntBuffer, UIntBuffer>(level_parents, total_nodes);
        *this = std::move(replacement);
    }

    /** Build nonoverlapping dyadic leaves without expanding coarse cells.
     *
     * Coordinates are minimum corners in finest-grid units; log_size is the
     * base-two logarithm of the leaf side length in those units. A mixed node
     * uses mask bits 16..23 to mark terminal children. Its compact child segment
     * in leaf_ids contains either a payload or an internal node index. Thus
     * parent still indexes internal nodes only, and lookup retains the same
     * nearest-internal-ancestor cache as the uniform tree.
     */
    void build_mixed(const UIntBuffer &x, const UIntBuffer &y, const UIntBuffer &z,
                     const UIntBuffer &log_size, const IntBuffer &payload,
                     uint32_t grid_resolution) {
        CompactReconstructionTree replacement;
        replacement.clear(grid_resolution);
        replacement.mixed_layout = true;
        const size_t count = x.size();
        if (y.size() != count || z.size() != count || log_size.size() != count ||
            payload.size() != count || count > uint64_t(UINT32_MAX))
            throw std::runtime_error("Compact reconstruction tree: invalid mixed leaf array sizes");
        if (!count) {
            *this = std::move(replacement);
            return;
        }
        if (!dr::slice(dr::all(log_size <= replacement.depth)))
            throw std::runtime_error("Compact reconstruction tree: mixed leaf size exceeds the root");
        UIntBuffer side = UIntBuffer(1u) << log_size;
        UIntBuffer alignment = side - 1u;
        if (!dr::slice(dr::all((x < grid_resolution) & (y < grid_resolution) &
                              (z < grid_resolution) &
                              (side <= grid_resolution - x) &
                              (side <= grid_resolution - y) &
                              (side <= grid_resolution - z) &
                              (((x | y | z) & alignment) == 0u) & (payload != -1))))
            throw std::runtime_error("Compact reconstruction tree: mixed leaves must be aligned and in bounds; payloads must be nonempty");

        KeyBuffer leaf_keys = reconstruction_tree_detail::morton(x, y, z);
        UIntBuffer permutation = reconstruction_tree_detail::order<UIntBuffer>(leaf_keys, 3u * replacement.depth);
        leaf_keys = dr::gather<KeyBuffer>(leaf_keys, permutation);
        UIntBuffer leaf_sizes = dr::gather<UIntBuffer>(log_size, permutation);
        IntBuffer payloads = dr::gather<IntBuffer>(payload, permutation);
        UIntBuffer ordinal = dr::arange<UIntBuffer>(count);
        KeyBuffer ends = leaf_keys + (KeyBuffer(1u) << KeyBuffer(3u * leaf_sizes));
        KeyBuffer previous_end = dr::gather<KeyBuffer>(ends, ordinal - 1u, ordinal > 0u);
        if (!dr::slice(dr::all((ordinal == 0u) | (leaf_keys >= previous_end))))
            throw std::runtime_error("Compact reconstruction tree: mixed leaves must not overlap");
        replacement.leaf_count = uint32_t(count);
        dr::eval(leaf_keys, leaf_sizes, payloads);

        std::vector<IntBuffer> edge_levels;
        std::vector<UIntBuffer> level_masks, level_bases, level_parents;
        KeyBuffer previous_keys;
        UIntBuffer previous_masks;
        uint32_t previous_count = 0, previous_offset = 0, total_nodes = 0, total_edges = 0;
        for (uint32_t level = 0; level <= replacement.depth; ++level) {
            UIntBuffer selected = dr::compress(leaf_sizes == level);
            const uint32_t terminal_count = uint32_t(selected.size());
            if (!terminal_count && !previous_count)
                continue;
            const uint64_t combined_count = uint64_t(terminal_count) + previous_count;
            if (combined_count > UINT32_MAX)
                throw std::runtime_error("Compact reconstruction tree: too many mixed child entries");
            const uint32_t child_count = uint32_t(combined_count);
            KeyBuffer keys;
            IntBuffer values;
            UIntBuffer descendants, terminal;
            if (terminal_count) {
                keys = dr::gather<KeyBuffer>(leaf_keys, selected) >> (3u * level);
                values = dr::gather<IntBuffer>(payloads, selected);
                descendants = dr::select(values >= 0, UIntBuffer(1u), UIntBuffer(1u << 8));
                terminal = dr::full<UIntBuffer>(1u, terminal_count);
            } else {
                keys = previous_keys;
                values = IntBuffer(dr::arange<UIntBuffer>(previous_count) + previous_offset);
                descendants = previous_masks;
                terminal = dr::zeros<UIntBuffer>(previous_count);
            }
            if (previous_count && terminal_count) {
                // Both inputs are sorted. A radix permutation is only needed
                // at levels where terminal leaves join an existing subtree.
                keys = dr::concat(keys, previous_keys);
                values = dr::concat(values, IntBuffer(dr::arange<UIntBuffer>(previous_count) + previous_offset));
                descendants = dr::concat(descendants, previous_masks);
                terminal = dr::concat(terminal, dr::zeros<UIntBuffer>(previous_count));
                UIntBuffer sorted = reconstruction_tree_detail::order<UIntBuffer>(keys, 3u * (replacement.depth - level));
                keys = dr::gather<KeyBuffer>(keys, sorted);
                values = dr::gather<IntBuffer>(values, sorted);
                descendants = dr::gather<UIntBuffer>(descendants, sorted);
                terminal = dr::gather<UIntBuffer>(terminal, sorted);
            }
            if (level == replacement.depth) {
                if (terminal_count) {
                    // A single root-sized leaf has no internal ancestors.
                    replacement.leaf_ids = values;
                    dr::eval(replacement.leaf_ids);
                }
                break;
            }

            UIntBuffer child_rows = dr::arange<UIntBuffer>(child_count);
            KeyBuffer parent_keys = keys >> 3,
                      preceding = dr::gather<KeyBuffer>(parent_keys, child_rows - 1u, child_rows > 0u);
            auto starts = (child_rows == 0u) | (parent_keys != preceding);
            UIntBuffer first = dr::compress(starts);
            const uint32_t node_count = uint32_t(first.size());
            if (uint64_t(total_nodes) + node_count > UINT32_MAX ||
                uint64_t(total_edges) + child_count > UINT32_MAX)
                throw std::runtime_error("Compact reconstruction tree: too many mixed nodes or edges");
            UIntBuffer parent_rows = dr::prefix_sum(UIntBuffer(starts), false) - 1u;
            UIntBuffer child_bit = UIntBuffer(1u) << UIntBuffer(keys & KeyBuffer(7u));
            UIntBuffer flags = dr::select((descendants & 0xffu) != 0u, child_bit, UIntBuffer(0u)) |
                               dr::select((descendants & 0xff00u) != 0u, child_bit << 8, UIntBuffer(0u)) |
                               dr::select(terminal != 0u, child_bit << 16, UIntBuffer(0u));
            UIntBuffer node_masks = dr::zeros<UIntBuffer>(node_count);
            dr::scatter_reduce(ReduceOp::Or, node_masks, flags, parent_rows);
            if (previous_count) {
                UIntBuffer links = dr::empty<UIntBuffer>(previous_count);
                dr::scatter(links, parent_rows + total_nodes,
                            UIntBuffer(values) - previous_offset, terminal == 0u);
                dr::eval(links);
                level_parents.back() = std::move(links);
            }
            UIntBuffer bases = first + total_edges;
            previous_keys = dr::gather<KeyBuffer>(parent_keys, first);
            dr::eval(values, node_masks, bases, previous_keys);
            edge_levels.push_back(values);
            level_masks.push_back(node_masks);
            level_bases.push_back(bases);
            level_parents.push_back(dr::full<UIntBuffer>(UINT32_MAX, node_count));
            replacement.root = previous_offset = total_nodes;
            previous_count = node_count;
            total_nodes += node_count;
            total_edges += child_count;
            previous_masks = node_masks & 0xffffu;
        }
        replacement.node_count = total_nodes;
        if (total_nodes) {
            replacement.leaf_ids = reconstruction_tree_detail::flatten<IntBuffer, UIntBuffer>(edge_levels, total_edges);
            replacement.masks = reconstruction_tree_detail::flatten<UIntBuffer, UIntBuffer>(level_masks, total_nodes);
            replacement.child_base = reconstruction_tree_detail::flatten<UIntBuffer, UIntBuffer>(level_bases, total_nodes);
            replacement.parent = reconstruction_tree_detail::flatten<UIntBuffer, UIntBuffer>(level_parents, total_nodes);
        }
        *this = std::move(replacement);
    }

    /// Look up the current DDA cell, retaining its nearest internal ancestor.
    /// A miss returns the largest missing child containing this cell. The caller
    /// keeps its existing boundary/tie handling when advancing over that block.
    template <typename UInt, typename Int>
    std::tuple<Int, UInt> lookup(const Int &x, const Int &y, const Int &z,
                                UInt &node, UInt &level,
                                dr::uint64_array_t<UInt> &anchor,
                                const dr::mask_t<UInt> &active) const {
        using Mask = dr::mask_t<UInt>;
        using Key = dr::uint64_array_t<UInt>;
        Mask live = active & (x >= 0) & (y >= 0) & (z >= 0) &
                    (x < int32_t(resolution)) & (y < int32_t(resolution)) & (z < int32_t(resolution));
        if (!leaf_count)
            return {Int(-1), dr::select(live, UInt(depth), UInt(0u))};
        if (mixed_layout && !node_count) {
            node = dr::select(live, UInt(0u), node);
            level = dr::select(live, UInt(depth), level);
            anchor = dr::select(live, Key(0u), anchor);
            return {dr::select(live, dr::gather<Int>(leaf_ids, UInt(0u), live), Int(-1)),
                    dr::select(live, UInt(depth), UInt(0u))};
        }
        if (!depth) {
            node = dr::select(live, UInt(0u), node);
            level = dr::select(live, UInt(0u), level);
            anchor = dr::select(live, Key(0u), anchor);
            return {dr::select(live, dr::gather<Int>(leaf_ids, UInt(0u), live), Int(-1)), UInt(0u)};
        }
        // The cached ancestor contains the previous query cell. XOR of the
        // three coordinates identifies their common ancestor without loading
        // a node prefix or computing a Morton key during traversal.
        Key packed = Key(UInt(x)) | (Key(UInt(y)) << depth) | (Key(UInt(z)) << (2u * depth));
        Key difference = packed ^ anchor;
        UInt changes = UInt(difference | (difference >> depth) | (difference >> (2u * depth))) &
                       ((uint32_t(1u) << depth) - 1u);
        Mask reset = live & ((node >= node_count) | (level == 0u) | (level > depth));
        UInt current_node = dr::select(reset, UInt(root), node),
             current_level = dr::select(reset, UInt(depth), level);
        dr::tie(current_node, current_level) = dr::while_loop(
            dr::make_tuple(current_node, current_level),
            [&](const UInt &n, const UInt &l) -> Mask {
                DRJIT_MARK_USED(n);
                return live & (l < depth) & ((changes >> l) != 0u);
            }, [&](UInt &n, UInt &l) {
                Mask ascend = live & (l < depth) & ((changes >> l) != 0u);
                n = dr::select(ascend, dr::gather<UInt>(parent, n, ascend), n);
                l += dr::select(ascend, UInt(1u), UInt(0u));
            }, "Compact reconstruction tree ascend");
        Int result(-1);
        UInt skip(0u);
        Mask running = live;
        dr::tie(current_node, current_level, result, skip, running) = dr::while_loop(
            dr::make_tuple(current_node, current_level, result, skip, running),
            [](const UInt &, const UInt &, const Int &, const UInt &, const Mask &run) -> Mask { return run; },
            [&](UInt &n, UInt &l, Int &value, UInt &missing_level, Mask &run) {
                UInt flags = dr::gather<UInt>(masks, n, run);
                UInt bits = (flags | (flags >> 8)) & 0xffu;
                UInt shift = l - 1u;
                UInt child = ((UInt(x) >> shift) & 1u) |
                             (((UInt(y) >> shift) & 1u) << 1) |
                             (((UInt(z) >> shift) & 1u) << 2);
                UInt bit = UInt(1u) << child;
                Mask occupied = run & ((bits & bit) != 0u);
                missing_level = dr::select(run & !occupied, l - 1u, missing_level);
                UInt address = dr::gather<UInt>(child_base, n, occupied) + dr::popcnt(bits & (bit - 1u));
                Mask leaf = occupied & (mixed_layout ? ((flags & (bit << 16)) != 0u) : (l == 1u));
                Int entry = dr::gather<Int>(leaf_ids, address, mixed_layout ? occupied : leaf);
                value = dr::select(leaf, entry, value);
                missing_level = dr::select(leaf, l - 1u, missing_level);
                Mask descend = occupied & !leaf;
                n = dr::select(descend, mixed_layout ? UInt(entry) : address, n);
                l -= dr::select(descend, UInt(1u), UInt(0u));
                run = descend;
            }, "Compact reconstruction tree descend");
        node = dr::select(live, current_node, node);
        level = dr::select(live, current_level, level);
        anchor = dr::select(live, packed, anchor);
        return {result, skip};
    }
};

NAMESPACE_END(mitsuba)
