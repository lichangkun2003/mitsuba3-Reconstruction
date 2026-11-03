/* Standalone compact-tree regression. The default path uses host arrays only;
 * pass --cuda explicitly to exercise GPU construction and cached lookup.
 * Compile against the checkout's Dr.Jit headers and drjit-core library.
 */
#include <drjit/array.h>
#include <drjit/dynamic.h>
#include <drjit/jit.h>
#include <mitsuba/render/reconstruction_tree.h>
#include <array>
#include <iostream>
#include <random>
#include <string>

namespace dr = drjit;
struct Leaf { uint32_t x, y, z, log_size; int32_t payload; };
using Point = std::array<int32_t, 3>;

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::pair<int32_t, uint32_t> oracle(const std::vector<Leaf> &leaves,
                                  const Point &point, uint32_t resolution) {
    if (point[0] < 0 || point[1] < 0 || point[2] < 0 ||
        uint32_t(point[0]) >= resolution || uint32_t(point[1]) >= resolution ||
        uint32_t(point[2]) >= resolution)
        return {-1, 0};
    for (const Leaf &leaf : leaves) {
        uint32_t side = 1u << leaf.log_size;
        if (uint32_t(point[0]) >= leaf.x && uint32_t(point[0]) < leaf.x + side &&
            uint32_t(point[1]) >= leaf.y && uint32_t(point[1]) < leaf.y + side &&
            uint32_t(point[2]) >= leaf.z && uint32_t(point[2]) < leaf.z + side)
            return {leaf.payload, leaf.log_size};
    }
    uint32_t depth = 0;
    while ((1u << depth) < resolution)
        ++depth;
    for (int log_size = int(depth); log_size >= 0; --log_size) {
        uint32_t side = 1u << log_size;
        uint32_t x = uint32_t(point[0]) & ~(side-1),
                 y = uint32_t(point[1]) & ~(side-1),
                 z = uint32_t(point[2]) & ~(side-1);
        bool intersects = false;
        for (const Leaf &leaf : leaves) {
            uint32_t other_side = 1u << leaf.log_size;
            intersects |= x < leaf.x + other_side && leaf.x < x + side &&
                          y < leaf.y + other_side && leaf.y < y + side &&
                          z < leaf.z + other_side && leaf.z < z + side;
        }
        if (!intersects)
            return {-1, uint32_t(log_size)};
    }
    throw std::runtime_error("Independent oracle found no empty block");
}

template <typename UIntBuffer>
void build(mitsuba::CompactReconstructionTree<UIntBuffer, dr::int32_array_t<UIntBuffer>> &tree,
           const std::vector<Leaf> &leaves, uint32_t resolution) {
    using IntBuffer = dr::int32_array_t<UIntBuffer>;
    std::vector<uint32_t> x, y, z, sizes;
    std::vector<int32_t> values;
    for (const Leaf &leaf : leaves) {
        x.push_back(leaf.x); y.push_back(leaf.y); z.push_back(leaf.z);
        sizes.push_back(leaf.log_size); values.push_back(leaf.payload);
    }
    tree.build_mixed(dr::load<UIntBuffer>(x.data(), x.size()),
                     dr::load<UIntBuffer>(y.data(), y.size()),
                     dr::load<UIntBuffer>(z.data(), z.size()),
                     dr::load<UIntBuffer>(sizes.data(), sizes.size()),
                     dr::load<IntBuffer>(values.data(), values.size()), resolution);
}

template <typename UIntBuffer>
void check(const std::vector<Leaf> &leaves, uint32_t resolution, bool exhaustive) {
    using IntBuffer = dr::int32_array_t<UIntBuffer>;
    using KeyBuffer = dr::uint64_array_t<UIntBuffer>;
    mitsuba::CompactReconstructionTree<UIntBuffer, IntBuffer> tree;
    build(tree, leaves, resolution);
    require(tree.leaf_count == leaves.size(), "Coarse leaves were expanded");
    require(tree.node_count <= leaves.size()*21, "Internal hierarchy expanded with grid volume");
    require(tree.allocated_bytes() <= 32 + 400*leaves.size(), "Mixed index is not compact");
    std::vector<Point> points;
    if (exhaustive) {
        for (int32_t z = -1; z <= int32_t(resolution); ++z)
            for (int32_t y = -1; y <= int32_t(resolution); ++y)
                for (int32_t x = -1; x <= int32_t(resolution); ++x)
                    points.push_back({x,y,z});
    } else {
        std::mt19937 rng(572114);
        for (size_t i = 0; i < 2048; ++i)
            points.push_back({int32_t(rng()%resolution), int32_t(rng()%resolution), int32_t(rng()%resolution)});
        for (const Leaf &leaf : leaves) {
            const int32_t side = 1u << leaf.log_size;
            for (int a : {-1, 0, 1})
                for (int b : {-1, 0, 1})
                    for (int c : {-1, 0, 1})
                        points.push_back({int32_t(leaf.x) + (a < 0 ? -1 : a == 0 ? side/2 : side),
                                          int32_t(leaf.y) + (b < 0 ? -1 : b == 0 ? side/2 : side),
                                          int32_t(leaf.z) + (c < 0 ? -1 : c == 0 ? side/2 : side)});
        }
    }
    UIntBuffer cached_node = dr::full<UIntBuffer>(UINT32_MAX, points.size()),
               cached_level = dr::zeros<UIntBuffer>(points.size());
    KeyBuffer cached_anchor = dr::zeros<KeyBuffer>(points.size());
    // Reuse caches across unrelated points, reverse direction, fine/coarse
    // transitions, absent blocks, and queries outside the root.
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<int32_t> x, y, z;
        std::vector<int32_t> expected_ids;
        std::vector<uint32_t> expected_sizes;
        for (const Point &point : points) {
            x.push_back(point[0]); y.push_back(point[1]); z.push_back(point[2]);
            auto expected = oracle(leaves, point, resolution);
            expected_ids.push_back(expected.first); expected_sizes.push_back(expected.second);
        }
        auto [ids, sizes] = tree.lookup(dr::load<IntBuffer>(x.data(), x.size()),
            dr::load<IntBuffer>(y.data(), y.size()), dr::load<IntBuffer>(z.data(), z.size()),
            cached_node, cached_level, cached_anchor, dr::full<dr::mask_t<UIntBuffer>>(true, points.size()));
        require(dr::slice(dr::all(ids == dr::load<IntBuffer>(expected_ids.data(), expected_ids.size()))),
                "Mixed lookup payload differs from independent AABB oracle");
        require(dr::slice(dr::all(sizes == dr::load<UIntBuffer>(expected_sizes.data(), expected_sizes.size()))),
                "Mixed lookup occupied size or empty skip differs from independent AABB oracle");
        std::rotate(points.begin(), points.begin()+points.size()/3, points.end());
        std::reverse(points.begin(), points.end());
    }
}

std::vector<Leaf> shells(uint32_t core_resolution, uint32_t shell_count) {
    uint32_t resolution = core_resolution << shell_count;
    std::vector<Leaf> leaves;
    for (uint32_t shell = 1; shell <= shell_count; ++shell) {
        uint32_t outer = core_resolution << shell, side = outer/4,
                 origin = (resolution - outer)/2, log_size = 0;
        while ((1u << log_size) < side)
            ++log_size;
        for (uint32_t z = 0; z < 4; ++z)
            for (uint32_t y = 0; y < 4; ++y)
                for (uint32_t x = 0; x < 4; ++x) {
                    if (x >= 1 && x <= 2 && y >= 1 && y <= 2 && z >= 1 && z <= 2)
                        continue;
                    leaves.push_back({origin+x*side, origin+y*side, origin+z*side,
                                      log_size, int32_t(leaves.size())+100});
                }
    }
    uint32_t minimum = (resolution - core_resolution)/2;
    for (uint32_t z : {minimum, minimum+core_resolution-1})
        for (uint32_t y : {minimum, minimum+core_resolution-1})
            for (uint32_t x : {minimum, minimum+core_resolution-1})
                leaves.push_back({x,y,z,0,int32_t(leaves.size())+100});
    std::reverse(leaves.begin(), leaves.end());
    return leaves;
}

template <typename UIntBuffer> void suite() {
    using IntBuffer = dr::int32_array_t<UIntBuffer>;
    using Tree = mitsuba::CompactReconstructionTree<UIntBuffer, IntBuffer>;
    check<UIntBuffer>({}, 32, true);
    check<UIntBuffer>({{0,0,0,5,37}}, 32, true);
    check<UIntBuffer>({{0,0,0,0,0}}, 1, true);
    check<UIntBuffer>({{16,16,16,4,7},{0,0,0,3,44},{8,0,0,0,3},
                       {9,0,0,0,5},{8,8,8,2,-7},{31,0,31,0,102}}, 32, true);
    check<UIntBuffer>(shells(256, 8), 256u << 8, false);
    check<UIntBuffer>(shells(1024, 8), 1024u << 8, false);
    check<UIntBuffer>(shells(1024, 10), 1u << 20, false);
    check<UIntBuffer>({{0,0,0,20,31},{(1u<<21)-1,(1u<<21)-1,(1u<<21)-1,0,-4}}, 1u<<21, false);
    check<UIntBuffer>({{0,0,0,21,91}}, 1u<<21, false);

    Tree tree;
    build(tree, {{0,0,0,2,7}}, 32);
    for (const std::vector<Leaf> &bad : std::vector<std::vector<Leaf>>{
            {{0,0,0,3,1},{1,1,1,0,2}}, {{0,0,0,0,1},{0,0,0,0,2}},
            {{1,0,0,3,1}}, {{32,0,0,0,1}}, {{0,0,0,6,1}}, {{0,0,0,0,-1}}}) {
        bool rejected = false;
        try { build(tree, bad, 32); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "Malformed mixed tree accepted");
        require(tree.leaf_count == 1 && tree.resolution == 32, "Invalid rebuild changed the existing tree");
    }
    // The original flat-cell builder remains an independent supported path.
    std::vector<uint32_t> cells{0,31,32*32*32-1};
    std::vector<int32_t> ids{7,-5,11};
    tree.build(dr::load<UIntBuffer>(cells.data(), cells.size()),
               dr::load<IntBuffer>(ids.data(), ids.size()), 32);
    require(!tree.mixed_layout && tree.leaf_count == 3, "Uniform build did not reset mixed layout");
}

int main(int argc, char **argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--cuda") {
            jit_set_log_level_stderr(LogLevel::Warn);
            jit_init(1u << uint32_t(JitBackend::CUDA));
            suite<dr::CUDAArray<uint32_t>>();
            jit_shutdown(0);
            std::cout << "mixed tree CUDA construction and lookup passed\n";
        } else {
            suite<dr::DynamicArray<uint32_t>>();
            std::cout << "mixed tree CPU construction and lookup passed\n";
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
