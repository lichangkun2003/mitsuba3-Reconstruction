/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions
are met:
  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the distribution.
  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Ported from Falcor_Reconstuction:
Source/RenderPasses/VoxelReconstructionNoLightTransport/PointCloudLoader.h/.cpp
Source commit: cd019690c8328c34f62050d392c9b22a83013a19
The CPU initialization algorithm and input conventions are preserved.
*/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace mitsuba::reconstruction
{
struct Seed
{
    uint32_t voxelIndex;
    std::array<float, 3> center = {0.5f, 0.5f, 0.5f}; // Voxel-local.
    std::array<float, 3> scale = {}; // World-space semi-axes.
    std::array<float, 4> rotation = {1.f, 0.f, 0.f, 0.f}; // w,x,y,z.
    std::array<float, 3> color = {}; // Mean linear RGB; zero when unavailable.
};
static_assert(sizeof(Seed) == 56, "Point-cloud seeds must match the shader buffer layout.");

struct Statistics
{
    uint64_t inputPoints = 0;
    uint64_t invalidPoints = 0;
    uint64_t outsidePoints = 0;
    uint64_t coloredPoints = 0; // Valid RGB points used by plain-cloud initialization.
    // 3DGS input only. Coverage is clipped to the grid, never to a fixed per-Gaussian radius.
    uint64_t droppedByOpacity = 0; // peak opacity below the visibility threshold
    uint64_t testedBlocks = 0;
    uint64_t skippedFullBlocks = 0;
    uint64_t testedCells = 0;     // exact ellipsoid/AABB tests at the boundary
    uint64_t coveredCells = 0;    // newly occupied cells, excluding repeated coverage
    uint64_t maxCoveredCells = 0; // largest number of newly occupied cells from one Gaussian
};

struct Result
{
    std::vector<Seed> seeds;
    Statistics statistics;
};

// Input positions use the NeRF world coordinates of the reference dataset.
// The renderer uses (x, z, -y).
//
// A 3DGS reconstruction carries the per-Gaussian properties (opacity, scale_0..2, rot_0..3), and
// Every voxel whose closed AABB intersects the Gaussian's threshold ellipsoid is occupied.
// This preserves thin walls even when the covered region misses every voxel center.
// Plain clouds fit each occupied cell's points with a padded PCA ellipsoid.
// RGB integer channels are normalized by their type range; float RGB is [0,1].
// RGB is interpreted as sRGB, linearized per point, and averaged per cell.
// Missing/invalid RGB leaves the geometric point valid; missing colors fall back to black.
// `opacityThreshold` is compared against the peak opacity sigmoid(opacity) and must be in (0, 1).
Result load(
    const std::filesystem::path& path,
    const std::array<uint32_t, 3>& voxelCount,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& voxelSize,
    double opacityThreshold
);

// In-memory plain point cloud. Positions are already in the renderer frame;
// colors are sRGB [0,1]. Invalid colors keep the geometric point, as for PLY.
// Uses exactly the same per-cell PCA fitting as the plain PLY loader.
Result load_points(
    const float* positions, const float* colors, size_t pointCount,
    const std::array<uint32_t, 3>& voxelCount,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& voxelSize
);
} // namespace mitsuba::reconstruction
