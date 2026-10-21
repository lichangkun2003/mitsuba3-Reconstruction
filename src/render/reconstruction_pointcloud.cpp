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

#include "reconstruction_pointcloud.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mitsuba::reconstruction
{
namespace
{
enum class ScalarType { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 };

struct Property
{
    std::string name;
    ScalarType valueType;
    bool isList = false;
    ScalarType countType = ScalarType::UInt8;
};

struct Element
{
    std::string name;
    uint64_t count;
    std::vector<Property> properties;
};

constexpr size_t kMissingProperty = std::numeric_limits<size_t>::max();

struct Header
{
    bool binary = false;
    std::vector<Element> elements;
    size_t vertexElement = 0;
    std::array<size_t, 3> positions;
    std::array<size_t, 3> colors = {kMissingProperty, kMissingProperty, kMissingProperty};
    // Optional per-Gaussian properties of a 3DGS reconstruction. Only when all of them are
    // present does the ellipsoid decide occupancy.
    size_t opacity = kMissingProperty;
    std::array<size_t, 3> scales = {kMissingProperty, kMissingProperty, kMissingProperty};
    std::array<size_t, 4> rotations = {kMissingProperty, kMissingProperty, kMissingProperty, kMissingProperty};
    bool hasGaussianProperties = false;
};

// Skip fully occupied blocks instead of clipping a Gaussian's physical coverage.
constexpr uint32_t kCoverageBlockSize = 8;

struct GaussianCoverage
{
    double covariance[3][3] = {};
    double worldToUnit[3][3] = {};
    double threshold = 0;

    double squaredDistance(const std::array<double, 3>& point) const
    {
        double result = 0;
        for (size_t axis = 0; axis < 3; ++axis)
        {
            double component = 0;
            for (size_t i = 0; i < 3; ++i) component += worldToUnit[axis][i] * point[i];
            result += component * component;
        }
        return result;
    }

    // Interval bounds in principal coordinates: -1 = outside, 1 = wholly inside,
    // 0 = uncertain. These bounds remain conservative for rotated, thin Gaussians.
    int classifyBox(const std::array<double, 3>& lower, const std::array<double, 3>& upper) const
    {
        double minimum = 0, maximum = 0;
        for (size_t axis = 0; axis < 3; ++axis)
        {
            double center = 0, extent = 0;
            for (size_t i = 0; i < 3; ++i)
            {
                center += worldToUnit[axis][i] * (lower[i] + upper[i]) * 0.5;
                extent += std::abs(worldToUnit[axis][i]) * (upper[i] - lower[i]) * 0.5;
            }
            const double error = 64 * std::numeric_limits<double>::epsilon() * (std::abs(center) + extent + 1);
            const double nearest = std::max(0.0, std::abs(center) - extent - error);
            const double farthest = std::abs(center) + extent + error;
            minimum += nearest * nearest;
            maximum += farthest * farthest;
        }
        const double tolerance = 1e-10 * std::max(1.0, threshold);
        if (minimum > threshold + tolerance) return -1;
        if (maximum <= threshold) return 1;
        return 0;
    }

    // Minimize q^T Sigma^-1 q over a closed AABB. If the origin is outside,
    // the minimum lies on a face, an edge, or a corner. Six face projections
    // plus twelve clamped edge projections cover all of these cases without
    // inverting an ill-conditioned covariance matrix.
    bool intersectsBox(const std::array<double, 3>& lower, const std::array<double, 3>& upper) const
    {
        bool containsOrigin = true;
        std::array<double, 3> nearest = {};
        for (size_t axis = 0; axis < 3; ++axis)
        {
            containsOrigin = containsOrigin && lower[axis] <= 0 && upper[axis] >= 0;
            nearest[axis] = std::clamp(0.0, lower[axis], upper[axis]);
        }
        const double tolerance = 1e-10 * std::max(1.0, threshold);
        if (containsOrigin || squaredDistance(nearest) <= threshold + tolerance) return true;

        for (size_t axis = 0; axis < 3; ++axis)
            for (size_t side = 0; side < 2; ++side)
            {
                const double bound = side == 0 ? lower[axis] : upper[axis];
                const double variance = covariance[axis][axis];
                if (!(variance > 0)) continue;
                if (bound * bound / variance > threshold + tolerance) continue;
                bool onFace = true;
                for (size_t i = 0; i < 3; ++i)
                {
                    if (i == axis) continue;
                    const double coordinate = covariance[i][axis] * (bound / variance);
                    const double error = 64 * std::numeric_limits<double>::epsilon() *
                        std::max({1.0, std::abs(coordinate), std::abs(lower[i]), std::abs(upper[i])});
                    onFace = onFace && coordinate >= lower[i] - error && coordinate <= upper[i] + error;
                }
                if (onFace) return true;
            }

        for (size_t freeAxis = 0; freeAxis < 3; ++freeAxis)
        {
            const size_t a = (freeAxis + 1) % 3;
            const size_t b = (freeAxis + 2) % 3;
            for (size_t sideA = 0; sideA < 2; ++sideA)
                for (size_t sideB = 0; sideB < 2; ++sideB)
                {
                    std::array<double, 3> point = {};
                    point[a] = sideA == 0 ? lower[a] : upper[a];
                    point[b] = sideB == 0 ? lower[b] : upper[b];
                    double numerator = 0, denominator = 0;
                    for (size_t i = 0; i < 3; ++i)
                    {
                        const double component = worldToUnit[i][a] * point[a] + worldToUnit[i][b] * point[b];
                        numerator += component * worldToUnit[i][freeAxis];
                        denominator += worldToUnit[i][freeAxis] * worldToUnit[i][freeAxis];
                    }
                    if (!(denominator > 0)) continue;
                    point[freeAxis] = std::clamp(-numerator / denominator, lower[freeAxis], upper[freeAxis]);
                    if (squaredDistance(point) <= threshold + tolerance) return true;
                }
        }
        return false;
    }
};

bool anyMissing(const size_t* indices, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        if (indices[i] == kMissingProperty) return true;
    return false;
}

// Rotation matrix of a (w, x, y, z) quaternion, the convention of gaussianRotationMatrix in the
// shaders. A degenerate quaternion falls back to the identity.
void makeRotation(const std::array<double, 4>& quaternion, double matrix[3][3])
{
    double norm = 0;
    for (size_t i = 0; i < 4; ++i) norm += quaternion[i] * quaternion[i];
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm < 1e-12)
    {
        for (size_t row = 0; row < 3; ++row)
            for (size_t column = 0; column < 3; ++column) matrix[row][column] = row == column ? 1.0 : 0.0;
        return;
    }
    const double w = quaternion[0] / norm;
    const double x = quaternion[1] / norm;
    const double y = quaternion[2] / norm;
    const double z = quaternion[3] / norm;
    matrix[0][0] = 1.0 - 2.0 * (y * y + z * z);
    matrix[0][1] = 2.0 * (x * y - z * w);
    matrix[0][2] = 2.0 * (x * z + y * w);
    matrix[1][0] = 2.0 * (x * y + z * w);
    matrix[1][1] = 1.0 - 2.0 * (x * x + z * z);
    matrix[1][2] = 2.0 * (y * z - x * w);
    matrix[2][0] = 2.0 * (x * z - y * w);
    matrix[2][1] = 2.0 * (y * z + x * w);
    matrix[2][2] = 1.0 - 2.0 * (x * x + y * y);
}

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error("Point-cloud PLY: " + message);
}

ScalarType scalarType(const std::string& name)
{
    if (name == "char" || name == "int8") return ScalarType::Int8;
    if (name == "uchar" || name == "uint8") return ScalarType::UInt8;
    if (name == "short" || name == "int16") return ScalarType::Int16;
    if (name == "ushort" || name == "uint16") return ScalarType::UInt16;
    if (name == "int" || name == "int32") return ScalarType::Int32;
    if (name == "uint" || name == "uint32") return ScalarType::UInt32;
    if (name == "float" || name == "float32") return ScalarType::Float32;
    if (name == "double" || name == "float64") return ScalarType::Float64;
    fail("unsupported property type '" + name + "'.");
}

bool isInteger(ScalarType type)
{
    return type != ScalarType::Float32 && type != ScalarType::Float64;
}

size_t scalarBytes(ScalarType type)
{
    switch (type)
    {
    case ScalarType::Int8: case ScalarType::UInt8: return 1;
    case ScalarType::Int16: case ScalarType::UInt16: return 2;
    case ScalarType::Int32: case ScalarType::UInt32: case ScalarType::Float32: return 4;
    case ScalarType::Float64: return 8;
    }
    fail("invalid scalar type.");
}

uint64_t parseCount(const std::string& text)
{
    uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
        fail("invalid element count '" + text + "'.");
    return value;
}

void requireEnd(std::istringstream& line)
{
    std::string extra;
    if (line >> extra) fail("unexpected token in the header: '" + extra + "'.");
}

Header readHeader(std::ifstream& input, uint64_t fileBytes)
{
    Header header;
    std::string line;
    if (!std::getline(input, line)) fail("empty file.");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "ply") fail("missing PLY magic.");

    bool hasFormat = false;
    bool hasEnd = false;
    uint64_t headerBytes = 4;
    while (std::getline(input, line))
    {
        headerBytes += line.size() + 1;
        if (headerBytes > 1024 * 1024) fail("header exceeds 1 MiB.");
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword.empty() || keyword == "comment" || keyword == "obj_info") continue;
        if (keyword == "format")
        {
            std::string format, version;
            if (hasFormat || !(tokens >> format >> version) || version != "1.0") fail("invalid PLY format declaration.");
            if (format != "ascii" && format != "binary_little_endian")
                fail("unsupported format '" + format + "'; expected ascii or binary_little_endian.");
            header.binary = format == "binary_little_endian";
            hasFormat = true;
            requireEnd(tokens);
        }
        else if (keyword == "element")
        {
            std::string name, count;
            if (!(tokens >> name >> count)) fail("invalid element declaration.");
            requireEnd(tokens);
            for (const auto& element : header.elements)
                if (element.name == name) fail("duplicate element '" + name + "'.");
            const uint64_t parsedCount = parseCount(count);
            // Every supported nonempty element consumes at least one byte per item.
            if (parsedCount > fileBytes) fail("element count exceeds the file size (truncated or invalid file).");
            header.elements.push_back({name, parsedCount, {}});
        }
        else if (keyword == "property")
        {
            if (header.elements.empty()) fail("property appears before its element.");
            std::string type, name;
            if (!(tokens >> type)) fail("invalid property declaration.");
            Property property;
            if (type == "list")
            {
                std::string countType, valueType;
                if (!(tokens >> countType >> valueType >> name)) fail("invalid list property declaration.");
                property.isList = true;
                property.countType = scalarType(countType);
                property.valueType = scalarType(valueType);
                if (!isInteger(property.countType)) fail("list count type must be an integer.");
            }
            else
            {
                if (!(tokens >> name)) fail("missing property name.");
                property.valueType = scalarType(type);
            }
            requireEnd(tokens);
            property.name = name;
            auto& properties = header.elements.back().properties;
            for (const auto& previous : properties)
                if (previous.name == name) fail("duplicate property '" + name + "'.");
            properties.push_back(std::move(property));
        }
        else if (keyword == "end_header")
        {
            requireEnd(tokens);
            hasEnd = true;
            break;
        }
        else fail("unsupported header declaration '" + keyword + "'.");
    }
    if (!hasFormat || !hasEnd) fail("missing format or end_header.");

    header.positions.fill(kMissingProperty);
    bool hasVertex = false;
    const std::array<std::string, 3> positionNames = {"x", "y", "z"};
    const std::array<std::string, 3> colorNames = {"red", "green", "blue"};
    const std::array<std::string, 3> scaleNames = {"scale_0", "scale_1", "scale_2"};
    const std::array<std::string, 4> rotationNames = {"rot_0", "rot_1", "rot_2", "rot_3"};
    for (size_t elementIndex = 0; elementIndex < header.elements.size(); ++elementIndex)
    {
        const auto& element = header.elements[elementIndex];
        if (element.count != 0 && element.properties.empty()) fail("nonempty element has no properties.");
        if (element.name != "vertex") continue;
        hasVertex = true;
        header.vertexElement = elementIndex;
        for (size_t propertyIndex = 0; propertyIndex < element.properties.size(); ++propertyIndex)
        {
            const auto& property = element.properties[propertyIndex];
            for (size_t axis = 0; axis < 3; ++axis)
            {
                if (property.name == positionNames[axis])
                {
                    if (property.isList) fail("vertex positions must be scalar properties.");
                    header.positions[axis] = propertyIndex;
                }
            }
            if (property.isList) continue;
            for (size_t axis = 0; axis < 3; ++axis)
                if (property.name == colorNames[axis] || property.name == "diffuse_" + colorNames[axis])
                    header.colors[axis] = propertyIndex;
            if (property.name == "opacity") header.opacity = propertyIndex;
            for (size_t axis = 0; axis < 3; ++axis)
                if (property.name == scaleNames[axis]) header.scales[axis] = propertyIndex;
            for (size_t i = 0; i < 4; ++i)
                if (property.name == rotationNames[i]) header.rotations[i] = propertyIndex;
        }
    }
    if (!hasVertex) fail("no vertex element.");
    for (const auto index : header.positions)
        if (index == kMissingProperty) fail("vertex element must contain x, y and z.");
    // Per-Gaussian properties are optional: a plain point cloud only marks its containing voxel.
    header.hasGaussianProperties = header.opacity != kMissingProperty &&
        !anyMissing(header.scales.data(), header.scales.size()) &&
        !anyMissing(header.rotations.data(), header.rotations.size());
    return header;
}

class PayloadReader
{
public:
    PayloadReader(std::ifstream& input, bool binary, uint64_t fileBytes) : mInput(input), mBinary(binary), mFileBytes(fileBytes) {}

    double scalar(ScalarType type)
    {
        if (!mBinary) return asciiScalar(type);
        unsigned char bytes[8] = {};
        const size_t size = scalarBytes(type);
        if (!mInput.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(size))) fail("truncated binary payload.");
        uint64_t bits = 0;
        for (size_t i = 0; i < size; ++i) bits |= uint64_t(bytes[i]) << (8 * i);
        switch (type)
        {
        case ScalarType::UInt8: case ScalarType::UInt16: case ScalarType::UInt32: return static_cast<double>(bits);
        case ScalarType::Int8: return bits < 128 ? double(bits) : double(int64_t(bits) - 256);
        case ScalarType::Int16: return bits < 32768 ? double(bits) : double(int64_t(bits) - 65536);
        case ScalarType::Int32: return bits < 2147483648ull ? double(bits) : double(int64_t(bits) - 4294967296ll);
        case ScalarType::Float32:
        {
            const uint32_t word = static_cast<uint32_t>(bits);
            float value;
            static_assert(sizeof(value) == sizeof(word), "PLY reader requires 32-bit float.");
            std::memcpy(&value, &word, sizeof(value));
            return value;
        }
        case ScalarType::Float64:
        {
            double value;
            static_assert(sizeof(value) == sizeof(bits), "PLY reader requires 64-bit double.");
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        }
        fail("invalid scalar type.");
    }

    void skip(const Property& property)
    {
        if (!property.isList)
        {
            scalar(property.valueType);
            return;
        }
        const double countValue = scalar(property.countType);
        if (countValue < 0 || countValue > double(mFileBytes)) fail("invalid list length.");
        const uint64_t count = static_cast<uint64_t>(countValue);
        if (mBinary)
        {
            const uint64_t bytes = count * scalarBytes(property.valueType);
            if (bytes > mFileBytes || bytes > uint64_t(std::numeric_limits<std::streamsize>::max())) fail("list exceeds file size.");
            mInput.ignore(static_cast<std::streamsize>(bytes));
            if (uint64_t(mInput.gcount()) != bytes) fail("truncated binary list.");
        }
        else
            for (uint64_t i = 0; i < count; ++i) scalar(property.valueType);
    }

private:
    double asciiScalar(ScalarType type)
    {
        std::string token;
        if (!(mInput >> token)) fail("truncated ASCII payload.");
        if (isInteger(type))
        {
            int64_t value = 0;
            const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
            if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size()) fail("invalid ASCII integer '" + token + "'.");
            int64_t lo = 0, hi = 0;
            switch (type)
            {
            case ScalarType::Int8: lo = -128; hi = 127; break;
            case ScalarType::UInt8: hi = 255; break;
            case ScalarType::Int16: lo = -32768; hi = 32767; break;
            case ScalarType::UInt16: hi = 65535; break;
            case ScalarType::Int32: lo = -2147483647ll - 1; hi = 2147483647ll; break;
            case ScalarType::UInt32: hi = 4294967295ll; break;
            default: break;
            }
            if (value < lo || value > hi) fail("ASCII integer is outside its declared type range.");
            return static_cast<double>(value);
        }
        double value = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value, std::chars_format::general);
        if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size()) fail("invalid ASCII number '" + token + "'.");
        return value;
    }

    std::ifstream& mInput;
    bool mBinary;
    uint64_t mFileBytes;
};

struct CloudPoint
{
    uint32_t cell;
    std::array<float, 3> local;
    std::array<float, 3> color;
    bool hasColor;
};

double normalizedColor(double value, ScalarType type)
{
    switch (type)
    {
    case ScalarType::UInt8: return value / 255.0;
    case ScalarType::Int8: return value / 127.0;
    case ScalarType::UInt16: return value / 65535.0;
    case ScalarType::Int16: return value / 32767.0;
    case ScalarType::UInt32: return value / 4294967295.0;
    case ScalarType::Int32: return value / 2147483647.0;
    default: return value; // Floating-point RGB must already be in [0,1].
    }
}

Seed fitPoints(const std::vector<CloudPoint>& points, size_t begin, size_t end, const std::array<float, 3>& voxelSize)
{
    Seed seed = {};
    seed.voxelIndex = points[begin].cell;
    const double count = double(end - begin);
    double mean[3] = {}, color[3] = {};
    size_t colorCount = 0;
    for (size_t i = begin; i < end; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            mean[j] += points[i].local[j];
            if (points[i].hasColor) color[j] += points[i].color[j];
        }
        colorCount += points[i].hasColor ? 1u : 0u;
    }
    for (size_t j = 0; j < 3; ++j)
    {
        mean[j] /= count;
        // Match optimizer center bounds before computing the enclosing scale.
        seed.center[j] = float(std::clamp(mean[j], 0.01, 0.99));
        seed.color[j] = colorCount ? float(color[j] / double(colorCount)) : 0.f;
        seed.scale[j] = 0.6f * voxelSize[j];
    }
    if (end - begin == 1) return seed; // No local orientation/extent evidence.

    // World-space covariance about the mean; voxel-local positions avoid large-world cancellation.
    double covariance[3][3] = {};
    double basis[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (size_t i = begin; i < end; ++i)
        for (size_t row = 0; row < 3; ++row)
            for (size_t col = 0; col < 3; ++col)
                covariance[row][col] += (points[i].local[row] - mean[row]) * voxelSize[row] *
                    (points[i].local[col] - mean[col]) * voxelSize[col] / count;
    // Jacobi rotations preserve an orthonormal, right-handed principal-axis frame.
    for (uint32_t iteration = 0; iteration < 32; ++iteration)
    {
        size_t p = 0, q = 1;
        for (size_t row = 0; row < 3; ++row)
            for (size_t col = row + 1; col < 3; ++col)
                if (std::abs(covariance[row][col]) > std::abs(covariance[p][q])) { p = row; q = col; }
        const double diagonal = std::abs(covariance[0][0]) + std::abs(covariance[1][1]) + std::abs(covariance[2][2]);
        if (std::abs(covariance[p][q]) <= 1e-12 * std::max(diagonal, 1e-30)) break;
        const double angle = 0.5 * std::atan2(2.0 * covariance[p][q], covariance[q][q] - covariance[p][p]);
        const double c = std::cos(angle), s = std::sin(angle);
        const double pp = covariance[p][p], qq = covariance[q][q], pq = covariance[p][q];
        for (size_t j = 0; j < 3; ++j)
        {
            if (j != p && j != q)
            {
                const double jp = covariance[j][p], jq = covariance[j][q];
                covariance[j][p] = covariance[p][j] = c * jp - s * jq;
                covariance[j][q] = covariance[q][j] = s * jp + c * jq;
            }
            const double bp = basis[j][p], bq = basis[j][q];
            basis[j][p] = c * bp - s * bq;
            basis[j][q] = s * bp + c * bq;
        }
        covariance[p][p] = c*c*pp - 2*c*s*pq + s*s*qq;
        covariance[q][q] = s*s*pp + 2*c*s*pq + c*c*qq;
        covariance[p][q] = covariance[q][p] = 0;
    }
    double axes[3] = {};
    for (size_t j = 0; j < 3; ++j)
    {
        double voxelWidthSquared = 0;
        for (size_t k = 0; k < 3; ++k)
            voxelWidthSquared += basis[k][j] * basis[k][j] * double(voxelSize[k]) * voxelSize[k];
        // Keep a quarter-voxel semi-axis floor, including planar/collinear clouds.
        axes[j] = std::max(std::sqrt(std::max(covariance[j][j], 0.0)), 0.25 * std::sqrt(voxelWidthSquared));
    }
    double maxQ = 1;
    for (size_t i = begin; i < end; ++i)
    {
        double q = 0;
        for (size_t j = 0; j < 3; ++j)
        {
            double projected = 0;
            for (size_t k = 0; k < 3; ++k)
                projected += basis[k][j] * (double(points[i].local[k]) - seed.center[k]) * voxelSize[k];
            q += projected * projected / (axes[j] * axes[j]);
        }
        maxQ = std::max(maxQ, q);
    }
    for (size_t j = 0; j < 3; ++j) seed.scale[j] = float(axes[j] * std::sqrt(maxQ) * 1.05);

    // Rotation matrix (principal axes in columns) -> quaternion, w first.
    double quat[4] = {};
    const double trace = basis[0][0] + basis[1][1] + basis[2][2];
    if (trace > 0)
    {
        const double s = 2 * std::sqrt(trace + 1);
        quat[0] = s * 0.25;
        quat[1] = (basis[2][1] - basis[1][2]) / s;
        quat[2] = (basis[0][2] - basis[2][0]) / s;
        quat[3] = (basis[1][0] - basis[0][1]) / s;
    }
    else
    {
        size_t i = 0;
        if (basis[1][1] > basis[i][i]) i = 1;
        if (basis[2][2] > basis[i][i]) i = 2;
        const size_t j = (i + 1) % 3, k = (i + 2) % 3;
        const double s = 2 * std::sqrt(std::max(0.0, 1 + basis[i][i] - basis[j][j] - basis[k][k]));
        quat[0] = (basis[k][j] - basis[j][k]) / s;
        quat[i + 1] = s * 0.25;
        quat[j + 1] = (basis[j][i] + basis[i][j]) / s;
        quat[k + 1] = (basis[k][i] + basis[i][k]) / s;
    }
    double norm = 0;
    for (double v : quat) norm += v*v;
    for (size_t j = 0; j < 4; ++j) seed.rotation[j] = float(quat[j] / std::sqrt(norm));
    return seed;
}

} // namespace

Result load_points(
    const float* positions, const float* colors, size_t pointCount,
    const std::array<uint32_t, 3>& voxelCount,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& voxelSize
)
{
    if (!pointCount || !positions || !colors)
        fail("in-memory point cloud must contain position and color triples.");
    uint64_t totalVoxelCount = 1;
    for (size_t axis = 0; axis < 3; ++axis)
    {
        if (!voxelCount[axis] || !std::isfinite(gridMin[axis]) ||
            !std::isfinite(voxelSize[axis]) || voxelSize[axis] <= 0)
            fail("invalid voxel grid dimensions or bounds.");
        totalVoxelCount *= voxelCount[axis];
        if (totalVoxelCount > std::numeric_limits<uint32_t>::max())
            fail("voxel grid exceeds the 32-bit index range.");
    }
    Result result;
    result.statistics.inputPoints = pointCount;
    std::vector<CloudPoint> cloudPoints;
    cloudPoints.reserve(pointCount);
    for (size_t i = 0; i < pointCount; ++i)
    {
        bool valid = true;
        for (size_t axis = 0; axis < 3; ++axis)
            valid = valid && std::isfinite(positions[i * 3 + axis]);
        if (!valid)
        {
            ++result.statistics.invalidPoints;
            continue;
        }
        std::array<uint32_t, 3> cell = {};
        std::array<double, 3> coordinate = {};
        for (size_t axis = 0; axis < 3; ++axis)
        {
            coordinate[axis] = (double(positions[i * 3 + axis]) - gridMin[axis]) / voxelSize[axis];
            if (!(coordinate[axis] >= 0 && coordinate[axis] < double(voxelCount[axis])))
            {
                valid = false;
                break;
            }
            cell[axis] = static_cast<uint32_t>(std::floor(coordinate[axis]));
        }
        if (!valid)
        {
            ++result.statistics.outsidePoints;
            continue;
        }
        CloudPoint point = {};
        point.cell = static_cast<uint32_t>(uint64_t(cell[0]) + uint64_t(cell[1]) * voxelCount[0] +
            uint64_t(cell[2]) * voxelCount[0] * voxelCount[1]);
        point.hasColor = true;
        for (size_t axis = 0; axis < 3; ++axis)
        {
            point.local[axis] = float(coordinate[axis] - cell[axis]);
            const double color = colors[i * 3 + axis];
            point.hasColor = point.hasColor && std::isfinite(color) && color >= 0 && color <= 1;
        }
        if (point.hasColor)
        {
            ++result.statistics.coloredPoints;
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const double color = colors[i * 3 + axis];
                point.color[axis] = float(color <= 0.04045 ? color / 12.92 :
                    std::pow((color + 0.055) / 1.055, 2.4));
            }
        }
        cloudPoints.push_back(point);
    }
    std::sort(cloudPoints.begin(), cloudPoints.end(), [](const CloudPoint& a, const CloudPoint& b) { return a.cell < b.cell; });
    for (size_t begin = 0; begin < cloudPoints.size();)
    {
        size_t end = begin + 1;
        while (end < cloudPoints.size() && cloudPoints[end].cell == cloudPoints[begin].cell) ++end;
        result.seeds.push_back(fitPoints(cloudPoints, begin, end, voxelSize));
        begin = end;
    }
    if (result.seeds.empty())
        fail("no occupied voxels inside the voxel grid (input=" + std::to_string(result.statistics.inputPoints) +
            ", invalid=" + std::to_string(result.statistics.invalidPoints) + ", outside=" +
            std::to_string(result.statistics.outsidePoints) + "). Check the point-cloud coordinates and grid bounds.");
    return result;
}

Result load(
    const std::filesystem::path& path,
    const std::array<uint32_t, 3>& voxelCount,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& voxelSize,
    double opacityThreshold
)
{
    // Occupancy of a 3DGS point cloud: a Gaussian contributes alpha * exp(-r^2 / 2) at a point,
    // with r^2 = q^T Sigma^-1 q. Occupy every cell whose AABB intersects the region
    // r^2 <= 2 ln(alpha / opacityThreshold), including thin regions between cell centers.
    if (!(opacityThreshold > 0.0 && opacityThreshold < 1.0))
        fail("opacity threshold must be in (0, 1).");

    uint64_t totalVoxelCount = 1;
    for (size_t axis = 0; axis < 3; ++axis)
    {
        if (voxelCount[axis] == 0 || !std::isfinite(gridMin[axis]) || !std::isfinite(voxelSize[axis]) || voxelSize[axis] <= 0)
            fail("invalid voxel grid dimensions or bounds.");
        totalVoxelCount *= voxelCount[axis];
        if (totalVoxelCount > std::numeric_limits<uint32_t>::max()) fail("voxel grid exceeds the 32-bit index range.");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open '" + path.string() + "'.");
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) fail("cannot determine file size.");
    const uint64_t fileBytes = static_cast<uint64_t>(end);
    input.seekg(0);
    const Header header = readHeader(input, fileBytes);
    PayloadReader reader(input, header.binary, fileBytes);

    Result result;
    std::vector<CloudPoint> cloudPoints;
    result.statistics.inputPoints = header.elements[header.vertexElement].count;

    // Occupancy bitmask: Gaussian coverage overlaps heavily, and 128^3 cells is only 256 KiB.
    std::vector<uint64_t> occupancy(static_cast<size_t>((totalVoxelCount + 63) / 64), 0);
    const auto cellOfWorld = [&](const std::array<double, 3>& world, std::array<uint32_t, 3>& cell)
    {
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const double coordinate = (world[axis] - double(gridMin[axis])) / double(voxelSize[axis]);
            if (!(coordinate >= 0 && coordinate < double(voxelCount[axis]))) return false;
            cell[axis] = static_cast<uint32_t>(std::floor(coordinate));
        }
        return true;
    };
    const auto cellIndexOf = [&](const std::array<uint32_t, 3>& cell)
    {
        return static_cast<uint32_t>(uint64_t(cell[0]) + uint64_t(cell[1]) * voxelCount[0] +
            uint64_t(cell[2]) * voxelCount[0] * voxelCount[1]);
    };
    const std::array<uint32_t, 3> blockCount = {
        (voxelCount[0] - 1) / kCoverageBlockSize + 1,
        (voxelCount[1] - 1) / kCoverageBlockSize + 1,
        (voxelCount[2] - 1) / kCoverageBlockSize + 1};
    const auto blockIndexOf = [&](uint32_t x, uint32_t y, uint32_t z)
    {
        return size_t(x) + size_t(y) * blockCount[0] + size_t(z) * blockCount[0] * blockCount[1];
    };
    std::vector<uint16_t> occupiedPerBlock(size_t(blockCount[0]) * blockCount[1] * blockCount[2], 0);
    const auto isOccupied = [&](const std::array<uint32_t, 3>& cell)
    {
        const uint32_t index = cellIndexOf(cell);
        return (occupancy[index >> 6] & (uint64_t(1) << (index & 63))) != 0;
    };
    const auto mark = [&](const std::array<uint32_t, 3>& cell)
    {
        const uint32_t index = cellIndexOf(cell);
        const uint64_t mask = uint64_t(1) << (index & 63);
        auto& word = occupancy[index >> 6];
        if ((word & mask) != 0) return false;
        word |= mask;
        ++occupiedPerBlock[blockIndexOf(cell[0] / kCoverageBlockSize,
            cell[1] / kCoverageBlockSize, cell[2] / kCoverageBlockSize)];
        return true;
    };

    for (size_t elementIndex = 0; elementIndex < header.elements.size(); ++elementIndex)
    {
        const auto& element = header.elements[elementIndex];
        const bool vertex = elementIndex == header.vertexElement;
        for (uint64_t item = 0; item < element.count; ++item)
        {
            std::array<double, 3> position = {};
            std::array<double, 3> color = {};
            double opacity = 0.0;
            std::array<double, 3> logScale = {};
            std::array<double, 4> rotation = {};
            for (size_t propertyIndex = 0; propertyIndex < element.properties.size(); ++propertyIndex)
            {
                const auto& property = element.properties[propertyIndex];
                bool consumed = false;
                if (vertex)
                {
                    for (size_t axis = 0; axis < 3; ++axis)
                    {
                        if (header.positions[axis] == propertyIndex)
                        {
                            position[axis] = reader.scalar(property.valueType);
                            consumed = true;
                            break;
                        }
                    }
                    if (!consumed && header.opacity == propertyIndex)
                    {
                        opacity = reader.scalar(property.valueType);
                        consumed = true;
                    }
                    for (size_t axis = 0; axis < 3 && !consumed; ++axis)
                        if (header.colors[axis] == propertyIndex)
                        {
                            color[axis] = normalizedColor(reader.scalar(property.valueType), property.valueType);
                            consumed = true;
                        }
                    if (!consumed)
                        for (size_t axis = 0; axis < 3 && !consumed; ++axis)
                        {
                            if (header.scales[axis] == propertyIndex)
                            {
                                logScale[axis] = reader.scalar(property.valueType);
                                consumed = true;
                            }
                        }
                    if (!consumed)
                        for (size_t i = 0; i < 4 && !consumed; ++i)
                        {
                            if (header.rotations[i] == propertyIndex)
                            {
                                rotation[i] = reader.scalar(property.valueType);
                                consumed = true;
                            }
                        }
                }
                if (!consumed) reader.skip(property);
            }
            if (!vertex) continue;

            bool valid = true;
            for (size_t axis = 0; axis < 3; ++axis)
                valid = valid && std::isfinite(position[axis]);
            if (!valid)
            {
                ++result.statistics.invalidPoints;
                continue;
            }
            const std::array<double, 3> world = {position[0], position[2], -position[1]};

            if (!header.hasGaussianProperties)
            {
                // Preserve each point for per-cell position, covariance, containment and color fitting.
                std::array<uint32_t, 3> cell = {};
                if (!cellOfWorld(world, cell))
                {
                    ++result.statistics.outsidePoints;
                    continue;
                }
                mark(cell);
                CloudPoint point = {};
                point.cell = cellIndexOf(cell);
                point.hasColor = !anyMissing(header.colors.data(), header.colors.size());
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    point.local[axis] = float((world[axis] - gridMin[axis]) / voxelSize[axis] - cell[axis]);
                    point.hasColor = point.hasColor && std::isfinite(color[axis]) && color[axis] >= 0 && color[axis] <= 1;
                }
                if (point.hasColor)
                {
                    ++result.statistics.coloredPoints;
                    for (size_t axis = 0; axis < 3; ++axis)
                        point.color[axis] = float(color[axis] <= 0.04045 ? color[axis] / 12.92 :
                            std::pow((color[axis] + 0.055) / 1.055, 2.4));
                }
                cloudPoints.push_back(point);
                continue;
            }

            // 3DGS Gaussian: opacity is a logit, scales are log of the world-unit sigma, the
            // quaternion is (w, x, y, z).
            for (size_t axis = 0; axis < 3; ++axis)
                valid = valid && std::isfinite(logScale[axis]);
            for (size_t i = 0; i < 4; ++i)
                valid = valid && std::isfinite(rotation[i]);
            // Drop NaN rows and the very transparent tail: an invisible Gaussian claims no voxels.
            const double alpha = 1.0 / (1.0 + std::exp(-opacity));
            if (!valid || !std::isfinite(alpha))
            {
                ++result.statistics.invalidPoints;
                continue;
            }
            if (alpha < opacityThreshold)
            {
                ++result.statistics.droppedByOpacity;
                continue;
            }

            std::array<double, 3> sigma = {};
            for (size_t axis = 0; axis < 3; ++axis) sigma[axis] = std::exp(logScale[axis]);
            if (!(std::isfinite(sigma[0]) && sigma[0] > 0 && std::isfinite(sigma[1]) && sigma[1] > 0 &&
                    std::isfinite(sigma[2]) && sigma[2] > 0))
            {
                ++result.statistics.invalidPoints;
                continue;
            }

            double rotationMatrix[3][3];
            makeRotation(rotation, rotationMatrix);

            // Sigma = R diag(sigma^2) R^T, so its diagonal gives the squared extent along each NeRF
            // axis, and the covered region is the ellipsoid r <= radius with r^2 <= threshold.
            const double radius = std::sqrt(2.0 * std::log(alpha / opacityThreshold));
            std::array<double, 3> half = {};
            for (size_t axis = 0; axis < 3; ++axis)
            {
                double variance = 0;
                for (size_t i = 0; i < 3; ++i)
                {
                    const double component = rotationMatrix[axis][i] * sigma[i];
                    variance += component * component;
                }
                half[axis] = radius * std::sqrt(variance);
            }
            // (x, y, z) -> (x, z, -y) is a 90 degree rotation, so the extents are simply permuted.
            half = {half[0], half[2], half[1]};

            GaussianCoverage coverage;
            coverage.threshold = radius * radius;
            // Transform the orientation together with the center into the renderer frame.
            double worldRotation[3][3];
            for (size_t i = 0; i < 3; ++i)
            {
                worldRotation[0][i] = rotationMatrix[0][i];
                worldRotation[1][i] = rotationMatrix[2][i];
                worldRotation[2][i] = -rotationMatrix[1][i];
            }
            for (size_t row = 0; row < 3; ++row)
                for (size_t column = 0; column < 3; ++column)
                {
                    coverage.worldToUnit[row][column] = worldRotation[column][row] / sigma[row];
                    for (size_t axis = 0; axis < 3; ++axis)
                        coverage.covariance[row][column] += worldRotation[row][axis] * sigma[axis] *
                            worldRotation[column][axis] * sigma[axis];
                }

            // Candidate cells: the axis-aligned box of the ellipsoid, clipped to the grid.
            std::array<uint32_t, 3> lo = {}, hi = {};
            bool overlaps = true;
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const double scale = double(voxelSize[axis]);
                const double low = (world[axis] - half[axis] - double(gridMin[axis])) / scale;
                const double high = (world[axis] + half[axis] - double(gridMin[axis])) / scale;
                if (high < 0 || low > double(voxelCount[axis]))
                {
                    overlaps = false;
                    break;
                }
                // Closed-box intersection also includes the cell below an exact grid boundary.
                lo[axis] = static_cast<uint32_t>(std::clamp(
                    std::floor(std::nextafter(low, -std::numeric_limits<double>::infinity())),
                    0.0, double(voxelCount[axis]) - 1.0));
                hi[axis] = static_cast<uint32_t>(std::min(double(voxelCount[axis]) - 1.0, std::floor(high)));
            }
            if (!overlaps)
            {
                ++result.statistics.outsidePoints;
                continue;
            }

            uint64_t covered = 0;
            const auto visit = [&](auto&& self, const std::array<uint32_t, 3>& first,
                const std::array<uint32_t, 3>& last) -> void
            {
                if (first == last && isOccupied(first)) return;
                std::array<double, 3> lower = {}, upper = {};
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    lower[axis] = double(gridMin[axis]) + double(first[axis]) * double(voxelSize[axis]) - world[axis];
                    upper[axis] = double(gridMin[axis]) + (double(last[axis]) + 1) * double(voxelSize[axis]) - world[axis];
                }
                const int classification = coverage.classifyBox(lower, upper);
                if (classification < 0) return;
                if (classification > 0)
                {
                    for (uint32_t z = first[2]; z <= last[2]; ++z)
                        for (uint32_t y = first[1]; y <= last[1]; ++y)
                            for (uint32_t x = first[0]; x <= last[0]; ++x)
                                covered += mark({x, y, z}) ? 1 : 0;
                    return;
                }
                if (first == last)
                {
                    ++result.statistics.testedCells;
                    if (coverage.intersectsBox(lower, upper)) covered += mark(first) ? 1 : 0;
                    return;
                }

                const std::array<uint32_t, 3> middle = {
                    first[0] + (last[0] - first[0]) / 2,
                    first[1] + (last[1] - first[1]) / 2,
                    first[2] + (last[2] - first[2]) / 2};
                for (uint32_t z = 0; z < (first[2] == last[2] ? 1u : 2u); ++z)
                    for (uint32_t y = 0; y < (first[1] == last[1] ? 1u : 2u); ++y)
                        for (uint32_t x = 0; x < (first[0] == last[0] ? 1u : 2u); ++x)
                            self(self,
                                {x == 0 ? first[0] : middle[0] + 1,
                                 y == 0 ? first[1] : middle[1] + 1,
                                 z == 0 ? first[2] : middle[2] + 1},
                                {x == 0 ? middle[0] : last[0],
                                 y == 0 ? middle[1] : last[1],
                                 z == 0 ? middle[2] : last[2]});
            };
            for (uint32_t z = lo[2] / kCoverageBlockSize; z <= hi[2] / kCoverageBlockSize; ++z)
                for (uint32_t y = lo[1] / kCoverageBlockSize; y <= hi[1] / kCoverageBlockSize; ++y)
                    for (uint32_t x = lo[0] / kCoverageBlockSize; x <= hi[0] / kCoverageBlockSize; ++x)
                    {
                        const std::array<uint32_t, 3> first = {
                            x * kCoverageBlockSize, y * kCoverageBlockSize, z * kCoverageBlockSize};
                        std::array<uint32_t, 3> last = {};
                        uint32_t capacity = 1;
                        for (size_t axis = 0; axis < 3; ++axis)
                        {
                            const uint32_t size = std::min(kCoverageBlockSize, voxelCount[axis] - first[axis]);
                            last[axis] = first[axis] + size - 1;
                            capacity *= size;
                        }
                        if (occupiedPerBlock[blockIndexOf(x, y, z)] == capacity)
                        {
                            ++result.statistics.skippedFullBlocks;
                            continue;
                        }
                        ++result.statistics.testedBlocks;
                        visit(visit, first, last);
                    }
            result.statistics.coveredCells += covered;
            result.statistics.maxCoveredCells = std::max(result.statistics.maxCoveredCells, covered);
        }
    }

    if (!header.hasGaussianProperties)
    {
        std::sort(cloudPoints.begin(), cloudPoints.end(), [](const CloudPoint& a, const CloudPoint& b) { return a.cell < b.cell; });
        for (size_t begin = 0; begin < cloudPoints.size();)
        {
            size_t groupEnd = begin + 1;
            while (groupEnd < cloudPoints.size() && cloudPoints[groupEnd].cell == cloudPoints[begin].cell) ++groupEnd;
            result.seeds.push_back(fitPoints(cloudPoints, begin, groupEnd, voxelSize));
            begin = groupEnd;
        }
    }
    // Gaussian coverage keeps the existing per-cell fallback, including cells with no input center.
    // Scan order is ascending, so the seeds come out sorted and each occupied index appears once.
    else
    for (size_t word = 0; word < occupancy.size(); ++word)
        for (uint32_t bit = 0; bit < 64; ++bit)
            if ((occupancy[word] >> bit) & 1ull)
            {
                const uint64_t index = uint64_t(word) * 64 + bit;
                if (index < totalVoxelCount)
                {
                    Seed seed = {};
                    seed.voxelIndex = static_cast<uint32_t>(index);
                    for (size_t axis = 0; axis < 3; ++axis) seed.scale[axis] = 0.6f * voxelSize[axis];
                    result.seeds.push_back(seed);
                }
            }
    if (result.seeds.empty())
        fail("no occupied voxels inside the voxel grid (input=" + std::to_string(result.statistics.inputPoints) +
            ", invalid=" + std::to_string(result.statistics.invalidPoints) + ", outside=" +
            std::to_string(result.statistics.outsidePoints) + "). Check the point-cloud coordinates and grid bounds.");
    return result;
}
} // namespace mitsuba::reconstruction
