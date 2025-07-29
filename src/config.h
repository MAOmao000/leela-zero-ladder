/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef CONFIG_H_INCLUDED
#define CONFIG_H_INCLUDED

#if defined(USE_TENSOR_RT)
#undef USE_CPU_ONLY
#if !defined(USE_CUDNN)
#define USE_CUDNN
#endif
#if !defined(USE_OPENCL)
#define USE_OPENCL
#endif

#elif defined(USE_CUDNN)
#undef USE_CPU_ONLY
#if !defined(USE_OPENCL)
#define USE_OPENCL
#endif

#elif defined(USE_OPENCL)
#undef USE_CPU_ONLY

#elif !defined(USE_CPU_ONLY)
#define USE_OPENCL

#endif

/*
 * Features
 *
 * USE_DNNL and USE_OPENBLAS: Optionally use a basic linear algebra library.
 * This is may perform faster than the included Eigen library,
 * and some BLAS libraries can target multiple CPU models.
 * Not all operations are performed on the GPU -
 * some operations won't get any speedup from it.
 * Also used for OpenCL self-checks.
 */
/*
 * We use Eigen by default, except on macOS, which has a fast BLAS
 * built-in. (Accelerate)
 */
#if !defined(__APPLE__) && !defined(__MACOSX)
#undef USE_ACCELERATE
#if defined(USE_DNNL)
#undef USE_OPENBLAS
#undef USE_EIGEN
#elif defined(USE_OPENBLAS)
#undef USE_EIGEN
#elif !defined(USE_EIGEN)
#define USE_EIGEN
#endif

#else
#if !defined(USE_ACCELERATE)
#define USE_ACCELERATE
#endif
#undef USE_DNNL
#undef USE_OPENBLAS
#undef USE_EIGEN
#endif

#ifndef NDEBUG
#include <crtdbg.h>
#endif

/*
 * We need to check for input while we are thinking.
 * That code isn't portable, so select something appropriate for the system.
 */
#ifdef _WIN32
#undef HAVE_SELECT
#define NOMINMAX
#else
#define HAVE_SELECT
#endif

/*
 * BOARD_SIZE: Define size of the board to compile Leela with, must be an odd
   number due to winograd tiles
 */
static constexpr auto BOARD_SIZE = 19;
static_assert(BOARD_SIZE % 2 == 1,
              "Code assumes odd board size, remove at your own risk!");

static constexpr auto NUM_INTERSECTIONS = BOARD_SIZE * BOARD_SIZE;
static constexpr auto POTENTIAL_MOVES = NUM_INTERSECTIONS + 1; // including pass

/*
 * KOMI: Define the default komi to use when training.
 */
static constexpr auto KOMI = 7.5f;

/*
 * NetworkType: Whether the weight of MiniGo includes se-net.
 */
enum class NetworkType {
    LEELA_ZERO, MINIGO_SE
};

static constexpr auto PROGRAM_NAME = "Leela Zero(ladder detection)";
static constexpr auto PROGRAM_VERSION_MAJOR = "3";
static constexpr auto PROGRAM_VERSION_MINOR = "0";
static constexpr auto PROGRAM_VERSION_PATCH = "4";

/*
 * OpenBLAS limitation: the default configuration on some Linuxes
 * is limited to 64 cores.
 */
#if defined(USE_OPENBLAS)
static constexpr auto MAX_CPUS = 64;
#else
static constexpr auto MAX_CPUS = 256;
#endif

#if !defined(USE_CPU_ONLY)
/*
 * USE_HALF: Include the half-precision OpenCL implementation when building.
 * The current implementation autodetects whether half-precision is better
 * or single-precision is better (half precision is chosen if it's 5% faster)
 * Half-precision OpenCL gains performance on some GPUs while losing some
 * accuracy on the calculation, but generally it is worth using half precision
 * if it is at least 5% faster.
 */
#include "half/half.hpp"
#if defined(USE_OPENCL_SELFCHECK)
// If OpenCL are fully usable, then check the OpenCL against CPU
// implementation with some probability.
static constexpr auto SELFCHECK_PROBABILITY = 2000;
#endif
#else
#undef USE_OPENCL_SELFCHECK
#endif

#if (_MSC_VER >= 1900) /* VC14+ Disable all deprecation warnings */
#pragma warning(disable : 4996)
#endif /* VC14+ */

#endif
