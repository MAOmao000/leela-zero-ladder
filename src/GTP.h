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

#ifndef GTP_H_INCLUDED
#define GTP_H_INCLUDED

#include "config.h"

#include <cstdio>
#include <string>
#include <vector>

#include "GameState.h"
#include "Network.h"
#include "UCTSearch.h"

#ifdef USE_TENSOR_RT
#include "NvInfer.h"

namespace trtLog {
// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    Logger(Severity severity = Severity::kERROR)
        : mReportableSeverity(severity) {}
    void log(ILogger::Severity severity, const char* msg) noexcept override {
        // suppress information level log
        if (severity <= mReportableSeverity) {
            switch (severity) {
                case Severity::kINTERNAL_ERROR:
                    std::cerr << "[F] " << msg << std::endl;
                    break;
                case Severity::kERROR:
                    std::cerr << "[E] " << msg << std::endl;
#ifndef NDEBUG
                    std::cout << boost::stacktrace::stacktrace();
                    exit(0);
#endif
                    break;
                case Severity::kWARNING:
                    std::cerr << "[W] " << msg << std::endl;
                    break;
                case Severity::kINFO:
                    std::cerr << "[I] " << msg << std::endl;
                    break;
                case Severity::kVERBOSE:
                    std::cerr << "[V] " << msg << std::endl;
                    break;
                default:
                    std::cerr << "[?] " << msg << std::endl;
            }
        }
    }
    nvinfer1::ILogger& getTRTLogger() noexcept {
        return *this;
    }
    void setReportableSeverity(Severity severity) noexcept {
        mReportableSeverity = severity;
    }
private:
    Severity mReportableSeverity;
};
}
#endif

struct MoveToAvoid {
    int color;
    size_t until_move;
    int vertex;

    MoveToAvoid(const int color, const size_t until_move, const int vertex)
        : color(color), until_move(until_move), vertex(vertex) {}

    bool operator==(const MoveToAvoid& other) const {
        return color == other.color && until_move == other.until_move
               && vertex == other.vertex;
    }
};

class AnalyzeTags {
    friend class LeelaTest;

public:
    AnalyzeTags() = default;
    AnalyzeTags(std::istringstream& cmdstream, const GameState& game);

    void add_move_to_avoid(int color, int vertex, size_t until_move);
    void add_move_to_allow(int color, int vertex, size_t until_move);
    int interval_centis() const;
    int invalid() const;
    int who() const;
    size_t post_move_count() const;
    bool is_to_avoid(int color, int vertex, size_t movenum) const;
    bool has_move_restrictions() const;

private:
    bool m_invalid{true};
    std::vector<MoveToAvoid> m_moves_to_avoid, m_moves_to_allow;
    int m_interval_centis{0};
    int m_who{FastBoard::INVAL};
    size_t m_min_moves{0};
};

extern bool cfg_gtp_mode;
extern bool cfg_allow_pondering;
extern unsigned int cfg_num_threads;
extern unsigned int cfg_batch_size;
extern int cfg_max_playouts;
extern int cfg_max_visits;
extern int cfg_min_visits;
extern size_t cfg_max_memory;
extern size_t cfg_max_tree_size;
extern int cfg_max_cache_ratio_percent;
extern int cfg_z_entries;
extern TimeManagement::enabled_t cfg_timemanage;
extern int cfg_lagbuffer_cs;
extern int cfg_resignpct;
extern int cfg_noise;
extern int cfg_random_cnt;
extern int cfg_random_min_visits;
extern float cfg_random_temp;
extern std::uint64_t cfg_rng_seed;
extern bool cfg_dumbpass;
enum class backend_t {
    NONE, OPENCL, CUDNN, CUDNNGRAPH, TENSORRT
};
extern backend_t cfg_backend;
extern bool cfg_NCHW;
#ifdef USE_OPENCL
extern std::vector<int> cfg_gpus;
extern bool cfg_sgemm_exhaustive;
extern bool cfg_tune_only;
enum class execute_t {
    NONE, SINGLE, DOUBLE
};
extern execute_t cfg_execute_context;
#ifdef USE_TENSOR_RT
extern trtLog::Logger cfg_logger;
#endif
extern bool cfg_cache_plan;
#ifdef USE_HALF
enum class precision_t {
    AUTO, SINGLE, HALF
};
extern precision_t cfg_precision;
#endif
#endif
extern float cfg_puct;
extern float cfg_logpuct;
extern float cfg_logconst;
extern float cfg_puct_init;
extern float cfg_puct_base;
extern float cfg_puct_log;
extern float cfg_dynamic_k_factor;
extern float cfg_dynamic_k_base;
extern float cfg_stdev_scale;
extern float cfg_stdev_prior;
extern float cfg_softmax_temp;
extern float cfg_fpu_reduction;
extern float cfg_fpu_root_reduction;
extern float cfg_ci_alpha;
extern float cfg_lcb_min_visit_ratio;
extern float cfg_cut_policy;
extern std::string cfg_logfile;
extern std::string cfg_weightsfile;
extern FILE* cfg_logfile_handle;
extern bool cfg_quiet;
extern std::string cfg_options_str;
extern bool cfg_benchmark;
extern bool cfg_cpu_only;
extern bool cfg_alpha_zero_search;
extern bool cfg_use_stdev_uct;

extern bool cfg_use_ray_ladder;
extern bool cfg_ladder_check;
extern int cfg_ladder_defense;
extern int cfg_ladder_offense;
extern int cfg_defense_stones;
extern int cfg_offense_stones;
extern int cfg_ladder_depth;

extern AnalyzeTags cfg_analyze_tags;

static constexpr size_t MiB = 1024LL * 1024LL;

/*
    A list of all valid GTP2 commands is defined here:
    https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
    GTP is meant to be used between programs. It's not a human interface.
*/
class GTP {
public:
    static std::unique_ptr<Network> s_network;
    static void initialize(std::unique_ptr<Network>&& network);
    static void execute(GameState& game, const std::string& xinput);
    static void setup_default_parameters();

private:
    static constexpr int GTP_VERSION = 2;

    static std::string get_life_list(const GameState& game, bool live);
    static const std::string s_commands[];
    static const std::string s_options[];
    static std::pair<std::string, std::string> parse_option(
        std::istringstream& is);
    static std::pair<bool, std::string> set_max_memory(
        size_t max_memory, int cache_size_ratio_percent);
    static void execute_setoption(UCTSearch& search, int id,
                                  const std::string& command);

    // Memory estimation helpers
    static size_t get_base_memory();
    static size_t add_overhead(const size_t s) {
        return s * 11LL / 10LL;
    }
    static size_t remove_overhead(const size_t s) {
        return s * 10LL / 11LL;
    }
};

#endif
