/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Henrik Forsten
    Copyright (C) 2025 MAOmao000

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
*/

#include "config.h"

#if defined(USE_TENSOR_RT)

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <inttypes.h>

#include "GTP.h"
#include "BackendTensorRT.h"

using namespace Utils;
using namespace nvinfer1;

class from_float {
public:
    from_float(const std::vector<float>& f) : m_f(f) {}

    operator const std::vector<float> &() {
        return m_f;
    }

    operator std::vector<half_float::half>() {
        auto ret = std::vector<half_float::half>(m_f.size());
        std::copy(cbegin(m_f), cend(m_f), begin(ret));
        return ret;
    }

private:
    const std::vector<float>& m_f;
};

template <typename net_t>
BackendTRT<net_t>::BackendTRT(
    const int gpu,
    const bool silent) {

    // Certain minor versions of TensorRT uses a global logger, which is bad.
    // Since TensorRT maintains ABI compatibility between minor versions, a dynamic library mismatch
    // does not necessarily generate a dynamic link error, therefore, an extra check is required.
    if (getInferLibVersion() / 100 != NV_TENSORRT_VERSION / 100) {
        myprintf("TensorRT backend: detected incompatible version of TensorRT library.\n");
        exit(EXIT_FAILURE);
    }

    auto best_bandwidth = 0.0;
    auto found_device = false;
    auto nDevices = 0;
    auto best_device_id = 0;
    cudaDeviceProp best_device;

    if (cudaGetDeviceCount(&nDevices) != cudaSuccess) {
        myprintf("cudaGetDeviceCount error(%d).\n", nDevices);
    }

    if (!silent) {
        myprintf("Detected %d CUDA devices.\n", nDevices);
    }

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        int clock_rate = 0;
        cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, i);
        auto bandwidth = 2.0f * clock_rate * (prop.memoryBusWidth / 8) / 1.0e6;
        if (!silent) {
            myprintf("Device Number: %d\n", i);
            myprintf("  Device name: %s\n", prop.name);
            myprintf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            myprintf("  Peak Memory Bandwidth (GB/s): %.1f\n\n", bandwidth);
        }

        bool preferred = (gpu == i);

        if (bandwidth > best_bandwidth || preferred) {
            best_bandwidth = bandwidth;
            best_device = prop;
            best_device_id = i;
            if (preferred) {
                best_bandwidth = std::numeric_limits<decltype(best_bandwidth)>::max();
            } else {
                best_bandwidth = bandwidth;
            }
            found_device = true;
        }
    }

    if (!found_device) {
        myprintf("No suitable CUDA device found.\n");
        exit(EXIT_FAILURE);
    }

    myprintf("Selected device: %s\n", best_device.name);
    myprintf("with compute capability %d.%d.\n", best_device.major, best_device.minor);

    cudaSetDevice(best_device_id);
    m_device_prop = best_device;
}

template <typename net_t>
void BackendTRT<net_t>::initialize(
    const int channels,
    const size_t batch_size,
    const NetworkType net_type,
    const size_t num_worker_threads,
    const std::string &model_hash) {

    // For compatibility with OpenCL implementation
    (void)channels;
    (void)batch_size;

    m_net_type = net_type;
    m_num_worker_threads = static_cast<int>(num_worker_threads);
    m_model_hash = model_hash;
}

template <typename net_t>
bool BackendTRT<net_t>::build(
    const int num_worker_threads,
    const int64_t batch_size) {

    // Bump this when between program versions we want to forcibly drop old timing caches and plan caches.
    std::string tune_desc = strprintf(
        R"|("salt"(%s_%s_%s)"model %s"(%s,%d,%d))|",
        PROGRAM_VERSION_MAJOR,
        PROGRAM_VERSION_MINOR,
        PROGRAM_VERSION_PATCH,
        typeid(net_t) == typeid(float) ? "single" : "half",
        "1.0",                    // model version
        Network::INPUT_CHANNELS,  // number of input channels
        batch_size
    );
    auto builder
        = TrtUniquePtr<IBuilder>(createInferBuilder(cfg_logger.getTRTLogger()));
    if (!builder) {
        std::cerr << "TensorRT backend: failed to create builder" << std::endl;
        return false;
    }
    auto config = TrtUniquePtr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "TensorRT backend: failed to create builder config" << std::endl;
        return false;
    }
    if (typeid(net_t) == typeid(half_float::half)) {
        config->setFlag(BuilderFlag::kFP16);
    }

    for (auto i = 0; i < num_worker_threads; i++) {
        auto profile = builder->createOptimizationProfile();
        if (!profile) {
            std::cerr << "TensorRT backend: failed to create optimization profile" << std::endl;
            return false;
        }
        profile->setDimensions("InputFeature", OptProfileSelector::kMIN,
            Dims4(1, m_layers[0].channels, BOARD_SIZE, BOARD_SIZE));
        profile->setDimensions("InputFeature", OptProfileSelector::kOPT,
            Dims4(batch_size, m_layers[0].channels, BOARD_SIZE, BOARD_SIZE));
        profile->setDimensions("InputFeature", OptProfileSelector::kMAX,
            Dims4(batch_size, m_layers[0].channels, BOARD_SIZE, BOARD_SIZE));
        if (m_net_type == NetworkType::MINIGO_SE) {
            profile->setDimensions("BatchSize", OptProfileSelector::kMIN,
                Dims4(1, m_layers[1].channels, 1, 1));
            profile->setDimensions("BatchSize", OptProfileSelector::kOPT,
                Dims4(batch_size, m_layers[1].channels, 1, 1));
            profile->setDimensions("BatchSize", OptProfileSelector::kMAX,
                Dims4(batch_size, m_layers[1].channels, 1, 1));
        }
        config->addOptimizationProfile(profile);
    }

    nvinfer1::NetworkDefinitionCreationFlags flags = 0U;
    auto network = TrtUniquePtr<INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network) {
        std::cerr << "TensorRT backend: failed to create network definition" << std::endl;
        return false;
    }
    std::filesystem::path path = cfg_weightsfile;
    std::string filename = path.filename().string();
    auto ext_i = filename.find_last_of(".");
    std::string weightsfile = filename.substr(0, ext_i);
    network->setName(weightsfile.c_str());
    if (!constructNetwork(network, tune_desc)) {
        std::cerr << "TensorRT backend: failed to construct network" << std::endl;
        return false;
    }
    if (m_device_prop.major >= 8) {
        // This is to avoid tactics that have shape switching overhead
        config->setTacticSources(1U << static_cast<uint32_t>(TacticSource::kJIT_CONVOLUTIONS));
        config->setBuilderOptimizationLevel(cfg_builder_opt_level);
    }
    // So that there are no concurrent kernel executions probably from other parts of code while profiling
    // See CUDA Runtime API document for more details related to NULL stream and synchronization behaviors
    config->setProfileStream(cudaStreamPerThread);
    // Typical runtime allocation is much less than the 2 GiB specified below
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 31);

    std::string plan;
    {
        static std::mutex tuneMutex;
        tuneMutex.lock();
        std::string cacheDir = Utils::leelaz_file("trtcache");
        std::filesystem::create_directory(cacheDir);
        assert(std::filesystem::exists(cacheDir));
        assert(std::filesystem::is_directory(cacheDir));

        uint8_t deviceHash[32];
        SHA2::get256(m_device_prop.name, deviceHash);

        // Truncated to 4 bytes
        char deviceIdent[4 * 2 + 1];
        for(int i = 0; i < 4; i++) {
            sprintf(deviceIdent + i * 2, "%02x", static_cast<unsigned char>(deviceHash[i]));
        }
        deviceIdent[sizeof(deviceIdent) - 1] = 0;

        std::string precision = typeid(net_t) == typeid(float) ? "single" : "half";
        std::string sep_char{std::filesystem::path::preferred_separator};

        uint8_t tuneHash[32];
        SHA2::get256(tune_desc.c_str(), tuneHash);
        // Truncated to 6 bytes
        char tuneIdent[6 * 2 + 1];
        for(int i = 0; i < 6; i++) {
            sprintf(tuneIdent + i * 2, "%02x", static_cast<unsigned char>(tuneHash[i]));
        }
        tuneIdent[sizeof(tuneIdent) - 1] = 0;

        if (cfg_cache_plan) {
            auto planCacheFile = strprintf(
                "%s%strt-%d_gpu-%s_tune-%s_net-%s_%s_%s_%s_%dx%d_batch%" PRId64 "x%d_%s",
                cacheDir.c_str(),
                sep_char.c_str(),
                getInferLibVersion(),
                deviceIdent,
                tuneIdent,
                network->getName(),
                PROGRAM_VERSION_MAJOR,
                PROGRAM_VERSION_MINOR,
                PROGRAM_VERSION_PATCH,
                BOARD_SIZE,
                BOARD_SIZE,
                batch_size,
                num_worker_threads,
                precision.c_str()
            );
            std::string paramStr = strprintf(
                "_%d_%s_%s_%s_%s_%d_%d_%" PRId64 "x%d_%s",
                getInferLibVersion(),
                deviceIdent,
                PROGRAM_VERSION_MAJOR,
                PROGRAM_VERSION_MINOR,
                PROGRAM_VERSION_PATCH,
                BOARD_SIZE,
                BOARD_SIZE,
                batch_size,
                num_worker_threads,
                precision.c_str()
            );
#ifdef NDEBUG
            lockFile(planCacheFile);
#endif
            try {
                plan = readFileBinary(planCacheFile);
            } catch (std::exception const& e) {
                (void) e;
            };
            if (plan.size() > 0) {
                if (plan.size() < 64 + paramStr.size()) {
                    std::cout << "Could not parse plan, unexpected size in " + planCacheFile << std::endl;
                    plan.clear();
                } else {
                    std::string cachedParamStr = plan.substr(plan.size() - paramStr.size());
                    std::string modelHash = plan.substr(plan.size() - 64 - paramStr.size(), 64);
                    if (modelHash != m_model_hash) {
                        std::cout << "Plan cache is corrupted or is for the wrong model in " + planCacheFile << std::endl;
                        plan.clear();
                    } else if (cachedParamStr != paramStr) {
                        std::cout << "Plan cache is corrupted or is for the wrong parameters in " + planCacheFile << std::endl;
                        plan.clear();
                    } else {
                        plan.erase(plan.size() - 64 - paramStr.size());
                    }
                }
            }
            if (plan.size() <= 0) {
                std::cout << "Creating new plan cache" << std::endl;
                auto planBuffer = std::unique_ptr<IHostMemory>(
                    builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    tuneMutex.unlock();
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
#ifdef NDEBUG
                    unlockFile();
#endif
                    return false;
                }
                plan.insert(
                    plan.end(),
                    static_cast<char*>(planBuffer->data()),
                    static_cast<char*>(planBuffer->data()) + planBuffer->size()
                );
                if (m_model_hash.size() != 64) {
                    tuneMutex.unlock();
                    std::cerr << "Unexpected model hash size" << std::endl;
#ifdef NDEBUG
                    unlockFile();
#endif
                    return false;
                }
                plan.insert(
                    plan.end(),
                    m_model_hash.begin(),
                    m_model_hash.end()
                );
                plan.insert(
                    plan.end(),
                    paramStr.begin(),
                    paramStr.end()
                );
                std::ofstream ofs;
                ofs.open(planCacheFile, std::ios_base::out | std::ios_base::binary);
                ofs.write(plan.data(), plan.size());
                ofs.close();
                std::cout << "Saved new plan cache to " + planCacheFile << std::endl;
                plan.erase(plan.size() - 64 - paramStr.size());
            } else {
                std::cout << "Using existing plan cache at " + planCacheFile << std::endl;
            }
#ifdef NDEBUG
            unlockFile();
#endif
        } else {
            auto timingCacheFile = strprintf(
                "%s%strt-%d_gpu-%s_tune-%s_%s_%s_%s_%dx%d_batch%" PRId64 "x%d_%s",
                cacheDir.c_str(),
                sep_char.c_str(),
                getInferLibVersion(),
                deviceIdent,
                tuneIdent,
                PROGRAM_VERSION_MAJOR,
                PROGRAM_VERSION_MINOR,
                PROGRAM_VERSION_PATCH,
                BOARD_SIZE,
                BOARD_SIZE,
                batch_size,
                num_worker_threads,
                precision.c_str()
            );
#ifdef NDEBUG
            lockFile(timingCacheFile);
#endif
            std::string timingCacheBlob;
            try {
                timingCacheBlob = readFileBinary(timingCacheFile);
            } catch (std::exception const& e) {
                (void) e;
            };
            if (timingCacheBlob.size() > 0)
                std::cout << "Using existing timing cache at " << timingCacheFile << std::endl;
            else
                std::cout << "Creating new timing cache" << std::endl;

            auto timingCache =
                std::unique_ptr<ITimingCache>(
                    config->createTimingCache(timingCacheBlob.data(), timingCacheBlob.size()));
            auto invalidTimingCache = !config->setTimingCache(*timingCache, false);
            if (invalidTimingCache) {
                std::cout << "Invalid timing cache, using new one instead" << std::endl;
                timingCache.reset(config->createTimingCache(nullptr, 0));
                config->setTimingCache(*timingCache, false);
            }
            std::unique_ptr<IHostMemory> planBuffer;
            if (invalidTimingCache || !timingCacheBlob.size()) {
                planBuffer.reset(builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    tuneMutex.unlock();
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
#ifdef NDEBUG
                    unlockFile();
#endif
                    return false;
                }
                auto serializedTimingCache = std::unique_ptr<IHostMemory>(
                    config->getTimingCache()->serialize());
                std::ofstream ofs;
                ofs.open(timingCacheFile, std::ios_base::out | std::ios_base::binary);
                ofs.write(static_cast<char*>(serializedTimingCache->data()), serializedTimingCache->size());
                ofs.close();
                std::cout << "Saved new timing cache to " << timingCacheFile << std::endl;
            } else {
                planBuffer.reset(builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    tuneMutex.unlock();
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
#ifdef NDEBUG
                    unlockFile();
#endif
                    return false;
                }
            }
            plan.insert(
                plan.end(),
                static_cast<char*>(planBuffer->data()),
                static_cast<char*>(planBuffer->data()) + planBuffer->size());
#ifdef NDEBUG
            unlockFile();
#endif
        }
        tuneMutex.unlock();
    }
    for (auto i = 0; i < num_worker_threads; i++) {
        std::unique_ptr<IRuntime> runtime
            = std::unique_ptr<IRuntime>(createInferRuntime(cfg_logger.getTRTLogger()));
        if (!runtime) {
            std::cerr << "createInferRuntime error: " << std::endl;
            return false;
        }
        runtime->setErrorRecorder(&trtErrorRecorder);
        std::unique_ptr<ICudaEngine> engine
            = std::unique_ptr<ICudaEngine>(
                runtime->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine) {
            std::cerr << "deserializeCudaEngine error: " << std::endl;
            return false;
        }
        std::unique_ptr<BackendTRTContext> context = std::make_unique<BackendTRTContext>();
        context->mContext.reset(engine->createExecutionContext());
        for (auto j = 0; j < engine->getNbIOTensors(); j++) {
            void* buffer = nullptr;
            auto name = engine->getIOTensorName(j);
            auto dims = engine->getTensorShape(name);
            std::string_view name_str{name};
            size_t size_byte;
            if (name_str == "BatchSize") {
                size_byte = sizeof(int32_t);
            } else {
                size_byte = sizeof(float);
            }
            size_t bytes = std::accumulate(
                dims.d + 1,
                dims.d + dims.nbDims,
                batch_size * size_byte,
                std::multiplies<size_t>());
            checkCUDA(cudaMalloc(&buffer, bytes));
            if (name_str == "BatchSize") {
                auto input_batch
                    = std::vector<int32_t>(batch_size * m_layers[1].channels, 0);
                checkCUDA(cudaMemcpy(
                    buffer,
                    (int32_t*)&input_batch[0],
                    bytes,
                    cudaMemcpyHostToDevice));
            }
            context->mBuffers.emplace(std::make_pair(name, buffer));
            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                context->mContext->setInputTensorAddress(name, buffer);
            } else {
                context->mContext->setOutputTensorAddress(name, buffer);
            }
        }
        context->m_buffers_allocated = true;
        context->mContext->setOptimizationProfileAsync(i, cudaStreamPerThread);
        mRuntime.emplace_back(std::move(runtime));
        mEngine.emplace_back(std::move(engine));
        m_context.emplace_back(std::move(context));
        trtErrorRecorder.clear();
    }
    return true;
}

template <typename net_t>
bool BackendTRT<net_t>::constructNetwork(
    TrtUniquePtr<INetworkDefinition>& network,
    std::string& tune_desc) {

    ITensor* inputFeature = nullptr;
    ITensor* outputConv = nullptr;
    ILayer* outPolicyLayer = nullptr;
    ILayer* outValueLayer = nullptr;
    ILayer* shapeLayer = nullptr;

    if (m_net_type == NetworkType::MINIGO_SE) {
        auto batchSizeTensor = initInputs(
            "BatchSize",
            network,
            m_layers[1].channels,
            1,
            1);

        // See. https://github.com/NVIDIA/TensorRT/issues/2282
        auto inShapeLayer = network->addShape(*batchSizeTensor);
        shapeLayer = network->addCast(*inShapeLayer->getOutput(0), DataType::kINT32);
    }

    for (auto iter = std::begin(m_layers);
         iter != std::end(m_layers); iter++) {

        const auto& layer = *iter;
        if (layer.is_input_convolution) {
            inputFeature = initInputs(
                "InputFeature",
                network,
                layer.channels,
                BOARD_SIZE,
                BOARD_SIZE);
            auto conv_weights = begin(layer.weights);
            auto conv_biases = begin(layer.weights) + 1;
            auto initialConvLayer = buildConvLayer(
                inputFeature,
                layer.filter_size,
                layer.weights_size[0],
                conv_weights[0],
                layer.weights_size[1],
                conv_biases[0],
                network,
                tune_desc,
                layer.name + ".conv",
                layer.outputs);
            auto outputConvLayer = buildActivationLayer(
                initialConvLayer->getOutput(0),
                network,
                tune_desc,
                layer.name + ".activation",
                ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else if (layer.is_residual_block && !layer.is_se_block) {
            if (!outputConv) {
                std::cerr << "outputConv is nullptr on residual block." << std::endl;
                return false;
            }
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases  = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases  = begin(layer.weights) + 3;
            auto firstConvLayer = buildConvLayer(
                outputConv,
                layer.filter_size,
                layer.weights_size[0],
                conv1_weights[0],
                layer.weights_size[1],
                conv1_biases[0],
                network,
                tune_desc,
                layer.name + ".conv.first",
                layer.outputs);
            auto firstActivationConvLayer = buildActivationLayer(
                firstConvLayer->getOutput(0),
                network,
                tune_desc,
                layer.name + ".activation.first",
                ActivationType::kRELU);
            auto secondConvLayer = buildConvLayer(
                firstActivationConvLayer->getOutput(0),
                layer.filter_size,
                layer.weights_size[2],
                conv2_weights[0],
                layer.weights_size[3],
                conv2_biases[0],
                network,
                tune_desc,
                layer.name + ".conv.second",
                layer.outputs);
            auto mergeLayer = network->addElementWise(
                *outputConv, *secondConvLayer->getOutput(0), ElementWiseOperation::kSUM);
            auto outputConvLayer = buildActivationLayer(
                mergeLayer->getOutput(0),
                network,
                tune_desc,
                layer.name + ".activation.final",
                ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else if (layer.is_residual_block && layer.is_se_block) {
            if (!shapeLayer || !outputConv) {
                std::cerr << "shapeLayer or outputConv is nullptr on residual se block." << std::endl;
                return false;
            }
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases  = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases  = begin(layer.weights) + 3;
            auto fc1_weights   = begin(layer.weights) + 4;
            auto fc1_biases    = begin(layer.weights) + 5;
            auto fc2_weights   = begin(layer.weights) + 6;
            auto fc2_biases    = begin(layer.weights) + 7;
            auto firstConvLayer = buildConvLayer(
                outputConv,
                layer.filter_size,
                layer.weights_size[0],
                conv1_weights[0],
                layer.weights_size[1],
                conv1_biases[0],
                network,
                tune_desc,
                layer.name + ".conv.first",
                layer.outputs);
            auto firstActivationConvLayer = buildActivationLayer(
                firstConvLayer->getOutput(0),
                network,
                tune_desc,
                layer.name + ".activation.first",
                ActivationType::kRELU);
            auto secondConvLayer = buildConvLayer(
                firstActivationConvLayer->getOutput(0),
                layer.filter_size,
                layer.weights_size[2],
                conv2_weights[0],
                layer.weights_size[3],
                conv2_biases[0],
                network,
                tune_desc,
                layer.name + ".conv.second",
                layer.outputs);
            // pool = tf.layers.average_pooling2d(residual, pool_size=go.N, strides=1, padding='valid')
            auto gpoolLayer = applyGPoolLayer(
                secondConvLayer->getOutput(0),
                network);
            // fc1 = tf.layers.dense(pool, units=channels // 2)
            auto thirdMatMulLayer = buildConvLayer(
                gpoolLayer->getOutput(0),
                1,
                layer.weights_size[4],
                fc1_weights[0],
                layer.weights_size[5],
                fc1_biases[0],
                network,
                tune_desc,
                layer.name + ".conv.third",
                layer.outputs / 2);
            // squeeze = tf.nn.relu(fc1)
            auto thirdActivationMatLayer = buildActivationLayer(
                thirdMatMulLayer->getOutput(0),
                network,
                tune_desc,
                layer.name + ".activation.third",
                ActivationType::kRELU);
            // fc2 = tf.layers.dense(squeeze, units=2*channels)
            auto fourthMatMulLayer = buildConvLayer(
                thirdActivationMatLayer->getOutput(0),
                1,
                layer.weights_size[6],
                fc2_weights[0],
                layer.weights_size[7],
                fc2_biases[0],
                network,
                tune_desc,
                layer.name + ".conv.fourth",
                layer.outputs * 2);
            // gamma = tf.split(fc2, 2, axis=3)
            auto gammaLayer = network->addSlice(
                *fourthMatMulLayer->getOutput(0),
                {4 ,{0, 0, 0, 0}},
                {4 ,{0, layer.channels, 1, 1}},
                {4 ,{1, 1, 1, 1}}
            );
            gammaLayer->setInput(2, *shapeLayer->getOutput(0));
            // bias = tf.split(fc2, 2, axis=3)
            auto biasLayer = network->addSlice(
                *fourthMatMulLayer->getOutput(0),
                {4 ,{0, layer.channels, 0, 0}},
                {4 ,{0, layer.channels, 1, 1}},
                {4 ,{1, 1, 1, 1}}
            );
            biasLayer->setInput(2, *shapeLayer->getOutput(0));
            // sig = tf.nn.sigmoid(gamma)
            auto sigLayer = buildActivationLayer(
                gammaLayer->getOutput(0),
                network,
                tune_desc,
                layer.name + ".activation.sig",
                ActivationType::kSIGMOID);
            // scale = tf.reshape(sig, [-1, 1, 1, channels])
            // excitation = tf.multiply(scale, residual) + bias
            auto scaleLayer = network->addElementWise(
                *sigLayer->getOutput(0),
                *secondConvLayer->getOutput(0),
                ElementWiseOperation::kPROD
            );
            // excitation = tf.multiply(scale, residual) + bias
            auto excitationLayer = network->addElementWise(
                *scaleLayer->getOutput(0),
                *biasLayer->getOutput(0),
                ElementWiseOperation::kSUM
            );
            // (inputs + excitation)
            auto mergeLayer = network->addElementWise(
                *outputConv,
                *excitationLayer->getOutput(0),
                ElementWiseOperation::kSUM);
            // shared_output = tf.nn.relu(inputs + excitation)
            auto outputConvLayer = buildActivationLayer(
                mergeLayer->getOutput(0),
                network,
                tune_desc,
                layer.name + ".activation.final",
                ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else {
            auto weights = begin(layer.weights);
            auto biases = begin(layer.weights) + 1;
            if (layer.is_value) {
                auto ip1_val_weight = begin(layer.weights) + 2;
                auto ip1_val_bias = begin(layer.weights)   + 3;
                auto ip2_val_weight = begin(layer.weights) + 4;
                auto ip2_val_bias = begin(layer.weights)   + 5;
                // value_conv = tf.layers.conv2d(shared_output, filters=1, kernel_size=1, padding='same', use_bias=False)
                // value_conv = tf.layers.batch_normalization(value_conv, axis=1, momentum=.95, epsilon=1e-5, center=False, scale=False, fused=True, training=False)
                auto valueConvLayer = buildConvLayer(
                    outputConv,
                    layer.filter_size,
                    layer.weights_size[0],
                    weights[0],
                    layer.weights_size[1],
                    biases[0],
                    network,
                    tune_desc,
                    layer.name + ".conv",
                    layer.outputs);
                // value_conv = tf.nn.relu(value_conv)
                auto actValueLayer = buildActivationLayer(
                    valueConvLayer->getOutput(0),
                    network,
                    tune_desc,
                    layer.name + ".act",
                    ActivationType::kRELU);
                // value_conv = tf.reshape(value_conv, [-1, 1 * go.N * go.N])
                int32_t const mmInputs = static_cast<int32_t>(
                    actValueLayer->getOutput(0)->getDimensions().d[1]
                    * actValueLayer->getOutput(0)->getDimensions().d[2]
                    * actValueLayer->getOutput(0)->getDimensions().d[3]); 
                auto inputReshape = network->addShuffle(*actValueLayer->getOutput(0));
                int32_t const variable_batch = static_cast<int32_t>(
                    actValueLayer->getOutput(0)->getDimensions().d[0]);
                inputReshape->setReshapeDimensions(Dims{4, {variable_batch, mmInputs, 1, 1}});
                // value_fc_hidden = tf.layers.dense(value_conv, units=256)
                auto val1MatMulLayer = buildConvLayer(
                    inputReshape->getOutput(0),
                    1,
                    layer.weights_size[2],
                    ip1_val_weight[0],
                    layer.weights_size[3],
                    ip1_val_bias[0],
                    network,
                    tune_desc,
                    layer.name + ".val1.matmul",
                    Network::VALUE_LAYER);
                // value_fc_hidden = tf.nn.relu(value_fc_hidden)
                auto val1ActLayer = buildActivationLayer(
                    val1MatMulLayer->getOutput(0),
                    network,
                    tune_desc,
                    layer.name + ".val1.activation",
                    ActivationType::kRELU);
                // value_fc_hidden = tf.layers.dense(value_fc_hidden, units=1)
                auto val2MatMulLayer = buildConvLayer(
                    val1ActLayer->getOutput(0),
                    1,
                    layer.weights_size[4],
                    ip2_val_weight[0],
                    layer.weights_size[5],
                    ip2_val_bias[0],
                    network,
                    tune_desc,
                    layer.name + ".val2.matmul",
                    1);
                // value_fc_hidden = tf.reshape(value_fc_hidden, [-1])
                // value_output = tf.nn.tanh(value_fc_hidden)
                outValueLayer = buildActivationLayer(
                    val2MatMulLayer->getOutput(0),
                    network,
                    tune_desc,
                    layer.name + ".val.tanh",
                    ActivationType::kTANH);
            } else {
                auto ip_pol_weight = begin(layer.weights) + 2;
                auto ip_pol_bias = begin(layer.weights)   + 3;
                // policy_conv = tf.layers.conv2d(shared_output, filters=2, kernel_size=1, padding='same', use_bias=False)
                // policy_conv = tf.layers.batch_normalization(policy_conv, axis=1, momentum=.95, epsilon=1e-5, center=False, scale=False, fused=True, training=False)
                auto policyConvLayer = buildConvLayer(
                    outputConv,
                    layer.filter_size,
                    layer.weights_size[0],
                    weights[0],
                    layer.weights_size[1],
                    biases[0],
                    network,
                    tune_desc,
                    layer.name + ".conv",
                    layer.outputs);
                // policy_conv = tf.nn.relu(policy_conv)
                auto actPolicyLayer = buildActivationLayer(
                    policyConvLayer->getOutput(0),
                    network,
                    tune_desc,
                    layer.name + ".act",
                    ActivationType::kRELU);
                // policy_conv = tf.reshape(policy_conv, [-1, 2 * go.N * go.N])
                int32_t const mmInputs = static_cast<int32_t>(
                    actPolicyLayer->getOutput(0)->getDimensions().d[1]
                    * actPolicyLayer->getOutput(0)->getDimensions().d[2]
                    * actPolicyLayer->getOutput(0)->getDimensions().d[3]);
                auto inputReshape = network->addShuffle(*actPolicyLayer->getOutput(0));
                int32_t const variable_batch = static_cast<int32_t>(
                    actPolicyLayer->getOutput(0)->getDimensions().d[0]);
                inputReshape->setReshapeDimensions(Dims{4, {variable_batch, mmInputs, 1, 1}});
                // logits = tf.layers.dense(policy_conv, units=go.N * go.N + 1)
                auto polMatMulLayer = buildConvLayer(
                    inputReshape->getOutput(0),
                    1,
                    layer.weights_size[2],
                    ip_pol_weight[0],
                    layer.weights_size[3],
                    ip_pol_bias[0],
                    network,
                    tune_desc,
                    layer.name + ".pol.matmul",
                    POTENTIAL_MOVES);
                // policy_output = tf.nn.softmax(logits)
                outPolicyLayer = network->addSoftMax(*polMatMulLayer->getOutput(0));
                static_cast<ISoftMaxLayer*>(outPolicyLayer)->setAxes(1U << 1);
            }
        }
    }
    if (!outPolicyLayer || !outValueLayer) {
        std::cerr << "outPolicyLayer or outValueLayer is nullptr on constructNetwork." << std::endl;
        return false;
    }
    // Mark the outputs for the network
    auto outputPolicy = outPolicyLayer->getOutput(0);
    network->markOutput(*outputPolicy);
    outputPolicy->setName("OutputPolicy");
    outputPolicy->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    outputPolicy->setType(DataType::kFLOAT);

    auto outputValue = outValueLayer->getOutput(0);
    network->markOutput(*outputValue);
    outputValue->setName("OutputValue");
    outputValue->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    std::cout << "Done constructing network..." << std::endl;
    outputValue->setType(DataType::kFLOAT);
    return true;
}

template <typename net_t>
ITensor* BackendTRT<net_t>::initInputs(
    char const *inputName,
    TrtUniquePtr<INetworkDefinition>& network,
    const int channels,
    const int rows,
    const int cols) {

    ITensor* inputFeature;

    std::string_view name_str{inputName};
    inputFeature = network->addInput(
        inputName,
        DataType::kFLOAT,
        {4, {-1, channels, rows, cols}});
    assert(inputFeature != nullptr);
    inputFeature->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    return inputFeature;
}

template <typename net_t>
ILayer* BackendTRT<net_t>::buildConvLayer(
    ITensor* input,
    unsigned int filter_size,
    int64_t weights_size,
    void* weights,
    int64_t biases_size,
    void* biases,
    TrtUniquePtr<INetworkDefinition>& network,
    std::string& tune_desc,
    std::string op_name,
    unsigned int outputs) {

    tune_desc += strprintf(
        R"|("%s"(%d,%d,%d))|",
        op_name.c_str(),
        filter_size,
        filter_size,
        outputs);

    auto data_type = (typeid(net_t) == typeid(float)) ? DataType::kFLOAT : DataType::kHALF;
    // For convenience, both I/O tensors have 3 dimentions (in addition to batch), so that
    // matmul is mathmatically equivalent to a 2D convolution of 1x1 features and 1x1 kernels.
    auto convLayer = network->addConvolutionNd(
        *input,
        outputs,
        {2, {filter_size, filter_size}},
        {
            data_type,
            weights,
            weights_size
        },
        {
            data_type,
            biases,
            biases_size
        }
    );
    if (filter_size == 1) {
        return convLayer;
    }
    convLayer->setDilationNd({2, {1, 1}});
    convLayer->setPaddingMode(PaddingMode::kSAME_UPPER);
    return convLayer;
}

template <typename net_t>
ILayer* BackendTRT<net_t>::buildActivationLayer(
    ITensor* input,
    TrtUniquePtr<INetworkDefinition>& network,
    std::string& tune_desc,
    std::string op_name,
    ActivationType act_type) {

    tune_desc += strprintf(
        R"|("%s"(%d))|",
        op_name.c_str(),
        (int)act_type);

    auto activationLayer = network->addActivation(*input, act_type);
    return activationLayer;
}

template <typename net_t>
ILayer* BackendTRT<net_t>::applyGPoolLayer(
    ITensor* input,
    TrtUniquePtr<INetworkDefinition>& network) {

    IPoolingLayer* gpoolMeanLayer
        = network->addPoolingNd(
            *input,
            PoolingType::kAVERAGE,
            DimsHW{BOARD_SIZE, BOARD_SIZE});
    return gpoolMeanLayer;
}

template <typename net_t>
void BackendTRT<net_t>::push_weights(
    const size_t layer,
    const std::vector<float>& weights_float,
    const bool use_host_mem) {

    const std::vector<net_t> weights = from_float(weights_float);
    if (layer >= m_layers.size()) {
        m_layers.emplace_back(BackendTRTLayer());
    }
    // When TensorRT chooses a precision for a layer,
    // it automatically converts weights as necessary to run the layer
    if (use_host_mem) {
        void *host_mem;
        checkCUDA(cudaHostAlloc((void **)&host_mem,
                                weights.size() * sizeof(net_t),
                                cudaHostAllocMapped));
        memcpy(host_mem, (net_t *)&weights[0], weights.size() * sizeof(net_t));
        m_layers.back().weights.emplace_back(host_mem);
        m_layers.back().weights_size.emplace_back((int64_t)weights.size());
    } else {
        void *device_mem;
        checkCUDA(cudaMalloc(
            (void **)&device_mem,
            weights.size() * sizeof(net_t))
        );
        checkCUDA(cudaMemcpyAsync(
            device_mem,
            (net_t *)&weights[0],
            weights.size() * sizeof(net_t),
            cudaMemcpyHostToDevice,
            cudaStreamPerThread)
        );
        m_layers.back().weights.emplace_back(device_mem);
        m_layers.back().weights_size.emplace_back((int64_t)weights.size());
    }
}

template <typename net_t>
void BackendTRT<net_t>::push_input_convolution(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases) {

    size_t layer = get_layer_count();

    push_weights(layer, weights);
    push_weights(layer, biases);

    m_layers[layer].is_input_convolution = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
    m_layers[layer].name = "in." + std::to_string(layer);
}

template <typename net_t>
void BackendTRT<net_t>::push_residual(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights_1,
    const std::vector<float>& biases_1,
    const std::vector<float>& weights_2,
    const std::vector<float>& biases_2) {

    size_t layer = get_layer_count();

    push_weights(layer, weights_1);
    push_weights(layer, biases_1);
    push_weights(layer, weights_2);
    push_weights(layer, biases_2);

    m_layers[layer].is_residual_block = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
    m_layers[layer].name = "res." + std::to_string(layer);
}

template <typename net_t>
void BackendTRT<net_t>::push_residual_se(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights_1,
    const std::vector<float>& biases_1,
    const std::vector<float>& weights_2,
    const std::vector<float>& biases_2,
    const std::vector<float>& se_fc1_w,
    const std::vector<float>& se_fc1_b,
    const std::vector<float>& se_fc2_w,
    const std::vector<float>& se_fc2_b) {

    size_t layer = get_layer_count();

    push_weights(layer, weights_1);
    push_weights(layer, biases_1);
    push_weights(layer, weights_2);
    push_weights(layer, biases_2);
    push_weights(layer, se_fc1_w);
    push_weights(layer, se_fc1_b);
    push_weights(layer, se_fc2_w);
    push_weights(layer, se_fc2_b);

    m_layers[layer].is_residual_block = true;
    m_layers[layer].is_se_block = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
    m_layers[layer].name = "res." + std::to_string(layer);
}

template <typename net_t>
void BackendTRT<net_t>::push_convolve(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    const std::vector<float>& ip1_w,
    const std::vector<float>& ip1_b,
    const std::vector<float>& ip2_w,
    const std::vector<float>& ip2_b) {

    size_t layer = get_layer_count();

    push_weights(layer, weights);
    push_weights(layer, biases);
    if (outputs == Network::OUTPUTS_POLICY) {
        push_weights(layer, ip1_w);
        push_weights(layer, ip1_b);
        m_layers[layer].is_policy = true;
        m_layers[layer].outputs = outputs;
        m_layers[layer].channels = channels;
        m_layers[layer].filter_size = filter_size;
        m_layers[layer].name = "pol." + std::to_string(layer);
        return;
    }
    push_weights(layer, ip1_w);
    push_weights(layer, ip1_b);
    push_weights(layer, ip2_w);
    push_weights(layer, ip2_b);
    m_layers[layer].is_value = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].name = "val." + std::to_string(layer);

    if (build(m_num_worker_threads, cfg_batch_size)) {
        return;
    }
    exit(EXIT_FAILURE);
}

template <typename net_t>
void BackendTRT<net_t>::forward_activations(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    BackendTRTContext& trt_context,
    const size_t batch_size) {

    const auto inSize =
        batch_size *
        sizeof(float) *
        m_layers[0].channels *
        NUM_INTERSECTIONS;

    const auto pol_elements = batch_size * POTENTIAL_MOVES;
    const auto val_elements = batch_size * 1;
    auto search = trt_context.mBuffers.find("InputFeature");
    assert(search != trt_context.mBuffers.end());
    checkCUDA(cudaMemcpyAsync(
        search->second,
        (float*)&input[0],
        inSize,
        cudaMemcpyHostToDevice,
        cudaStreamPerThread)
    );
    trt_context.mContext->setInputShape(
        "InputFeature",
        Dims4(
            batch_size,
            m_layers[0].channels,
            BOARD_SIZE,
            BOARD_SIZE)
    );
    if (m_net_type == NetworkType::MINIGO_SE) {
        trt_context.mContext->setInputShape(
            "BatchSize",
            Dims4(
                batch_size,
                m_layers[1].channels,
                1,
                1)
        );
    }
    ASSERT(trt_context.mContext->enqueueV3(cudaStreamPerThread));
    search = trt_context.mBuffers.find("OutputPolicy");
    assert(search != trt_context.mBuffers.end());
    checkCUDA(cudaMemcpyAsync(
        &output_pol[0],
        search->second,
        pol_elements * sizeof(float),
        cudaMemcpyDeviceToHost,
        cudaStreamPerThread)
    );
    search = trt_context.mBuffers.find("OutputValue");
    assert(search != trt_context.mBuffers.end());
    checkCUDA(cudaMemcpyAsync(
        &output_val[0],
        search->second,
        val_elements * sizeof(float),
        cudaMemcpyDeviceToHost,
        cudaStreamPerThread)
    );
    // Asynchronously enqueue the inference work
    cudaStreamSynchronize(cudaStreamPerThread);
    trtErrorRecorder.clear();
}

template <typename net_t>
void BackendTRT<net_t>::forward(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    const int tid,
    const size_t batch_size) {

    forward_activations(input, output_pol, output_val, *m_context[tid], batch_size);
}

template class BackendTRT<float>;
template class BackendTRT<half_float::half>;
#endif
