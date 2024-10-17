/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Henrik Forsten
    Copyright (C) 2024 MAOmao000

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

#include "GTP.h"
#include "Utils.h"
#include "BackendTensorRT.h"

using namespace Utils;
using namespace nvinfer1;

// Define TRT entrypoints
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0
#include "NvInferRuntime.h"

template <typename net_t>
bool BackendTRT<net_t>::build(
    const int num_worker_threads,
    const size_t batch_size) {

    // Bump this when between program versions we want to forcibly drop old timing caches and plan caches.
    mTuneDesc = strprintf(
        R"|("salt"(%s%s)"model %s"(%s,%d,%d,%d))|",
        PROGRAM_VERSION_MAJOR,
        PROGRAM_VERSION_MINOR,
        typeid(net_t) == typeid(float) ? "single" : "half",
        "1.0",                    // modelVersion,
        Network::INPUT_CHANNELS,  // numInputChannels,
        cfg_execute_context,
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
    bool usingFP16 = false;
    if (builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
        usingFP16 = true;
    }

    const auto explicitBatchFlag =
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TrtUniquePtr<INetworkDefinition>(builder->createNetworkV2(explicitBatchFlag));
    if (!network) {
        std::cerr << "TensorRT backend: failed to create network definition" << std::endl;
        return false;
    }
    std::filesystem::path path = cfg_weightsfile;
    std::string filename = path.filename().string();
    auto ext_i = filename.find_last_of(".");
    std::string weightsfile = filename.substr(0, ext_i);
    network->setName(weightsfile.c_str());

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "TensorRT backend: failed to create optimization profile" << std::endl;
        return false;
    }
    if (cfg_execute_context == execute_t::SINGLE) {
        constructNetwork(network, profile, nullptr, batch_size);
        config->addOptimizationProfile(profile);
    } else {
        auto profile_n = builder->createOptimizationProfile();
        if (!profile_n) {
            std::cerr << "TensorRT backend: failed to create optimization profile" << std::endl;
            return false;
        }
        constructNetwork(network, profile, profile_n, batch_size);
        config->addOptimizationProfile(profile);
        config->addOptimizationProfile(profile_n);
    }

    if (this->m_device_prop.major >= 8) {
        // This is to avoid tactics that have shape switching overhead
        config->setTacticSources(1U << static_cast<uint32_t>(TacticSource::kJIT_CONVOLUTIONS));
        config->setBuilderOptimizationLevel(2);
    }
    // So that there are no concurrent kernel executions probably from other parts of code while profiling
    // See CUDA Runtime API document for more details related to NULL stream and synchronization behaviors
    config->setProfileStream(cudaStreamLegacy);
    // Typical runtime allocation is much less than the 1 GiB specified below
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);

    std::string plan;
    {
        static std::mutex tuneMutex;
        tuneMutex.lock();
        std::string cacheDir = Utils::leelaz_file("trtcache");
        std::filesystem::create_directory(cacheDir);
        assert(std::filesystem::exists(cacheDir));
        assert(std::filesystem::is_directory(cacheDir));

        uint8_t deviceHash[32];
        SHA2::get256(this->m_device_prop.name, deviceHash);

        // Truncated to 4 bytes
        char deviceIdent[4 * 2 + 1];
        for(int i = 0; i < 4; i++) {
            sprintf(deviceIdent + i * 2, "%02x", static_cast<unsigned char>(deviceHash[i]));
        }
        deviceIdent[sizeof(deviceIdent) - 1] = 0;

        std::string precision = typeid(net_t) == typeid(float) ? "single" : "half";
        std::string sep_char{std::filesystem::path::preferred_separator};

        uint8_t tuneHash[32];
        SHA2::get256(mTuneDesc.c_str(), tuneHash);
        // Truncated to 6 bytes
        char tuneIdent[6 * 2 + 1];
        for(int i = 0; i < 6; i++) {
            sprintf(tuneIdent + i * 2, "%02x", static_cast<unsigned char>(tuneHash[i]));
        }
        tuneIdent[sizeof(tuneIdent) - 1] = 0;

        if (cfg_cache_plan) {
            auto planCacheFile = strprintf(
                "%s%strt-%d_gpu-%s_tune-%s_net-%s_%s%s_%dx%d_%d_batch%d_fp%d_%s",
                cacheDir.c_str(),
                sep_char.c_str(),
                getInferLibVersion(),
                deviceIdent,
                tuneIdent,
                network->getName(),
                PROGRAM_VERSION_MAJOR,
                PROGRAM_VERSION_MINOR,
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_execute_context,
                batch_size,
                usingFP16 ? 16 : 32,
                precision.c_str()
            );
            std::string paramStr = strprintf(
                "_%d_%s_%s%s_%d_%d_%d_%d_%d_%s",
                getInferLibVersion(),
                deviceIdent,
                PROGRAM_VERSION_MAJOR,
                PROGRAM_VERSION_MINOR,
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_execute_context,
                batch_size,
                usingFP16 ? 16 : 32,
                precision.c_str()
            );
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
                    if (modelHash != this->m_model_hash) {
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
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
                    return false;
                }
                plan.insert(
                    plan.end(),
                    static_cast<char*>(planBuffer->data()),
                    static_cast<char*>(planBuffer->data()) + planBuffer->size()
                );
                if (this->m_model_hash.size() != 64) {
                    std::cerr << "Unexpected model hash size" << std::endl;
                    return false;
                }
                plan.insert(
                    plan.end(),
                    this->m_model_hash.begin(),
                    this->m_model_hash.end()
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
                tuneMutex.unlock();
            } else {
                tuneMutex.unlock();
                std::cout << "Using existing plan cache at " + planCacheFile << std::endl;
            }
        } else {
            auto timingCacheFile = strprintf(
                "%s%strt-%d_gpu-%s_tune-%s_%dx%d_%d_batch%d_fp%d_%s",
                cacheDir.c_str(),
                sep_char.c_str(),
                getInferLibVersion(),
                deviceIdent,
                tuneIdent,
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_execute_context,
                batch_size,
                usingFP16 ? 16 : 32,
                precision.c_str()
            );

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
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
                    return false;
                }
                auto serializedTimingCache = std::unique_ptr<IHostMemory>(
                    config->getTimingCache()->serialize());
                std::ofstream ofs;
                ofs.open(timingCacheFile, std::ios_base::out | std::ios_base::binary);
                ofs.write(static_cast<char*>(serializedTimingCache->data()), serializedTimingCache->size());
                ofs.close();
                std::cout << "Saved new timing cache to " << timingCacheFile << std::endl;
                tuneMutex.unlock();
            } else {
                tuneMutex.unlock();
                planBuffer.reset(builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    std::cerr << "TensorRT backend: failed to create plan" << std::endl;
                    return false;
                }
            }
            plan.insert(
                plan.end(),
                static_cast<char*>(planBuffer->data()),
                static_cast<char*>(planBuffer->data()) + planBuffer->size());
        }
    }

    for (auto i = 0; i < num_worker_threads; i++) {
        std::unique_ptr<IRuntime> runtime
            = std::unique_ptr<IRuntime>(createInferRuntime(cfg_logger.getTRTLogger()));
        if (!runtime) {
            std::cerr << "createInferRuntime error: " << std::endl;
            return false;
        }
        std::unique_ptr<ICudaEngine> engine
            = std::unique_ptr<ICudaEngine>(
                runtime->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine) {
            std::cerr << "deserializeCudaEngine error: " << std::endl;
            return false;
        }
        std::unique_ptr<BackendContext> context = std::make_unique<BackendContext>();
        context->mContext.reset(engine->createExecutionContext());
        if (cfg_execute_context == execute_t::DOUBLE) {
            context->mContext_n.reset(engine->createExecutionContext());
        }
        for (auto j = 0; j < engine->getNbIOTensors(); j++) {
            void* buffer = nullptr;
            auto name = engine->getIOTensorName(j);
            auto dims = engine->getTensorShape(name);
            std::string_view name_str{name};
            size_t size_byte;
            if (name_str == "BatchSize") {
                size_byte = sizeof(int32_t);
            } else if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
                size_byte = sizeof(float);
            } else {
                size_byte = sizeof(net_t);
            }
            size_t bytes = std::accumulate(
                dims.d + 1,
                dims.d + dims.nbDims,
                batch_size * size_byte,
                std::multiplies<size_t>());
            checkCUDA(cudaMalloc(&buffer, bytes));
            if (name_str == "BatchSize") {
                auto input_batch = std::vector<int32_t>(batch_size * this->m_layers[1].channels, 0);
                checkCUDA(cudaMemcpy(
                    buffer,
                    (int32_t*)&input_batch[0],
                    bytes,
                    cudaMemcpyHostToDevice));
            }
            context->mBuffers.emplace(std::make_pair(name, buffer));
            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                context->mContext->setInputTensorAddress(name, buffer);
                if (cfg_execute_context == execute_t::DOUBLE) {
                    context->mContext_n->setInputTensorAddress(name, buffer);
                }
            } else {
                context->mContext->setOutputTensorAddress(name, buffer);
                if (cfg_execute_context == execute_t::DOUBLE) {
                    context->mContext_n->setOutputTensorAddress(name, buffer);
                }
            }
        }
        context->mContext->setOptimizationProfileAsync(0, cudaStreamPerThread);
        if (cfg_execute_context == execute_t::DOUBLE) {
            context->mContext_n->setOptimizationProfileAsync(1, cudaStreamPerThread);
        }
        cudaStreamSynchronize(cudaStreamPerThread);
        context->m_buffers_allocated = true;
        mRuntime.emplace_back(std::move(runtime));
        mEngine.emplace_back(std::move(engine));
        this->m_context.emplace_back(std::move(context));
    }
    return true;
}

template <typename net_t>
void BackendTRT<net_t>::constructNetwork(
    TrtUniquePtr<INetworkDefinition>& network,
    IOptimizationProfile* profile,
    IOptimizationProfile* profile_n,
    const size_t batch_size) {

    ITensor* inputFeature = nullptr;
    ITensor* outputConv = nullptr;
    ILayer* outPolicyLayer = nullptr;
    ILayer* outValueLayer = nullptr;
    ILayer* shapeLayer = nullptr;
    IShapeLayer* inShapeLayer = nullptr;
    ICastLayer* castLayer = nullptr;

    if (this->m_net_type == NetworkType::MINIGO_SE) {
        auto batchSizeTensor = initInputs(
            "BatchSize",
            network,
            profile,
            profile_n,
            this->m_layers[1].channels,
            1,
            1,
            batch_size);

        // See. https://github.com/NVIDIA/TensorRT/issues/2282
        inShapeLayer = network->addShape(*batchSizeTensor);
        castLayer = network->addCast(*inShapeLayer->getOutput(0), DataType::kINT32);

        shapeLayer = network->addUnary(
            *castLayer->getOutput(0),
            UnaryOperation::kABS);
    }

    for (auto iter = std::begin(this->m_layers);
         iter != std::end(this->m_layers); iter++) {

        const auto& layer = *iter;
        if (layer.is_input_convolution) {
            inputFeature = initInputs(
                "InputFeature",
                network,
                profile,
                profile_n,
                layer.channels,
                BOARD_SIZE,
                BOARD_SIZE,
                batch_size);
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
                layer.name + ".conv",
                layer.outputs);
            auto outputConvLayer = buildActivationLayer(
                initialConvLayer->getOutput(0),
                network,
                layer.name + ".activation",
                ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else if (layer.is_residual_block && !layer.is_se_block) {
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
                layer.name + ".conv.first",
                layer.outputs);
            auto firstActivationConvLayer = buildActivationLayer(
                firstConvLayer->getOutput(0),
                network,
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
                layer.name + ".conv.second",
                layer.outputs);
            auto mergeLayer = network->addElementWise(
                *outputConv, *secondConvLayer->getOutput(0), ElementWiseOperation::kSUM);
            mergeLayer->setName((layer.name + ".merge").c_str());
            auto outputConvLayer = buildActivationLayer(
                mergeLayer->getOutput(0),
                network,
                layer.name + ".activation.final",
                ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else if (layer.is_residual_block && layer.is_se_block) {
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
                layer.name + ".conv.first",
                layer.outputs);
            auto firstActivationConvLayer = buildActivationLayer(
                firstConvLayer->getOutput(0),
                network,
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
                layer.name + ".conv.second",
                layer.outputs);
            // pool = tf.layers.average_pooling2d(residual, pool_size=go.N, strides=1, padding='valid')
            auto gpoolLayer = applyGPoolLayer(
                secondConvLayer->getOutput(0),
                network,
                layer.name + ".gpool");
            // fc1 = tf.layers.dense(pool, units=channels // 2)
            auto thirdMatMulLayer = buildConvLayer(
                gpoolLayer->getOutput(0),
                1,
                layer.weights_size[4],
                fc1_weights[0],
                layer.weights_size[5],
                fc1_biases[0],
                network,
                layer.name + ".conv.third",
                layer.outputs / 2);
            // squeeze = tf.nn.relu(fc1)
            auto thirdActivationMatLayer = buildActivationLayer(
                thirdMatMulLayer->getOutput(0),
                network,
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
                layer.name + ".conv.fourth",
                layer.outputs * 2);
            // gamma, bias = tf.split(fc2, 2, axis=3)
            auto gammaLayer = network->addSlice(
                *fourthMatMulLayer->getOutput(0),
                {4 ,{0, 0, 0, 0}},
                {4 ,{0, layer.channels, 1, 1}},
                {4 ,{1, 1, 1, 1}}
            );
            gammaLayer->setInput(2, *shapeLayer->getOutput(0));
            gammaLayer->setName((layer.name + ".gamma").c_str());
            // gamma, bias = tf.split(fc2, 2, axis=3)
            auto biasLayer = network->addSlice(
                *fourthMatMulLayer->getOutput(0),
                {4 ,{0, layer.channels, 0, 0}},
                {4 ,{0, layer.channels, 1, 1}},
                {4 ,{1, 1, 1, 1}}
            );
            biasLayer->setInput(2, *shapeLayer->getOutput(0));
            biasLayer->setName((layer.name + ".bias").c_str());
            // sig = tf.nn.sigmoid(gamma)
            auto sigLayer = buildActivationLayer(
                gammaLayer->getOutput(0),
                network,
                layer.name + ".activation.sig",
                ActivationType::kSIGMOID);
            sigLayer->setName((layer.name + ".sig").c_str());
            // scale = tf.reshape(sig, [-1, 1, 1, channels])
            // excitation = tf.multiply(scale, residual) + bias
            auto scaleLayer = network->addElementWise(
                *sigLayer->getOutput(0),
                *secondConvLayer->getOutput(0),
                ElementWiseOperation::kPROD
            );
            scaleLayer->setName((layer.name + ".scale").c_str());
            // excitation = tf.multiply(scale, residual) + bias
            auto excitationLayer = network->addElementWise(
                *scaleLayer->getOutput(0),
                *biasLayer->getOutput(0),
                ElementWiseOperation::kSUM
            );
            excitationLayer->setName((layer.name + ".excitation").c_str());
            // (inputs + excitation)
            auto mergeLayer = network->addElementWise(
                *outputConv,
                *excitationLayer->getOutput(0),
                ElementWiseOperation::kSUM);
            mergeLayer->setName((layer.name + ".merge").c_str());
            // shared_output = tf.nn.relu(inputs + excitation)
            auto outputConvLayer = buildActivationLayer(
                mergeLayer->getOutput(0),
                network,
                layer.name + ".activation.final",
                ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else {
            const auto niter = std::next(iter);
            auto weights = begin(layer.weights);
            auto biases = begin(layer.weights) + 1;
            if (niter == std::end(this->m_layers)) {
                //outValueLayer = buildConvLayer(
                auto valueConvLayer = buildConvLayer(
                    outputConv,
                    layer.filter_size,
                    layer.weights_size[0],
                    weights[0],
                    layer.weights_size[1],
                    biases[0],
                    network,
                    layer.name + ".conv",
                    layer.outputs);
                // value_conv = tf.nn.relu(value_conv)
                outValueLayer = buildActivationLayer(
                    valueConvLayer->getOutput(0),
                    network,
                    layer.name + ".act",
                    ActivationType::kRELU);
            } else {
                //outPolicyLayer = buildConvLayer(
                auto policyConvLayer = buildConvLayer(
                    outputConv,
                    layer.filter_size,
                    layer.weights_size[0],
                    weights[0],
                    layer.weights_size[1],
                    biases[0],
                    network,
                    layer.name + ".conv",
                    layer.outputs);
                // policy_conv = tf.nn.relu(policy_conv)
                outPolicyLayer = buildActivationLayer(
                    policyConvLayer->getOutput(0),
                    network,
                    layer.name + ".act",
                    ActivationType::kRELU);
            }
        }
    }
    // Mark the outputs for the network
    auto outputPolicy = outPolicyLayer->getOutput(0);
    network->markOutput(*outputPolicy);
    outputPolicy->setName("OutputPolicy");
    outputPolicy->setType(DataType::kFLOAT);
    outputPolicy->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    auto outputValue = outValueLayer->getOutput(0);
    network->markOutput(*outputValue);
    outputValue->setName("OutputValue");
    outputValue->setType(DataType::kFLOAT);
    outputValue->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    std::cout << "Done constructing network..." << std::endl;
}

template <typename net_t>
ITensor* BackendTRT<net_t>::initInputs(
    char const *inputName,
    TrtUniquePtr<INetworkDefinition>& network,
    IOptimizationProfile* profile,
    IOptimizationProfile* profile_n,
    const int channels,
    const int rows,
    const int cols,
    const size_t batch_size) {

    ITensor* inputFeature;

    std::string_view name_str{inputName};
    if (name_str == "BatchSize") {
        inputFeature = network->addInput(
            inputName,
            DataType::kINT32,
            {4, {-1, channels, rows, cols}});
    } else if (typeid(net_t) == typeid(float)) {
        inputFeature = network->addInput(
            inputName,
            DataType::kFLOAT,
            {4, {-1, channels, rows, cols}});
    } else {
        inputFeature = network->addInput(
            inputName,
            DataType::kHALF,
            {4, {-1, channels, rows, cols}});
    }
    assert(inputFeature != nullptr);
    inputFeature->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    if (cfg_execute_context == execute_t::SINGLE) {
        profile->setDimensions(
            inputName,
            OptProfileSelector::kMIN,
            Dims4(1, channels, rows, cols));
        profile->setDimensions(
            inputName,
            OptProfileSelector::kOPT,
            Dims4(batch_size, channels, rows, cols));
        profile->setDimensions(
            inputName,
            OptProfileSelector::kMAX,
            Dims4(batch_size, channels, rows, cols));
    } else {
        profile->setDimensions(
            inputName,
            OptProfileSelector::kMIN,
            Dims4(1, channels, rows, cols));
        profile->setDimensions(
            inputName,
            OptProfileSelector::kOPT,
            Dims4(1, channels, rows, cols));
        profile->setDimensions(
            inputName,
            OptProfileSelector::kMAX,
            Dims4(1, channels, rows, cols));
        profile_n->setDimensions(
            inputName,
            OptProfileSelector::kMIN,
            Dims4(batch_size, channels, rows, cols));
        profile_n->setDimensions(
            inputName,
            OptProfileSelector::kOPT,
            Dims4(batch_size, channels, rows, cols));
        profile_n->setDimensions(
            inputName,
            OptProfileSelector::kMAX,
            Dims4(batch_size, channels, rows, cols));
    }
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
    std::string op_name,
    unsigned int outputs) {

    mTuneDesc += strprintf(
        R"|("%s"(%d,%d,%d))|",
        op_name.c_str(),
        filter_size,
        filter_size,
        outputs);

    // For convenience, both I/O tensors have 3 dimentions (in addition to batch), so that
    // matmul is mathmatically equivalent to a 2D convolution of 1x1 features and 1x1 kernels.
    IConvolutionLayer *convLayer;
    convLayer = network->addConvolutionNd(
        *input,
        outputs,
        {2, {filter_size, filter_size}},
        {
            DataType::kFLOAT,
            weights,
            weights_size
        },
        {
            DataType::kFLOAT,
            biases,
            biases_size
        }
    );
    convLayer->setName(op_name.c_str());
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
    std::string op_name,
    ActivationType act_type) {

    mTuneDesc += strprintf(
        R"|("%s"(%d))|",
        op_name.c_str(),
        (int)act_type);

    auto activationLayer = network->addActivation(*input, act_type);
    activationLayer->setName(op_name.c_str());
    return activationLayer;
}

template <typename net_t>
ILayer* BackendTRT<net_t>::applyGPoolLayer(
    ITensor* input,
    TrtUniquePtr<INetworkDefinition>& network,
    std::string op_name) {

    IPoolingLayer* gpoolMeanLayer
        = network->addPoolingNd(
            *input,
            PoolingType::kAVERAGE,
            DimsHW{BOARD_SIZE, BOARD_SIZE});
    auto gpoolMeanLayerName = op_name + "/gpmean";
    gpoolMeanLayer->setName(gpoolMeanLayerName.c_str());
    return gpoolMeanLayer;
}

template <typename net_t>
void BackendTRT<net_t>::push_weights(
    const size_t layer,
    const std::vector<float>& weights) {

    if (layer >= this->m_layers.size()) {
        this->m_layers.emplace_back(BackendLayer());
    }
    // When TensorRT chooses a precision for a layer,
    // it automatically converts weights as necessary to run the layer
    void *host_mem;
    checkCUDA(cudaHostAlloc(
        (void **)&host_mem,
        weights.size() * sizeof(float),
        cudaHostAllocMapped)
    );
    memcpy(host_mem, (float *)&weights[0], weights.size() * sizeof(float));
    this->m_layers.back().weights.emplace_back(host_mem);
    this->m_layers.back().weights_size.emplace_back((int64_t)weights.size());
}

template <typename net_t>
void BackendTRT<net_t>::push_weights_col_major(
    const size_t layer,
    const std::vector<float>& weights,
    const int row,
    const int column,
    const int channels) {

    if (layer >= this->m_layers.size()) {
        this->m_layers.emplace_back(BackendLayer());
    }
    // When TensorRT chooses a precision for a layer,
    // it automatically converts weights as necessary to run the layer
    // Transpose from model's CK to TensorRT's KC
    auto weightSize = weights.size() * sizeof(float);
    auto transposed_weights = std::vector<float>(weights.size());
    for (int ch = 0; ch < channels; ch++) {
        for (int i = 0; i < column; i++) {
            for (int j = 0; j < row; j++) {
                transposed_weights[ch * column * row + j * column + i] =
                    (float)weights[ch * column * row + i * row + j];
            }
        }
    }
    void *host_mem;
    checkCUDA(cudaHostAlloc(
        (void **)&host_mem,
        weightSize,
        cudaHostAllocMapped)
    );
    memcpy(host_mem, (float*)&transposed_weights[0], weightSize);
    this->m_layers.back().weights.emplace_back(host_mem);
    this->m_layers.back().weights_size.emplace_back((int64_t)weights.size());
}

template <typename net_t>
void BackendTRT<net_t>::push_input_convolution(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    const float scale) {               // Dummy arguments for inheritance usage

    (void)scale;

    size_t layer = get_layer_count();

    push_weights(layer, weights);
    push_weights(layer, biases);

    this->m_layers[layer].is_input_convolution = true;
    this->m_layers[layer].outputs = outputs;
    this->m_layers[layer].filter_size = filter_size;
    this->m_layers[layer].channels = channels;
    this->m_layers[layer].name = "in." + std::to_string(layer);
}

template <typename net_t>
void BackendTRT<net_t>::push_residual(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights_1,
    const std::vector<float>& biases_1,
    const std::vector<float>& weights_2,
    const std::vector<float>& biases_2,
    const float scale_1,   // Dummy arguments for inheritance usage
    const float scale_2,   // Dummy arguments for inheritance usage
    const float scale_3) { // Dummy arguments for inheritance usage

    (void)scale_1;
    (void)scale_2;
    (void)scale_3;

    size_t layer = get_layer_count();

    push_weights(layer, weights_1);
    push_weights(layer, biases_1);
    push_weights(layer, weights_2);
    push_weights(layer, biases_2);

    this->m_layers[layer].is_residual_block = true;
    this->m_layers[layer].outputs = outputs;
    this->m_layers[layer].filter_size = filter_size;
    this->m_layers[layer].channels = channels;
    this->m_layers[layer].name = "res." + std::to_string(layer);
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
    const std::vector<float>& se_fc2_b,
    const float scale_1,   // Dummy arguments for inheritance usage
    const float scale_2,   // Dummy arguments for inheritance usage
    const float scale_3) { // Dummy arguments for inheritance usage

    (void)scale_1;
    (void)scale_2;
    (void)scale_3;

    size_t layer = get_layer_count();

    push_weights(layer, weights_1);
    push_weights(layer, biases_1);
    push_weights(layer, weights_2);
    push_weights(layer, biases_2);
    push_weights(layer, se_fc1_w);
    push_weights(layer, se_fc1_b);
    push_weights(layer, se_fc2_w);
    push_weights(layer, se_fc2_b);

    this->m_layers[layer].is_residual_block = true;
    this->m_layers[layer].is_se_block = true;
    this->m_layers[layer].outputs = outputs;
    this->m_layers[layer].filter_size = filter_size;
    this->m_layers[layer].channels = channels;
    this->m_layers[layer].name = "res." + std::to_string(layer);
}

template <typename net_t>
void BackendTRT<net_t>::push_convolve(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases) {

    size_t layer = get_layer_count();

    push_weights(layer, weights);
    push_weights(layer, biases);
    this->m_layers[layer].outputs = outputs;
    this->m_layers[layer].channels = channels;
    this->m_layers[layer].filter_size = filter_size;
    if (outputs == Network::OUTPUTS_POLICY) {
        this->m_layers[layer].is_policy = true;
        this->m_layers[layer].name = "pol." + std::to_string(layer);
        return;
    }
    this->m_layers[layer].is_value = true;
    this->m_layers[layer].name = "val." + std::to_string(layer);

    if (build(this->m_num_worker_threads, cfg_batch_size)) {
        return;
    }
    exit(EXIT_FAILURE);
}

template <typename net_t>
void BackendTRT<net_t>::forward_activations(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    BackendContext& cudnn_context,
    const int tid,
    const size_t batch_size) {

    const auto inSize =
        batch_size *
        sizeof(net_t) *
        this->m_layers[0].channels *
        NUM_INTERSECTIONS;

    size_t pol_elements;
    size_t val_elements;
    pol_elements =
        batch_size *
        this->m_layers[this->m_layers.size() - 2].outputs *
        NUM_INTERSECTIONS;
    val_elements =
        batch_size *
        this->m_layers.back().outputs *
        NUM_INTERSECTIONS;
    std::vector<net_t> pol_net_t = std::vector<net_t>(pol_elements);
    std::vector<net_t> val_net_t = std::vector<net_t>(val_elements);
    auto search = cudnn_context.mBuffers.find("InputFeature");
    assert(search != cudnn_context.mBuffers.end());
    if (typeid(net_t) == typeid(float)) {
        checkCUDA(cudaMemcpyAsync(
            search->second,
            (net_t*)&input[0],
            inSize,
            cudaMemcpyHostToDevice,
            this->m_streams[tid])
        );
    } else {
        auto input_net_t =
            std::vector<net_t>(
                batch_size * this->m_layers[0].channels * NUM_INTERSECTIONS);
        std::copy(input.begin(), input.end(), input_net_t.begin());
        checkCUDA(cudaMemcpyAsync(
            search->second,
            (net_t*)&input_net_t[0],
            inSize,
            cudaMemcpyHostToDevice,
            this->m_streams[tid])
        );
    }
    if (cfg_execute_context == execute_t::SINGLE || batch_size == 1) {
        cudnn_context.mContext->setInputShape(
            "InputFeature",
            Dims4(
                batch_size,
                this->m_layers[0].channels,
                BOARD_SIZE,
                BOARD_SIZE)
        );
    } else {
        cudnn_context.mContext_n->setInputShape(
            "InputFeature",
            Dims4(
                batch_size,
                this->m_layers[0].channels,
                BOARD_SIZE,
                BOARD_SIZE)
        );
    }
    if (this->m_net_type == NetworkType::MINIGO_SE) {
        if (cfg_execute_context == execute_t::SINGLE || batch_size == 1) {
            cudnn_context.mContext->setInputShape(
                "BatchSize",
                Dims(
                    {
                        4,
                        {
                            (unsigned int)batch_size,
                            this->m_layers[1].channels,
                            1,
                            1
                        }
                    }
                )
            );
        } else {
            cudnn_context.mContext_n->setInputShape(
                "BatchSize",
                Dims(
                    {
                        4,
                        {
                            (unsigned int)batch_size,
                            this->m_layers[1].channels,
                            1,
                            1
                        }
                    }
                )
            );
        }
    }
    if (cfg_execute_context == execute_t::SINGLE || batch_size == 1) {
        ASSERT(cudnn_context.mContext->enqueueV3(this->m_streams[tid]));
    } else {
        ASSERT(cudnn_context.mContext_n->enqueueV3(this->m_streams[tid]));
    }
    search = cudnn_context.mBuffers.find("OutputPolicy");
    assert(search != cudnn_context.mBuffers.end());
    checkCUDA(cudaMemcpyAsync(
        &output_pol[0],
        search->second,
        pol_elements * sizeof(float),
        cudaMemcpyDeviceToHost,
        this->m_streams[tid])
    );
    search = cudnn_context.mBuffers.find("OutputValue");
    assert(search != cudnn_context.mBuffers.end());
    checkCUDA(cudaMemcpyAsync(
        &output_val[0],
        search->second,
        val_elements * sizeof(float),
        cudaMemcpyDeviceToHost,
        this->m_streams[tid])
    );
    // Asynchronously enqueue the inference work
    cudaStreamSynchronize(this->m_streams[tid]);
}

template class BackendTRT<float>;
template class BackendTRT<half_float::half>;

#endif