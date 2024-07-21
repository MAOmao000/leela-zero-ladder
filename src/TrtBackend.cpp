/*
    This file is part of Leela Zero.
    Copyright (C) 2024 Maomao000

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
#include "GTP.h"
#include "Utils.h"
#include "CuDNN.h"

using namespace Utils;
namespace trt {
    Logger gLogger;
}
struct StringError : public std::exception {
    std::string message;
    StringError(const char* m)
        :exception(),message(m)
    {}
    StringError(const std::string& m)
        :exception(),message(m)
    {}

    const char* what() const throw () final
        {return message.c_str();}
};

template <typename net_t>
bool TrtResNet<net_t>::build() {
    // Bump this when between program versions we want to forcibly drop old timing caches and plan caches.
    static constexpr int tuneSalt = 4;
    if (typeid(net_t) == typeid(float)) {
        mTuneDesc = strprintf(
            R"|("salt"(%d)"model float"(%s,%d,%d))|",
            tuneSalt,
            "1.0",                    // modelVersion,
            cfg_num_threads,
            Network::INPUT_CHANNELS); // numInputChannels,
    } else {
        mTuneDesc = strprintf(
            R"|("salt"(%d)"model half"(%s,%d,%d))|",
            tuneSalt,
            "1.0",                    // modelVersion,
            cfg_num_threads,
            Network::INPUT_CHANNELS); // numInputChannels,
    }
    auto builder
        = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt::gLogger.getTRTLogger()));
    if (!builder) {
        std::cerr << "TensorRT backend: failed to create builder" << std::endl;
        return false;
    }
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "TensorRT backend: failed to create builder config" << std::endl;
        return false;
    }
    bool usingFP16 = false;
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        usingFP16 = true;
    }
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<int>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
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
    if(!profile) {
        std::cerr << "TensorRT backend: failed to create optimization profile" << std::endl;
        return false;
    }

    constructNetwork(builder, network, config, profile);
    config->addOptimizationProfile(profile);

    if (m_cudnn.m_device_prop.major >= 8) {
        // This is to avoid tactics that have shape switching overhead
        config->setTacticSources(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kJIT_CONVOLUTIONS));
        config->setBuilderOptimizationLevel(2);
    }
    // So that there are no concurrent kernel executions probably from other parts of code while profiling
    // See CUDA Runtime API document for more details related to NULL stream and synchronization behaviors
    config->setProfileStream(cudaStreamLegacy);
    // Typical runtime allocation is much less than the 1 GiB specified below
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);

    std::string plan;
    {
        static std::mutex tuneMutex;
        tuneMutex.lock();
        //std::string cacheDir = "trtcache";
        std::string cacheDir = Utils::leelaz_file("trtcache");
        bool result = std::filesystem::create_directory(cacheDir);
        assert(std::filesystem::exists(cacheDir));
        assert(std::filesystem::is_directory(cacheDir));

        uint8_t deviceHash[32];
        SHA2::get256(m_cudnn.m_device_prop.name, deviceHash);

        // Truncated to 4 bytes
        char deviceIdent[4 * 2 + 1];
        for(int i = 0; i < 4; i++) {
            sprintf(deviceIdent + i * 2, "%02x", static_cast<unsigned char>(deviceHash[i]));
        }
        deviceIdent[sizeof(deviceIdent) - 1] = 0;

        if (cfg_cache_plan) {
            auto planCacheFile = strprintf(
                "%s/trt-%d_gpu-%s_net-%s_%d_%s%dx%d_batch%d_fp%d",
                cacheDir.c_str(),
                getInferLibVersion(),
                deviceIdent,
                network->getName(),
                tuneSalt,
                "exact",
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_batch_size,
                usingFP16 ? 16 : 32
            );
            std::string paramStr = strprintf(
                "_%d_%s_%d_%s_%d_%d_%d_%d",
                getInferLibVersion(),
                deviceIdent,
                tuneSalt,
                "exact",
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_batch_size,
                usingFP16 ? 16 : 32
            );
            try {
                plan = readFileBinary(planCacheFile);
            } catch(const StringError& e) {
                (void) e;
            };

            if (plan.size() > 0) {
                if (plan.size() < 64 + paramStr.size()) {
                    std::cout << "Could not parse plan, unexpected size in " + planCacheFile << std::endl;
                    plan.clear();
                } else {
                    std::string cachedParamStr = plan.substr(plan.size() - paramStr.size());
                    std::string modelHash = plan.substr(plan.size() - 64 - paramStr.size(), 64);
                    if (modelHash != m_cudnn.m_model_hash) {
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
                auto planBuffer = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    throw StringError("TensorRT backend: failed to create plan");
                }
                plan.insert(
                    plan.end(),
                    static_cast<char*>(planBuffer->data()),
                    static_cast<char*>(planBuffer->data()) + planBuffer->size()
                );
                if (m_cudnn.m_model_hash.size() != 64) {
                    throw StringError("Unexpected model hash size");
                }
                plan.insert(
                    plan.end(),
                    m_cudnn.m_model_hash.begin(),
                    m_cudnn.m_model_hash.end()
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
            uint8_t tuneHash[32];
            SHA2::get256(mTuneDesc.c_str(), tuneHash);
            // Truncated to 6 bytes
            char tuneIdent[6 * 2 + 1];
            for(int i = 0; i < 6; i++) {
                sprintf(tuneIdent + i * 2, "%02x", static_cast<unsigned char>(tuneHash[i]));
            }
            tuneIdent[sizeof(tuneIdent) - 1] = 0;

            auto timingCacheFile = strprintf(
                "%s/trt-%d_gpu-%s_tune-%s_%s%dx%d_batch%d_fp%d",
                cacheDir.c_str(),
                getInferLibVersion(),
                deviceIdent,
                tuneIdent,
                "exact",
                BOARD_SIZE,
                BOARD_SIZE,
                cfg_batch_size,
                usingFP16 ? 16 : 32);

            std::string timingCacheBlob;
            try {
                timingCacheBlob = readFileBinary(timingCacheFile);
            } catch (const StringError& e) {
                (void) e;
            };
            if (timingCacheBlob.size() > 0)
                std::cout << "Using existing timing cache at " << timingCacheFile << std::endl;
            else
                std::cout << "Creating new timing cache" << std::endl;

            auto timingCache =
                std::unique_ptr<nvinfer1::ITimingCache>(
                    config->createTimingCache(timingCacheBlob.data(), timingCacheBlob.size()));
            auto invalidTimingCache = !config->setTimingCache(*timingCache, false);
            if (invalidTimingCache) {
                std::cout << "Invalid timing cache, using new one instead" << std::endl;
                timingCache.reset(config->createTimingCache(nullptr, 0));
                config->setTimingCache(*timingCache, false);
            }

            std::unique_ptr<nvinfer1::IHostMemory> planBuffer;
            if (invalidTimingCache || !timingCacheBlob.size()) {
                planBuffer.reset(builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    throw StringError("TensorRT backend: failed to create plan");
                }
                auto serializedTimingCache = std::unique_ptr<nvinfer1::IHostMemory>(
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
                    throw StringError("TensorRT backend: failed to create plan");
                }
            }
            plan.insert(
                plan.end(),
                static_cast<char*>(planBuffer->data()),
                static_cast<char*>(planBuffer->data()) + planBuffer->size());
        }
    }

    mRuntime.reset(nvinfer1::createInferRuntime(trt::gLogger.getTRTLogger()));
    if (!mRuntime) {
        std::cerr << "createInferRuntime error: " << network << std::endl;
        return false;
    }

    mEngine.reset(mRuntime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!mEngine) {
        std::cerr << "deserializeCudaEngine error: " << network << std::endl;
        return false;
    }
    return true;
}

template <typename net_t>
void TrtResNet<net_t>::constructNetwork(TrtUniquePtr<nvinfer1::IBuilder>& builder,
                                        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
                                        TrtUniquePtr<nvinfer1::IBuilderConfig>& config,
                                        nvinfer1::IOptimizationProfile* profile) {
    nvinfer1::ITensor* inputFeature;
    nvinfer1::ILayer* initialConvLayer;
    nvinfer1::ITensor* outputConv = nullptr;
    nvinfer1::ILayer* policyConvLayer = nullptr;
    nvinfer1::ILayer* valueConvLayer = nullptr;

    for (auto iter = std::begin(m_cudnn_net.m_layers);
         iter != std::end(m_cudnn_net.m_layers); iter++) {

        const auto& layer = *iter;
        if (layer.is_input_convolution) {
            inputFeature = initInputs(network, layer, profile);
            auto conv_weights = begin(layer.weights);
            auto conv_biases = begin(layer.weights) + 1;
            initialConvLayer = buildConvLayer(
                inputFeature,
                layer.filter_size,
                layer.weights_size[0],
                conv_weights[0],
                layer.weights_size[1],
                conv_biases[0],
                network,
                layer.name + ".conv",
                layer.channels,
                layer.outputs);
            auto outputConvLayer = buildActivationLayer(
                initialConvLayer->getOutput(0),
                network,
                layer.name + ".activation",
                nvinfer1::ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else if (layer.is_residual_block && !layer.is_se_block) {
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases = begin(layer.weights) + 3;
            auto firstConvLayer = buildConvLayer(
                outputConv,
                layer.filter_size,
                layer.weights_size[0],
                conv1_weights[0],
                layer.weights_size[1],
                conv1_biases[0],
                network,
                layer.name + ".conv.first",
                layer.channels,
                layer.outputs);
            auto firstActivationConvLayer = buildActivationLayer(
                firstConvLayer->getOutput(0),
                network,
                layer.name + ".activation.first",
                nvinfer1::ActivationType::kRELU);
            auto secondConvLayer = buildConvLayer(
                firstActivationConvLayer->getOutput(0),
                layer.filter_size,
                layer.weights_size[2],
                conv2_weights[0],
                layer.weights_size[3],
                conv2_biases[0],
                network,
                layer.name + ".conv.second",
                layer.channels,
                layer.outputs);
            auto mergeLayer = network->addElementWise(
                *outputConv, *secondConvLayer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
            mergeLayer->setName((layer.name + ".merge").c_str());
            auto outputConvLayer = buildActivationLayer(
                mergeLayer->getOutput(0),
                network,
                layer.name + ".activation.final",
                nvinfer1::ActivationType::kRELU);
            outputConv = outputConvLayer->getOutput(0);
        } else {
            const auto niter = std::next(iter);
            if (niter == std::end(m_cudnn_net.m_layers)) {
                valueConvLayer = buildConvLayer(
                    outputConv,
                    layer.filter_size,
                    layer.weights_size[0],
                    layer.weights[0],
                    0,
                    nullptr,
                    network,
                    layer.name + ".value",
                    layer.channels,
                    layer.outputs);
            } else {
                policyConvLayer = buildConvLayer(
                    outputConv,
                    layer.filter_size,
                    layer.weights_size[0],
                    layer.weights[0],
                    0,
                    nullptr,
                    network,
                    layer.name + ".policy",
                    layer.channels,
                    layer.outputs);
            }
        }
    }
    // Mark the outputs for the network
    auto outputPolicy = policyConvLayer->getOutput(0);
    network->markOutput(*outputPolicy);
    outputPolicy->setName("OutputPolicy");
    if (typeid(net_t) == typeid(float)) {
        outputPolicy->setType(nvinfer1::DataType::kFLOAT);
    } else {
        outputPolicy->setType(nvinfer1::DataType::kHALF);
    }
    outputPolicy->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    auto outputValue = valueConvLayer->getOutput(0);
    network->markOutput(*outputValue);
    outputValue->setName("OutputValue");
    if (typeid(net_t) == typeid(float)) {
        outputValue->setType(nvinfer1::DataType::kFLOAT);
    } else {
        outputValue->setType(nvinfer1::DataType::kHALF);
    }
    outputValue->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    std::cout << "Done constructing network..." << std::endl;
}

template <typename net_t>
nvinfer1::ITensor* TrtResNet<net_t>::initInputs(
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    const CuDNN_Layer layer,
    nvinfer1::IOptimizationProfile* profile) {

    auto numInChannels = layer.channels;
    auto numOutChannels = layer.outputs;
    auto nnYLen = BOARD_SIZE;
    auto nnXLen = BOARD_SIZE;
    auto maxBatchSize = cfg_batch_size;
    nvinfer1::ILayer* dataOut{nullptr};
    nvinfer1::ITensor* inputFeature;
    if (typeid(net_t) == typeid(float)) {
        inputFeature
            = network->addInput("InputFeature",
                                nvinfer1::DataType::kFLOAT,
                                {4, {-1, numInChannels, nnYLen, nnXLen}});
    } else {
        inputFeature
            = network->addInput("InputFeature",
                                nvinfer1::DataType::kHALF,
                                {4, {-1, numInChannels, nnYLen, nnXLen}});
    }
    assert(inputFeature != nullptr);

    inputFeature->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    profile->setDimensions("InputFeature",
                           nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(1, numInChannels, nnYLen, nnXLen));
    profile->setDimensions("InputFeature",
                           nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(maxBatchSize, numInChannels, nnYLen, nnXLen));
    profile->setDimensions("InputFeature",
                           nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(maxBatchSize, numInChannels, nnYLen, nnXLen));
    return inputFeature;
}

template <typename net_t>
nvinfer1::ILayer* TrtResNet<net_t>::buildConvLayer(
    nvinfer1::ITensor* input,
    unsigned int filter_size,
    int64_t weights_size,
    void* weights,
    int64_t biases_size,
    void* biases,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::string op_name,
    unsigned int channels,
    unsigned int outputs) {

    auto dilationX = 1;
    auto dilationY = 1;

    mTuneDesc += strprintf(
        R"|("%s"(%d,%d,%d,%d,%d,%d))|",
        op_name.c_str(),
        filter_size,
        filter_size,
        channels,
        outputs,
        dilationX,
        dilationY);

    nvinfer1::IConvolutionLayer *convLayer;
    if (biases_size > 0) {
        if (typeid(net_t) == typeid(float)) {
            convLayer = network->addConvolutionNd(
                *input,
                outputs,
                {2, {filter_size, filter_size}},
                {
                    nvinfer1::DataType::kFLOAT,
                    weights,
                    weights_size
                },
                {
                    nvinfer1::DataType::kFLOAT,
                    biases,
                    biases_size
                }
            );
        } else {
            convLayer = network->addConvolutionNd(
                *input,
                outputs,
                {2, {filter_size, filter_size}},
                {
                    nvinfer1::DataType::kHALF,
                    weights,
                    weights_size
                },
                {
                    nvinfer1::DataType::kHALF,
                    biases,
                    biases_size
                }
            );
        }
    } else {
        if (typeid(net_t) == typeid(float)) {
            convLayer = network->addConvolutionNd(
                *input,
                outputs,
                {2, {filter_size, filter_size}},
                {
                    nvinfer1::DataType::kFLOAT,
                    weights,
                    weights_size
                },
                {nvinfer1::DataType::kFLOAT, nullptr, 0}
            );
        } else {
            convLayer = network->addConvolutionNd(
                *input,
                outputs,
                {2, {filter_size, filter_size}},
                {
                    nvinfer1::DataType::kHALF,
                    weights,
                    weights_size
                },
                {nvinfer1::DataType::kHALF, nullptr, 0}
            );
        }
    }
    convLayer->setDilationNd({2, {dilationY, dilationX}});
    convLayer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    convLayer->setName(op_name.c_str());

    return convLayer;
}

template <typename net_t>
nvinfer1::IScaleLayer* TrtResNet<net_t>::buildMatBiasLayer(
    nvinfer1::ITensor* input,
    int64_t biases_size,
    void* biases,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::string op_name,
    unsigned int channels,
    unsigned int outputs) {

    mTuneDesc += strprintf(
        R"|("%s"(%d))|",
        op_name.c_str(),
        channels);

    nvinfer1::IScaleLayer *matBiasLayer;
    if (typeid(net_t) == typeid(float)) {
        matBiasLayer = network->addScale(
            *input,
            nvinfer1::ScaleMode::kCHANNEL,
            {
                nvinfer1::DataType::kFLOAT,
                biases,
                biases_size
            },
            {nvinfer1::DataType::kFLOAT, nullptr, 0},
            {nvinfer1::DataType::kFLOAT, nullptr, 0}
        );
    } else {
        matBiasLayer = network->addScale(
            *input,
            nvinfer1::ScaleMode::kCHANNEL,
            {
                nvinfer1::DataType::kHALF,
                biases,
                biases_size
            },
            {nvinfer1::DataType::kHALF, nullptr, 0},
            {nvinfer1::DataType::kHALF, nullptr, 0}
        );
    }
    matBiasLayer->setName(op_name.c_str());

    return matBiasLayer;
}

template <typename net_t>
nvinfer1::ILayer* TrtResNet<net_t>::buildActivationLayer(
    nvinfer1::ITensor* input,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::string op_name,
    nvinfer1::ActivationType act_type) {

    mTuneDesc += strprintf(
        R"|("%s"(%d))|",
        op_name.c_str(),
        (int)act_type);

    auto activationLayer = network->addActivation(*input, act_type);
    activationLayer->setName(op_name.c_str());
    return activationLayer;
}

template <typename net_t>
nvinfer1::ILayer* TrtResNet<net_t>::applyGPoolLayer(
    nvinfer1::ITensor* input,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::string op_name) {

    nvinfer1::ILayer* gpoolMeanLayer
        = network->addReduce(*input, nvinfer1::ReduceOperation::kAVG, 1U << 2 | 1U << 3, true);
    auto gpoolMeanLayerName = op_name + "/gpmean";
    gpoolMeanLayer->setName(gpoolMeanLayerName.c_str());
    return gpoolMeanLayer;
}

template <typename net_t>
nvinfer1::ILayer* TrtResNet<net_t>::buildMatMulLayer(
    nvinfer1::ITensor* input,
    int64_t weights_size,
    void* weights,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::string op_name,
    unsigned int channels,
    unsigned int outputs) {

    mTuneDesc += strprintf(
        R"|("%s"(%d,%d))|",
        op_name.c_str(),
        channels,
        outputs);

    // For convenience, both I/O tensors have 3 dimentions (in addition to batch), so that
    // matmul is mathmatically equivalent to a 2D convolution of 1x1 features and 1x1 kernels.
    nvinfer1::IConvolutionLayer *matMulLayer;
    if (typeid(net_t) == typeid(float)) {
        matMulLayer = network->addConvolutionNd(
            *input,
            outputs,
            {2, {1, 1}},
            {
                nvinfer1::DataType::kFLOAT,
                weights,
                weights_size
            },
            {nvinfer1::DataType::kFLOAT, nullptr, 0}
        );
    } else {
        matMulLayer = network->addConvolutionNd(
            *input,
            outputs,
            {2, {1, 1}},
            {
                nvinfer1::DataType::kHALF,
                weights,
                weights_size
            },
            {nvinfer1::DataType::kHALF, nullptr, 0}
        );
    }
    matMulLayer->setName(op_name.c_str());

    return matMulLayer;
}

template class TrtResNet<float>;
#ifdef USE_HALF
template class TrtResNet<half_float::half>;
#endif
#endif
