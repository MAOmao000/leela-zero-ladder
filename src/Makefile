THE_OS := $(shell uname -s)

default:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=gcc CXX=g++ \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -Wno-deprecated-copy -pipe -O3 -g -ffast-math -flto -march=native -std=c++17 -DNDEBUG' \
		LDFLAGS='$(LDFLAGS) -flto -g' \
		leelaz

debug:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=gcc CXX=g++ \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -Wno-deprecated-copy -pipe -Og -g -std=c++17' \
		LDFLAGS='$(LDFLAGS) -g' \
		leelaz

clang:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=clang CXX=clang++ \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -Wno-deprecated-copy -O3 -ffast-math -flto -mfpmath=sse -march=native -std=c++17 -DNDEBUG' \
		LDFLAGS='$(LDFLAGS) -flto -fuse-linker-plugin' \
		leelaz

DYNAMIC_LIBS = -lboost_system -lboost_filesystem -lboost_program_options -lpthread -lz
LIBS =
USE_CUDNN = 0

ifeq ($(THE_OS),Linux)
# for Linux with OpenBLAS
	CXXFLAGS += -I/usr/include/openblas -I./Eigen
	DYNAMIC_LIBS += -lopenblas
	DYNAMIC_LIBS += -lOpenCL
	LDFLAGS += -L/usr/local/cuda-12.2/lib64
ifeq ($(BACKEND),CUDNN)
	CUDA_INCLUDE = /usr/local/cuda-12.2/targets/x86_64-linux/include
	CUDNN_INCLUDE = /usr/include
	CUDNN_LIB = /usr/local/cuda-12.2/lib64
	CXXFLAGS += -I$(CUDNN_INCLUDE) -I$(CUDA_INCLUDE) -DUSE_BLAS -DUSE_CUDNN
	NVCCFLAGS = -std=c++17 -O3 -Xcompiler -Wall -DUSE_CUDNN
	NVCCFLAGS += -I$(CUDNN_INCLUDE) -I$(CUDA_INCLUDE)
	NVCCFLAGS += -gencode arch=compute_${CUDA_ARCHITECTURES},code=sm_${CUDA_ARCHITECTURES}
	NVCCFLAGS += -gencode arch=compute_${CUDA_ARCHITECTURES},code=compute_${CUDA_ARCHITECTURES}
	DYNAMIC_LIBS += -lcudart_static -lcuda -lcudnn -lcublas
	LDFLAGS  += -L$(CUDNN_LIB)
endif
ifeq ($(BACKEND),CUDNN_GRAPH)
	CUDA_INCLUDE = /usr/local/cuda-12.2/targets/x86_64-linux/include
	CUDNN_INCLUDE = /usr/include
	CUDNN_FE_INCLUDE = ${HOME}/cudnn-frontend-1.5.2/include
	CUDNN_LIB = /usr/local/cuda-12.2/lib64
	CXXFLAGS += -I$(CUDNN_INCLUDE) -I$(CUDA_INCLUDE) -I$(CUDA_FE_INCLUDE) -DUSE_BLAS -DUSE_CUDNN -DUSE_CUDNN_GRAPH
	NVCCFLAGS = -std=c++17 -O3 -Xcompiler -Wall -DUSE_CUDNN
	NVCCFLAGS += -I$(CUDNN_INCLUDE) -I$(CUDA_INCLUDE)
	NVCCFLAGS += -gencode arch=compute_${CUDA_ARCHITECTURES},code=sm_${CUDA_ARCHITECTURES}
	NVCCFLAGS += -gencode arch=compute_${CUDA_ARCHITECTURES},code=compute_${CUDA_ARCHITECTURES}
	DYNAMIC_LIBS += -lcudart_static -lcuda -lcudnn -lcublas
	LDFLAGS  += -L$(CUDNN_LIB)
endif
endif
ifeq ($(THE_OS),Darwin)
# for macOS (comment out the Linux part)
	LIBS += -framework Accelerate
	LIBS += -framework OpenCL
	CXXFLAGS += -I./Eigen
	CXXFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers
endif

# for MKL instead of OpenBLAS
#DYNAMIC_LIBS += -lmkl_rt
#CXXFLAGS += -I/opt/intel/mkl/include
#LDFLAGS  += -L/opt/intel/mkl/lib/intel64/

CXXFLAGS += -I.
CPPFLAGS += -MD -MP

#sources = $(wildcard *.cpp)
#objects = $(sources:.cpp=.o)
#deps = $(sources:%.cpp=%.d)

BUILD_DIR ?= .
SRC_DIRS ?= .
sources:= $(shell find $(SRC_DIRS) -maxdepth 1 -name '*.cpp' -or -name '*.cu')
objects := $(sources:%=$(BUILD_DIR)/%.o)
deps := $(objects:.o=.d)

-include $(deps)

$(BUILD_DIR)/leelaz: $(objects)
	$(CXX) $(objects) $(LDFLAGS) -o $@ $(LIBS) $(DYNAMIC_LIBS)

$(BUILD_DIR)/%.cu.o: %.cu
	nvcc $(NVCCFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

clean:
	-$(RM) leelaz $(objects) $(deps)

.PHONY: clean default debug clang
