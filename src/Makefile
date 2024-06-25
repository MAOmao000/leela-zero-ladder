THE_OS := $(shell uname -s)

default:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=gcc CXX=g++ NVCC=nvcc \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -Wno-deprecated-copy -pipe -O3 -g -ffast-math -flto -march=native -std=c++17 -DNDEBUG' \
		NVCCFLAGS='$(NVCCFLAGS) -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-function"
		LDFLAGS='$(LDFLAGS) -flto -g' \
		leelaz

debug:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=gcc CXX=g++ NVCC=nvcc \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -Wno-deprecated-copy -pipe -Og -g -std=c++17' \
		NVCCFLAGS='$(NVCCFLAGS) -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-function"
		LDFLAGS='$(LDFLAGS) -g' \
		leelaz

clang:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=clang CXX=clang++ NVCC=nvcc \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -Wno-deprecated-copy -O3 -ffast-math -flto -mfpmath=sse -march=native -std=c++17 -DNDEBUG' \
		NVCCFLAGS='$(NVCCFLAGS) -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-function"
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
	CXXFLAGS += -I$(CUDNN_INCLUDE) -I$(CUDA_INCLUDE) -DUSE_CUDNN
	NVCCFLAGS += -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89
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

sources = $(wildcard *.cpp) $(wildcard *.cu)
#sources = Network.cpp FullBoard.cpp KoState.cpp Training.cpp \
#	  TimeControl.cpp UCTSearch.cpp GameState.cpp Leela.cpp \
#	  SGFParser.cpp Timing.cpp Utils.cpp FastBoard.cpp \
#	  SGFTree.cpp Zobrist.cpp FastState.cpp GTP.cpp Random.cpp \
#	  SMP.cpp UCTNode.cpp UCTNodePointer.cpp UCTNodeRoot.cpp \
#	  OpenCL.cpp OpenCLScheduler.cpp NNCache.cpp Tuner.cpp CPUPipe.cpp \
#	  LadderDetection.cpp GoBoard.cpp Ladder.cpp Pattern.cpp SearchBoard.cpp ZobristHash.cpp

objects = $(sources:.cpp=.o) $(sources:.cu=.o)
deps = $(sources:%.cpp=%.d) $(sources:%.cu=%.d)

-include $(deps)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

leelaz: $(objects)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS) $(DYNAMIC_LIBS)

clean:
	-$(RM) leelaz $(objects) $(deps)

.PHONY: clean default debug clang
