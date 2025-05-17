# Makefile – only the bits we need for conv2d modelling
NVCC      = nvcc
CUDAFLAGS = --std=c++11 -O3 -arch=sm_70    # adjust arch as needed

TARGET    = conv2d
SRC       = conv2d.cu
HDRS      = dnn.hpp params.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(HDRS)
	$(NVCC) $(CUDAFLAGS) -o $@ $(SRC)

clean:
	$(RM) $(TARGET)
