# Compiler
NVCC = nvcc

# Program name
PROGRAM_NAME = simulation

# Compiler flags
CXXFLAGS = -O3 -std=c++17

# Target executable name
TARGET = $(PROGRAM_NAME)

# Source files
SRC = test_cuda.cu

# Header files
HEADERS = cudoinfo.h, satellite.h, groundpoint.h, check_covered.h

# Object files
OBJ = $(SRC:.cu=.o)

# CUDA libraries
LIBS =

# Rules
all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
