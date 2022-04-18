CXX = nvcc
CXXFLAGS = -arch=sm_61 -Xcompiler -fopenmp

# turning off auto-vectorization since this can make hand-vectorized code slower
TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cu
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
