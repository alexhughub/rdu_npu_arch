CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

TARGET = run_sweeps
SRCS = run_co_design_sweeps.cpp RDU_vs_NPU_CoDesignSimulator.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp RDU_vs_NPU_CoDesignSimulator.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
