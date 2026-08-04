CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

TARGET = run_sweeps
SRCS = run_workload_sweeps.cpp RDUMicroSim.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp RDUMicroSim.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
