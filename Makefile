
USE_BOOST ?= 1
DEBUG ?= 0
USE_OPTION_PARSER = 0

BOOST_HOME ?= $(shell dirname $(shell echo $$LD_LIBRARY_PATH | tr ':' '\n' | grep boost/lib | head -n 1))
MPI_HOME ?= $(shell dirname $(dirname $(which mpicc)))

MPICXX = $(MPI_HOME)/bin/mpic++
MPIRUN = $(MPI_HOME)/bin/mpirun

ifeq ($(USE_BOOST),1)
CXX = $(MPICXX)
else
CXX = g++
endif

CXXFLAGS = -Wall -finline-functions -funswitch-loops

# Detect Support for C++11 (C++0x) from GCC Version 
GNUC_CPP0X := $(shell mpic++ --version | perl -ne 'if (/g++\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')

ifeq ($(GNUC_CPP0X), 1)
	CXXFLAGS += -std=c++11
endif

CXXFLAGS += -I./hw-parser -I./hw-component
CXXFLAGS += -I./ISA-Def -I./DEV-Def -I./trace-parser -I./trace-driven -I./common
CXXFLAGS += -I./common/CLI -I./common/CLI/impl -I$(MPI_HOME)/include
CXXFLAGS += -I$(BOOST_HOME)/include
CXXFLAGS += -I./parda
CXXFLAGS += $(shell pkg-config --cflags glib-2.0)

LIBRARIES = -L$(BOOST_HOME)/lib -lboost_mpi -lboost_serialization
LIBRARIES += $(shell pkg-config --libs glib-2.0)

ifeq ($(DEBUG),1)
OPTFLAGS = -O0 -g3 -fPIC
else
OPTFLAGS = -O3 -fPIC
endif

OBJ_PATH = obj


TARGET = gpu-simulator.x


exist_OBJ_PATH = $(shell if [ -d $(OBJ_PATH) ]; then echo "exist"; else echo "noexist"; fi)

ifeq ("$(exist_OBJ_PATH)", "noexist")
$(shell mkdir $(OBJ_PATH))
endif

OBJS = $(OBJ_PATH)/splay.o $(OBJ_PATH)/process_args.o $(OBJ_PATH)/parda_print.o $(OBJ_PATH)/narray.o $(OBJ_PATH)/parda.o
OBJS += $(OBJ_PATH)/hw-parser.o $(OBJ_PATH)/Scoreboard.o $(OBJ_PATH)/RegisterBankAllocator.o
OBJS += $(OBJ_PATH)/IBuffer.o $(OBJ_PATH)/PipelineUnit.o
OBJS += $(OBJ_PATH)/PrivateSM.o $(OBJ_PATH)/OperandCollector.o
OBJS += $(OBJ_PATH)/hw-stt.o $(OBJ_PATH)/inst-stt.o 
OBJS += $(OBJ_PATH)/memory-space.o $(OBJ_PATH)/inst-memadd-info.o $(OBJ_PATH)/sass-inst.o $(OBJ_PATH)/inst-trace.o
OBJS += $(OBJ_PATH)/kernel-trace.o $(OBJ_PATH)/mem-access.o $(OBJ_PATH)/kernel-info.o $(OBJ_PATH)/trace-warp-inst.o
OBJS += $(OBJ_PATH)/common_def.o $(OBJ_PATH)/trace-parser.o $(OBJ_PATH)/main.o

ifeq ($(USE_OPTION_PARSER),1)
OBJS += $(OBJ_PATH)/option_parser.o
endif

default: $(TARGET)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(MPICXX) $(CXXFLAGS) $(OPTFLAGS) $(LIBRARIES) -o $@ $^

$(OBJ_PATH)/main.o: main.cc
	$(MPICXX) $(CXXFLAGS) $(OPTFLAGS) $(LIBRARIES) -o $@ -c $^

$(OBJ_PATH)/trace-parser.o: trace-parser/trace-parser.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/trace-warp-inst.o: trace-driven/trace-warp-inst.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/kernel-info.o: trace-driven/kernel-info.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/mem-access.o: trace-driven/mem-access.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/kernel-trace.o: trace-driven/kernel-trace.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/PipelineUnit.o: hw-component/PipelineUnit.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/Scoreboard.o: hw-component/Scoreboard.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/RegisterBankAllocator.o: hw-component/RegisterBankAllocator.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/IBuffer.o: hw-component/IBuffer.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/OperandCollector.o: hw-component/OperandCollector.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/PrivateSM.o: hw-component/PrivateSM.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/hw-stt.o: trace-driven/hw-stt.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/inst-stt.o: trace-driven/inst-stt.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/common_def.o: common/common_def.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/option_parser.o: common/option_parser.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/memory-space.o: trace-parser/memory-space.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/sass-inst.o: trace-parser/sass-inst.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/inst-memadd-info.o: trace-parser/inst-memadd-info.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/inst-trace.o: trace-parser/inst-trace.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/hw-parser.o: hw-parser/hw-parser.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/splay.o: parda/splay.c
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/process_args.o: parda/process_args.c
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/parda_print.o: parda/parda_print.c
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/parda.o: parda/parda.c
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

$(OBJ_PATH)/narray.o: parda/narray.c
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ -c $^

.PHTONY: clean

clean:
	rm -f $(OBJS)
	rm -f $(TARGET)
	rm -rf $(OBJ_PATH)
