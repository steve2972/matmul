TARGET=main
OBJECTS=timer.o util.o mat_mul.o

CC=gcc
CFLAGS=-std=c11 -O3 -Wall -fopenmp
LDLIBS=-lm

# MPI settings
#CC=mpicc
#CFLAGS+=-DUSE_MPI
#MPIRUN_FLAGS=-np 2 -H gpu05,gpu06 --bind-to none --mca btl openib,self --mca btl_openib_allow_ib true

# OpenCL settings
OBJECTS+=opencl_util.o
CFLAGS+=-I/usr/local/cuda/include
LDLIBS+=-lOpenCL

all: $(TARGET)

$(TARGET): $(OBJECTS) mat_mul_ref.o

clean:
	rm -rf $(TARGET) $(OBJECTS)

validation: $(TARGET)
	./$(TARGET) -v 896 1536 640
	./$(TARGET) -v 1408 512 1792
	./$(TARGET) -v 640 512 2048
	./$(TARGET) -v 512 384 1664
	./$(TARGET) -v 1152 768 896
	./$(TARGET) -v 1024 384 1664
	./$(TARGET) -v 1792 640 2048
	./$(TARGET) -v 384 896 640
	./$(TARGET) -v 256 1024 1920
	./$(TARGET) -v 1792 768 1152

performance: $(TARGET)
	./$(TARGET) -v -w 3 -n 3 8192 8192 8192
