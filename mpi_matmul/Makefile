TARGET=main
OBJECTS=timer.o util.o mat_mul.o

CC=mpicc
CFLAGS=-std=c11 -O3 -Wall -fopenmp
LDFLAGS=-lm
MPIRUN_FLAGS=-np 2 -H gpu03,gpu02 --bind-to none --mca btl openib,self --mca btl_openib_allow_ib true

all: $(TARGET)

$(TARGET): $(OBJECTS) mat_mul_ref.o

clean:
	rm -rf $(TARGET) $(OBJECTS)

validation: $(TARGET)
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 896 1536 640
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 1408 512 1792
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 640 512 2048
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 512 384 1664
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 1152 768 896
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 1024 384 1664
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 1792 640 2048
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 384 896 640
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 256 1024 1920
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v 1792 768 1152

performance: $(TARGET)
	mpirun $(MPIRUN_FLAGS) ./$(TARGET) -v -w 3 -n 3 8192 8192 8192
