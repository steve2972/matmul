TARGET=main
OBJECTS=timer.o util.o mat_mul.o

CFLAGS=-std=c11 -O2 -fopenmp
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS) mat_mul_ref.o

clean:
	rm -rf $(TARGET) $(OBJECTS)

validation: $(TARGET)
	./$(TARGET) -v -t 48 1008 48 16
	./$(TARGET) -v -t 49 80 176 336
	./$(TARGET) -v -t 28 656 320 464
	./$(TARGET) -v -t 48 224 720 752
	./$(TARGET) -v -t 2 576 464 672
	./$(TARGET) -v -t 64 704 496 736
	./$(TARGET) -v -t 28 48 960 720
	./$(TARGET) -v -t 48 1024 496 672
	./$(TARGET) -v -t 46 800 880 416
	./$(TARGET) -v -t 56 816 304 352

performance: $(TARGET)
	./$(TARGET) -v -t 16 -w 3 -n 3 4096 4096 4096

threadtest: $(TARGET)
	./$(TARGET) -t 1 -n 10 512 512 512
	./$(TARGET) -t 2 -n 10 512 512 512
	./$(TARGET) -t 3 -n 10 512 512 512
	./$(TARGET) -t 4 -n 10 512 512 512
	./$(TARGET) -t 5 -n 10 512 512 512
	./$(TARGET) -t 6 -n 10 512 512 512
	./$(TARGET) -t 7 -n 10 512 512 512
	./$(TARGET) -t 8 -n 10 512 512 512
	./$(TARGET) -t 9 -n 10 512 512 512
	./$(TARGET) -t 10 -n 10 512 512 512
	./$(TARGET) -t 11 -n 10 512 512 512
	./$(TARGET) -t 12 -n 10 512 512 512
	./$(TARGET) -t 13 -n 10 512 512 512
	./$(TARGET) -t 14 -n 10 512 512 512
	./$(TARGET) -t 15 -n 10 512 512 512
	./$(TARGET) -t 16 -n 10 512 512 512
	./$(TARGET) -t 17 -n 10 512 512 512
	./$(TARGET) -t 18 -n 10 512 512 512
	./$(TARGET) -t 19 -n 10 512 512 512
	./$(TARGET) -t 20 -n 10 512 512 512
	./$(TARGET) -t 21 -n 10 512 512 512
	./$(TARGET) -t 22 -n 10 512 512 512
	./$(TARGET) -t 23 -n 10 512 512 512
	./$(TARGET) -t 24 -n 10 512 512 512
	./$(TARGET) -t 25 -n 10 512 512 512
	./$(TARGET) -t 26 -n 10 512 512 512
	./$(TARGET) -t 27 -n 10 512 512 512
	./$(TARGET) -t 28 -n 10 512 512 512
	./$(TARGET) -t 29 -n 10 512 512 512
	./$(TARGET) -t 30 -n 10 512 512 512
	./$(TARGET) -t 31 -n 10 512 512 512
	./$(TARGET) -t 32 -n 10 512 512 512
	./$(TARGET) -t 33 -n 10 512 512 512
	./$(TARGET) -t 34 -n 10 512 512 512
	./$(TARGET) -t 35 -n 10 512 512 512
	./$(TARGET) -t 36 -n 10 512 512 512
	./$(TARGET) -t 37 -n 10 512 512 512
	./$(TARGET) -t 38 -n 10 512 512 512
	./$(TARGET) -t 39 -n 10 512 512 512
	./$(TARGET) -t 40 -n 10 512 512 512
	./$(TARGET) -t 41 -n 10 512 512 512
	./$(TARGET) -t 42 -n 10 512 512 512
	./$(TARGET) -t 43 -n 10 512 512 512
	./$(TARGET) -t 44 -n 10 512 512 512
	./$(TARGET) -t 45 -n 10 512 512 512
	./$(TARGET) -t 46 -n 10 512 512 512
	./$(TARGET) -t 47 -n 10 512 512 512
	./$(TARGET) -t 48 -n 10 512 512 512
	./$(TARGET) -t 49 -n 10 512 512 512
	./$(TARGET) -t 50 -n 10 512 512 512
	./$(TARGET) -t 51 -n 10 512 512 512
	./$(TARGET) -t 52 -n 10 512 512 512
	./$(TARGET) -t 53 -n 10 512 512 512
	./$(TARGET) -t 54 -n 10 512 512 512
	./$(TARGET) -t 55 -n 10 512 512 512
	./$(TARGET) -t 56 -n 10 512 512 512
	./$(TARGET) -t 57 -n 10 512 512 512
	./$(TARGET) -t 58 -n 10 512 512 512
	./$(TARGET) -t 59 -n 10 512 512 512
	./$(TARGET) -t 60 -n 10 512 512 512
	./$(TARGET) -t 61 -n 10 512 512 512
	./$(TARGET) -t 62 -n 10 512 512 512
	./$(TARGET) -t 63 -n 10 512 512 512
	./$(TARGET) -t 64 -n 10 512 512 512