LIBFILES = src/OptimalKernelLaunch.cu
LIB = src/OptimalKernelLaunch.o
TESTFILES = src/test.cu

all: $(LIB) test
test: $(LIB)
	nvcc -o test $(TESTFILES) $(LIB) -I src/
lib:
$(LIB):
	nvcc -c $(LIBFILES) -o $(LIB)
clean:
	rm -rf $(LIB)
	rm -rf test

