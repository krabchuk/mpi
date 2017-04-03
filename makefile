PROGRAM = a.out
CFLAGS = -g -pg -Wall -O3 --fast-math
COMPILER = mpicxx

$(PROGRAM): main.o read_print.o gauss.o mpi_functions.o
	$(COMPILER) $(CFLAGS) main.o read_print.o gauss.o mpi_functions.o -o $(PROGRAM)

main.o: main.cpp head.h mpi_functions.h
	$(COMPILER) $(CFLAGS) -c main.cpp
read_print.o: read_print.cpp head.h
	$(COMPILER) $(CFLAGS) -c read_print.cpp
gauss.o: gauss.cpp head.h mpi_functions.h
	$(COMPILER) $(CFLAGS) -c gauss.cpp
mpi_functions.o: mpi_functions.cpp mpi_functions.h head.h
	$(COMPILER) $(CFLAGS) -c mpi_functions.cpp

clean:
	rm -f $(PROGRAM) leak.out* gmon* *.o

