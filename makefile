test: main.cpp
	g++ main.cpp -o main -fopenmp
	OMP_NUM_THREADS=1 ./main 20 1
	OMP_NUM_THREADS=1 ./main 20 13
	OMP_NUM_THREADS=1 ./main 20 20
	OMP_NUM_THREADS=2 ./main 20 1
	OMP_NUM_THREADS=2 ./main 20 13
	OMP_NUM_THREADS=2 ./main 20 20
	OMP_NUM_THREADS=4 ./main 20 1
	OMP_NUM_THREADS=4 ./main 20 13
	OMP_NUM_THREADS=4 ./main 20 20
	OMP_NUM_THREADS=8 ./main 20 1
	OMP_NUM_THREADS=8 ./main 20 13
	OMP_NUM_THREADS=8 ./main 20 20
	OMP_NUM_THREADS=1 ./main 24 1
	OMP_NUM_THREADS=1 ./main 24 13
	OMP_NUM_THREADS=1 ./main 24 24
	OMP_NUM_THREADS=2 ./main 24 1
	OMP_NUM_THREADS=2 ./main 24 13
	OMP_NUM_THREADS=2 ./main 24 24
	OMP_NUM_THREADS=4 ./main 24 1
	OMP_NUM_THREADS=4 ./main 24 13
	OMP_NUM_THREADS=4 ./main 24 24
	OMP_NUM_THREADS=8 ./main 24 1
	OMP_NUM_THREADS=8 ./main 24 13
	OMP_NUM_THREADS=8 ./main 24 24
	OMP_NUM_THREADS=1 ./main 28 1
	OMP_NUM_THREADS=1 ./main 28 13
	OMP_NUM_THREADS=1 ./main 28 28
	OMP_NUM_THREADS=2 ./main 28 1
	OMP_NUM_THREADS=2 ./main 28 13
	OMP_NUM_THREADS=2 ./main 28 28
	OMP_NUM_THREADS=4 ./main 28 1
	OMP_NUM_THREADS=4 ./main 28 13
	OMP_NUM_THREADS=4 ./main 28 28
	OMP_NUM_THREADS=8 ./main 28 1
	OMP_NUM_THREADS=8 ./main 28 13
	OMP_NUM_THREADS=8 ./main 28 28
	OMP_NUM_THREADS=1 ./main 30 1
	OMP_NUM_THREADS=1 ./main 30 13
	OMP_NUM_THREADS=1 ./main 30 30
	OMP_NUM_THREADS=2 ./main 30 1
	OMP_NUM_THREADS=2 ./main 30 13
	OMP_NUM_THREADS=2 ./main 30 30
	OMP_NUM_THREADS=4 ./main 30 1
	OMP_NUM_THREADS=4 ./main 30 13
	OMP_NUM_THREADS=4 ./main 30 30
	OMP_NUM_THREADS=8 ./main 30 1
	OMP_NUM_THREADS=8 ./main 30 13
	OMP_NUM_THREADS=8 ./main 30 30
