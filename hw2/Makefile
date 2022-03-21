CC=g++
CFLAGS = -std=c99

all : va1 va2 omp2 omp3 omp4 omp5 omp6 mmult jacobi gs

va1: val_test01_solved.cpp
	g++ -o val_test01_solved val_test01_solved.cpp

va2: val_test02_solved.cpp
	g++ -o val_test01_solved val_test02_solved.cpp

omp2: omp_solved2.c
	gcc -fopenmp -o omp_solved2 omp_solved2.c

omp3: omp_solved3.c
	gcc -fopenmp -o omp_solved3 omp_solved3.c

omp4: omp_solved4.c
	gcc -fopenmp -o omp_solved4 omp_solved4.c

omp5: omp_solved5.c
	gcc -fopenmp -o omp_solved5 omp_solved5.c

omp6: omp_solved6.c
	gcc -fopenmp -o omp_solved6 omp_solved6.c

mmult: MMult1.cpp
	g++ -fopenmp -O3 -march=native -o MMult1 MMult1.cpp 

jacobi: jacobi2D-omp.cpp
	g++ -fopenmp  -O3 -o jacobi2D-omp jacobi2D-omp.cpp

gs: gs2D-omp.cpp
	g++ -fopenmp  -O3 -o gs2D-omp gs2D-omp.cpp


clean : 
	rm -rf val_test01_solved val_test01_solved omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 MMult1 jacobi2D-omp gs2D-omp
