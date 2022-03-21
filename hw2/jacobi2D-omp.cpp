#include <stdio.h>
#include "utils.h"
#include <cmath>
#include <omp.h>	

//function declarations
double** create_f(int N);
double** create_u(int N);
double** jacobi(int N, double**f, int iter, int nrT);
void print_u(double** u, int N);

double** create_f(int N)
{
	double** f = (double**) malloc((N + 2) * (N + 2) * sizeof(double));
	for(int i = 0; i < N + 2; i++)
	{
		f[i] = (double*)malloc((N + 2) * sizeof(double));
		f[i][0] = 0;
		f[i][N+1] = 0;
		for(int j = 1; j <= N; j++)
			f[i][j] = 1;
	}
	//print_u(f, N);
	return f;
}

double** create_u(int N)
{
	double** U = (double**) malloc((N + 2) * (N + 2) * sizeof(double));
	for(int i = 0; i < N + 2; i++)
	{
		U[i] = (double*)malloc((N + 2) * sizeof(double));
		for(int j = 0; j < N + 2; j++)
			U[i][j] = 0;
	}
	return U;

}

double** jacobi(int N, double**f, int iter, int nrT)
{
	double h_2 = 1.0 / ((N+1) *(N+1));
	double** u = create_u(N);
	double** u_prev = create_u(N);
	for(int k = 0; k < iter; k++)
	{
		//set u
		// parallel region where both u and u_prev matrices are shared
		#pragma omp parallel shared(u, u_prev) num_threads(nrT)
		{ 
			#pragma omp for collapse(2)
			for(int i = 1; i <= N; i++)
				for(int j = 1; j <= N; j++)
					u[i][j] = 0.25 * (h_2 * f[i][j] + u_prev[i-1][j] + u_prev[i][j-1] + u_prev[i+1][j] + u_prev[i][j+1]); //jacobi update

		//set u_prev
			#pragma omp for collapse(2)
			for(int i = 1; i <= N; i++)
				for(int j = 1; j <= N; j++)
					u_prev[i][j] = u[i][j];
		}

	}

	//print_u(u, N);
	return u;
}

void print_u(double** u, int N)
{
	for(int i = 0; i < N + 2; i++)
	{
		for(int j = 0; j < N + 2; j++)
		{
			printf("%f ", u[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

int main()
{
	//Timer t;
	int N = 1000;
	int iter = 100;
	int nrThreads[6] = {1, 2, 4, 8, 32, 64};
	double** f =  create_f(N);
	double singleC;
	double t = 1.0;
	printf("N: %d, Number of Iteratins: %d\n", N, iter);

	for(int nrTIndex = 0; nrTIndex < 6; nrTIndex++) 
	{
	    int nrT = nrThreads[nrTIndex];
	    #ifdef _OPENMP
	    	t = omp_get_wtime();
	    #endif
	    double** u_jacobi = jacobi(N, f, iter, nrT);
	    #ifdef _OPENMP
	    	t = omp_get_wtime() - t;
	    #endif   
            free(u_jacobi);
	    if(nrT == 1)
	      singleC = t;
	    printf("%d: time elapsed = %f, speedup = %f\n", nrT, t, singleC/t);
   	}

	free(f);
	return 0;
}
