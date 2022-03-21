#include <stdio.h>
#include "utils.h"
#include <cmath>
//function declarations
double** create_A(int N);
double* jacobi(int N, double* f, int iter);
double* gauss_seidel(int N, double* f, int iter);
double norm(double** A, double* u, double* f, int N);
void print_u(double* u, int N);


double** create_A(int N)
{
	double** A = (double**) malloc((N + 2) * (N + 2) * sizeof(double));
	for(int i = 0; i < N + 2; i++)
	{
		A[i] = (double*)malloc((N + 2) * sizeof(double));
		for(int j = 0; j < N + 2; j++)
			A[i][j] = 0;
	}

	double h_2 = (N+1) * (N+1);
	for(int i = 0; i < N + 2; i++)
		for(int j = 0; j < N + 2; j++)
		{	if(i == 0 || i == N + 1)
				A[i][i] = 1 *  h_2;
			else if(i == 1)
			{
				A[i][i] = 2 * h_2;
				A[i][i + 1] = (-1) * h_2; 
 			}
 			else if(i == N)
 			{
 				A[i][i-1] = (-1) * h_2;
 				A[i][i] = 2 * h_2;
 			}
 			else
 			{
 				A[i][i-1] = (-1) * h_2;
 				A[i][i] = 2 * h_2;
 				A[i][i+1] = (-1) * h_2;
 			}
		}

	return A;
}

double* jacobi(int N, double* f, int iter)
{
	double norm_val;
	double** A = create_A(N);
	double* u = (double*)malloc((N + 2) * sizeof(double));
	double* u_prev = (double*)malloc((N + 2) * sizeof(double));
	double init = 0;
	for(int i = 0; i < N + 2; i++)
	{
		u[i] = 0;
		u_prev[i] = 0;
	}
	double sum;
	for(int k = 0; k < iter; k++)
	{
		for(int i = 1; i <= N; i++)
		{
			sum = 0;
			for(int j = 1; j <= N; j++)
				sum = sum + (A[i][j] * u_prev[j]);
			sum = sum - A[i][i] * u_prev[i];
			u[i] = (f[i] - sum)/ A[i][i];
			// printf("i = %d:  %f\n",i ,A[i][i]);
		}
		for(int i = 0 ; i < N + 2; i++)
			u_prev[i] = u[i];
		norm_val = norm(A, u, f, N);
		printf("iteration %d: norm = %f\n", k, norm_val);
		if(k == 0)
			init = norm_val; //stores the initial value of norm
		else if(init / norm_val >= 1e+6)
		{
			printf("%s\n", "initial residual is decreased by a factor of 1e+6");
			break;
		}
	}
	free(A);
	free(u_prev);
	printf("u vector using Jacobi method: ");
	print_u(u, N);
	return u;
}

double* gauss_seidel(int N, double* f, int iter)
{
	double norm_val;
	double** A = create_A(N);
	double* u = (double*)malloc((N + 2) * sizeof(double));
	double* u_prev = (double*)malloc((N + 2) * sizeof(double));
	double init = 0;
	for(int i = 0; i < N + 2; i++)
	{
		u[i] = 0;
		u_prev[i] = 0;
	}
	for(int k = 0; k < iter; k++)
	{
		for(int i = 1; i <= N; i++)
		{
			u[i] = f[i];
			for(int j = 0; j < i; j++)
				u[i] = u[i] - A[i][j] * u[j];
			for(int j = i + 1; j <= N; j++)
				u[i] = u[i] - A[i][j] * u_prev[j];
			u[i] = u[i] / A[i][i];
		}
		for(int i = 0; i < N + 2; i++)
			u_prev[i] = u[i];
		norm_val = norm(A, u, f, N);
		printf("iteration %d: norm = %f\n", k, norm_val);
		if(k == 0)
			init = norm_val; //stores the initial value of norm
		else if(init / norm_val >= 1e+6)
		{
			printf("%s\n", "initial residual is decreased by a factor of 1e+6");
			break;
		}
	}
	free(A);
	free(u_prev);
	printf("u vector using Gauss-Seidel method: ");
	print_u(u, N);
	return u;

}

double norm(double** A, double* u, double* f, int N)
{
	double total = 0;
	double sum;
	double* diff = (double*) malloc((N + 2) * sizeof(double));
	for(int i = 0; i < N + 2; i++)
	{	sum = 0;
		for(int j = 0; j < N + 2; j++)
		{
			sum += (A[i][j]*u[j]);
		}
		diff[i] = sum - f[i];
	}

	for(int i = 0; i < N + 2; i++)
		total += (diff[i] * diff[i]);
	total = sqrt(total);
	free(diff);
	return total;
}

void print_u(double* u, int N)
{
	for(int i = 0; i < N + 2; i++)
	{
		printf("%f ", u[i]);
	}
	printf("\n");
}

int main()
{
	Timer t; //timer for calculating runtimes
	int N = 10000;
	double* f = (double*) malloc((N + 2) * sizeof(double));
	f[0] =0;
	f[N + 1] = 0;
	for(int i = 1; i <= N; i++)
		f[i] = 1; // setting the function values f[0] and f[N + 1] = 0 by defintion of u
	
	t.tic();
	double* u_jacobi = jacobi(N, f, 10000);
	printf("Time to run Jacobi method %10f\n",t.toc());
	free(u_jacobi);
	t.tic();
	double* u_gauss_seidel = gauss_seidel(N, f, 100);
	printf("Time to run Gauss-Seide method %10f\n",t.toc());
	free(u_gauss_seidel);
	free(f);
	return 0;
}