#include <stdio.h>
#include <cmath>
#include <omp.h>

//function declarations
double** create_f(int N);
double** create_u(int N);
double** jacobi(int N, double**f, int iter);
void print_u(double** u, int N);
void print_uGPU(double* u, long N);
void print_error(double* u, double** u_ref, long N);
void residual(double** u, double** u_old);
double norm(double* u, double** u_ref, long N);
void test_jacobi();


__global__ void jacobiGPU(double* u, double* u_prev, double* u_error, double* f, long N, double h_2, long iter)
{
	long row = blockIdx.y * blockDim.y + threadIdx.y; //calculate the row index
	long col = blockIdx.x * blockDim.x + threadIdx.x;//calclate the col index
	__syncthreads();
	if(1 <= row && row <= N && 1 <= col && col <= N)
	{
		u[(row * (N+2)) + col] = 0.25 * (h_2 * f[(row * (N+2)) + col] + u_prev[((row - 1) * (N+2)) + col] + u_prev[(row * (N + 2)) + (col-1)] + u_prev[((row+1) * (N + 2)) + (col)] + u_prev[(row * (N + 2)) + (col +1)]);
		__syncthreads();
		//u_error[(row * (N + 2)) + col] = fabs(u_prev[(row * (N + 2)) + col] - u[(row * (N + 2)) + col]);
		u_prev[(row * (N + 2)) + col] = u[(row * (N + 2)) + col];
		__syncthreads();
	}

}


// __global__ void jacobi_GPU(double* u, double* u_prev, long N)
// {
// 	long row = blockIdx.y * blockDim.y + threadIdx.y; //calculate the row index
// 	long col = blockIdx.x * blockDim.x + threadIdx.x;//calclate the col index
// 	__syncthreads();
// 	if(1 <= row && row <= N && 1 <= col && col <= N)
// 	{
// 		u_prev[(row * (N + 2)) + col] = u[(row * (N + 2)) + col];
// 		__syncthreads();
// 	}

// }


int main()
{
	test_jacobi();
	return 0;
}

void test_jacobi()
{
	
	
	for(long N = 100; N < 1400; N += 100)
	{
	long iter = 1000;
	double h_2 = 1.0 / ((N+1) *(N+1));

	//Allocate Host and Device memory
	double *u, *f_, *u_error;
	cudaMallocHost(&u, (N+2) * (N+2) * sizeof(double));
	cudaMallocHost(&f_, (N+2) * (N+2) * sizeof(double));
	cudaMallocHost(&u_error, (N+2) * (N+2) * sizeof(double));
	double *u_d, *u_prev_d, *f_d, *u_error_d;
	cudaMalloc((void**)&u_d, (N+2) * (N+2) * sizeof(double));
	cudaMalloc((void**)&u_prev_d, (N+2) * (N+2) * sizeof(double));
	cudaMalloc((void**)&f_d, (N+2) * (N+2) * sizeof(double));
	cudaMalloc((void**)&u_error_d, (N+2) * (N+2) * sizeof(double));


	//CPU function call
	double** f = create_f(N);
	//initialize f_d
	for(int i = 0; i < N + 2; i++)
	{
		f_[i * (N+2) + 0] = 0;
		f_[(i * (N+2)) + (N+1)] = 0;
		for(int j = 1; j <= N; j++)
			f_[(i * (N+2)) + j] = 1;
	}

	//print_uGPU(f_, N);
	cudaMemcpyAsync(f_d, f_, (N+2) * (N+2) * sizeof(double),cudaMemcpyHostToDevice);

	//CPU jacobi
    double tt = omp_get_wtime();
	double** u_ref = jacobi(N, f, iter);
	double cputime = omp_get_wtime() - tt;
	long thread = 32;
	long grid_r = (N + thread - 1)/thread; 
	long grid_c = (N + thread - 1)/thread;
	//printf("gridsize: %d\n", grid_r);
	//GPU function call
	dim3 threads(thread, thread);
	dim3 grid(grid_r, grid_c);
	double gputime = 0.0;
	
	for(long i = 0; i < iter; i++)
	{
		double maxError = -1;
		tt = omp_get_wtime();
		jacobiGPU<<<grid, threads>>>(u_d, u_prev_d, u_error_d, f_d, N, h_2, iter);
		cudaMemcpy(u_error, u_error_d, (N+2) * (N+2) *sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		gputime += omp_get_wtime() - tt;

		//find error
		for(int i = 0; i < (N + 2) * (N + 2); i++)
			if(maxError < u_error[i])
				maxError = u_error[i];
		// jacobi_GPU<<<grid, threads>>>(u_d, u_prev_d, N);
		// cudaDeviceSynchronize();
		//printf("residual: %e\n", maxError);
	}
	tt = omp_get_wtime();
	cudaMemcpyAsync(u, u_d, (N+2) * (N+2) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(f_, f_d, (N+2) * (N+2) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	gputime += omp_get_wtime() - tt;

	//print_error(u, u_ref, N);
	printf("N: %d, Iterations: %d, Error: %e, CPUTime: %f, GPUTime: %f, Speed Up: %f\n", N, iter, norm(u, u_ref, N), cputime, gputime, cputime/gputime);
	// print_u(u_ref, N);
	// print_uGPU(u, N);
	// print_u(f, N);
	cudaFree(u_d); cudaFree(u_prev_d); cudaFree(f_d); cudaFree(u_error_d);
	cudaFreeHost(u); cudaFreeHost(f_); cudaFreeHost(u_error);
	free(u_ref);
	free(f);
	}
}


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

double** jacobi(int N, double**f, int iter)
{
	double h_2 = 1.0 / ((N+1) *(N+1));
	double** u = create_u(N);
	double** u_prev = create_u(N);
	for(int k = 0; k < iter; k++)
	{

		for(int i = 1; i <= N; i++)
			for(int j = 1; j <= N; j++)
				u[i][j] = 0.25 * (h_2 * f[i][j] + u_prev[i-1][j] + u_prev[i][j-1] + u_prev[i+1][j] + u_prev[i][j+1]); //jacobi update


		for(int i = 1; i <= N; i++)
			for(int j = 1; j <= N; j++)
				u_prev[i][j] = u[i][j];
	}
	free(u_prev);
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

void print_uGPU(double* u, long N)
{
	for(long i = 0; i < N + 2; i++)
	{
		for(long j = 0; j < N + 2; j++)
		{
			printf("%f ", u[(i * (N + 2)) + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

double norm(double* u, double** u_ref, long N)
{
	double error = 0.0;
	for(long i = 1; i <= N; i++)
		for(long j = 1; j <= N; j++)
			error += fabs(u[(i * (N+2)) + j] - u_ref[i][j]);
	return error;
}	

void print_error(double* u, double** u_ref, long N)
{
	for(long i = 1; i <= N; i++)
		for(long j = 1; j <= N; j++)
			printf("u_ref: %f  u: %f  error: %e\n", u_ref[i][j], u[(i * (N+2)) + j], fabs(u[(i * (N+2)) + j] - u_ref[i][j]));
		
}
