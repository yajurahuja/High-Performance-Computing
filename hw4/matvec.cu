#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define BLOCKSIZE 1024
#define BLOCKSIZE_ 32
#define NT 32

void test_dotProduct();
void dotProductCPU(double* A, double* B, double* sum_ref, long N); //CPU function sequential
double check_sumDP(double sum, double sum_ref);


//matrix-vector multiplication
void test_matVec_2D();
void test_matVec();
double check_sum(double* sum, double* sum_ref, long r, long c);
double check_sum_(double* sum, double* sum_ref, long r, long c); //Compare the CPU and GPU answers
void mat_VecCPU(double * A, double* B, double* sum, long r, long c); //CPU funtion sequential

//gpu function
__global__ void dotProductGPU(const double* A, const double* B, double* sum, long n) //this function calculates the dot product of two vectors
{
	__shared__ double smem[BLOCKSIZE]; //Shared memory for a block: Used for reduction
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //calculate the index
	if(idx < n) smem[threadIdx.x] = A[idx] * B[idx]; //add the product to the shared memory
	else smem[threadIdx.x] =  0.0;

	__syncthreads(); //wait for all the threads to finish

	//parallel reduction for to get partial sums
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) //traverse from 1/2 array to start
	{
		if(threadIdx.x < s) {
			smem[threadIdx.x] += smem[threadIdx.x + s]; 
		}
		__syncthreads();
	}
	// writing partial sums to global memory
	if(threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x]; 
}

__global__ void addPartialSumsGPU(double* sum, long n)
{
	__shared__ double smem[BLOCKSIZE];
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //calculate the index
	if(idx < n) smem[threadIdx.x] = sum[idx]; //add the product to the shared memory
	else smem[threadIdx.x] =  0.0;

	__syncthreads(); //wait for all the threads to finish

	//parallel reduction for to get partial sums
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) //traverse from 1/2 array to start
	{
		if(threadIdx.x < s) {
			smem[threadIdx.x] += smem[threadIdx.x + s]; 
		}	
		__syncthreads();
	}
	// writing partial sums to global memory
	if(threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x]; 
}


__global__ void mat_VecGPU_(const double* A, const double* B, double* sum, long r, long c)
{
	__shared__ double smem[BLOCKSIZE];
	long idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < r) smem[threadIdx.x] = 0.0;
	__syncthreads();

	if(idx < r) 
	{
	 	for(long i = 0; i < c; i++)
	 	{
	 		smem[threadIdx.x] += A[(idx * c) + i] * B[i];
	 	}
	}
	else smem[threadIdx.x] = 0.0;
	__syncthreads();

	if(idx < r) sum[idx] = smem[threadIdx.x];
}

__global__ void mat_VecGPU(const double* A, const double* B, double* sum, long r, long c) //this function calculates the dot product of two vectors
{
	__shared__ double smem[BLOCKSIZE_][BLOCKSIZE_]; //Shared memory for a block: Used for reduction
	long row = blockIdx.y * blockDim.y + threadIdx.y; //calculate the row index
	long col = blockIdx.x * blockDim.x + threadIdx.x;//calclate the col index

	if(row < r && col < c) 
	{
		smem[threadIdx.y][threadIdx.x] = A[(row * c) + col] * B[col]; //add the product to the shared memory
	}
	else smem[threadIdx.y][threadIdx.x] =  0.0;

	__syncthreads(); //wait for all the threads to finish

	//parallel reduction for to get partial sums
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) //traverse from 1/2 array to start
	{
		if(threadIdx.x < s) {
			smem[threadIdx.y][threadIdx.x] += smem[threadIdx.y][threadIdx.x + s]; 
		}
		__syncthreads();
	}
	// writing partial sums to global memory
	if(threadIdx.x == 0) sum[(row * c) + blockIdx.x] = smem[threadIdx.y][threadIdx.x]; 
}

__global__ void addPartialSumsGPU_(double* sum, long r, long c)
{
	__shared__ double smem[BLOCKSIZE_][BLOCKSIZE_]; //Shared memory for a block: Used for reduction
	long row = blockIdx.y * blockDim.y + threadIdx.y; //calculate the row index
	long col = blockIdx.x * blockDim.x + threadIdx.x;//calclate the col index
	if(row < r && col < c) smem[threadIdx.y][threadIdx.x] = sum[(row * c) + col]; //add the product to the shared memory
	else smem[threadIdx.y][threadIdx.x] =  0.0;

	__syncthreads(); //wait for all the threads to finish

	//parallel reduction for to get partial sums
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) //traverse from 1/2 array to start
	{
		if(threadIdx.x < s) {
			smem[threadIdx.y][threadIdx.x] += smem[threadIdx.y][threadIdx.x + s]; 
		}
		__syncthreads();
	}
	// writing partial sums to global memory
	if(threadIdx.x == 0) sum[(row * c)+ blockIdx.x] = smem[threadIdx.y][threadIdx.x]; 
}

int main()
{
	test_dotProduct(); //test dot product
	printf("\n\n");
	test_matVec();  //test matVec Mult
	test_matVec_2D(); //test matVec Mult with reduction
	return 0;
}

void test_dotProduct()
{
	long start_size = 10000;
	long end_size = 100000;
	long iter_size = 10000;
	printf("Testing Dot Product: CPU vs GPU\n");
	for(long size = start_size; size < end_size; size = size + iter_size)
	{
		//Allocate Vector memory
		double *A, *B, *sum, *sum_ref;
		// A = (double*) malloc(size * sizeof(double));
		// B = (double*) malloc(size * sizeof(double));
		cudaMallocHost(&A, size * sizeof(double)); 
		cudaMallocHost(&B, size * sizeof(double));
		cudaMallocHost(&sum, size* sizeof(double));
		cudaMallocHost(&sum_ref, sizeof(double));

		//Initialize Vectors;
		for(long i = 0; i < size; i++) 
			A[i] = drand48();


		for(long i = 0; i < size; i++) 
			B[i] = drand48();


		*sum = 0.0; 
		*sum_ref = 0.0;

		//Allocate memory for the GPU device
		double *A_d, *B_d, *sum_d;
		cudaMalloc((void**)&A_d, size * sizeof(double));
		cudaMalloc((void**)&B_d, size * sizeof(double));
		cudaMalloc((void**)&sum_d, size * sizeof(double));

		//copy from host to device
		cudaMemcpyAsync(A_d, A, size * sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpyAsync(B_d, B, size * sizeof(double),cudaMemcpyHostToDevice);

		//Calling the functions
		//GPU
		//printf("Running GPU kernel\n");
		double tt = omp_get_wtime();
		dotProductGPU<<<size/1024 + 1, 1024>>>(A_d, B_d, sum_d, size);
		addPartialSumsGPU<<<1, 1024>>>(sum_d, size);
		cudaDeviceSynchronize();
		double gputime = omp_get_wtime() - tt;
		cudaMemcpyAsync(sum, sum_d, size * sizeof(double), cudaMemcpyDeviceToHost);
		tt = omp_get_wtime();
		dotProductCPU(A, B, sum_ref, size); //dot product using 
		double cputime = omp_get_wtime() - tt;
		// for(long i = 0; i < size; i++)
		// 	printf("%f ", sum[i]);
		// printf("\n");
		printf("Size: %ld Error: %e CPU Time: %f GPU Time: %f Bandwidth: %f Gb/s Speedup: %f\n", size, check_sumDP(sum[0], *sum_ref), cputime, gputime, 3*size*sizeof(double) / gputime/1e9, cputime/gputime);
		cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(sum); cudaFreeHost(sum_ref);
		cudaFree(A_d); cudaFree(B_d); cudaFree(sum_d);
	}
	return;
}

void dotProductCPU(double* A, double* B, double* sum_ref, long N)
{
	for(int i = 0 ; i < N; i++)
		*sum_ref = *sum_ref + (A[i] * B[i]);
}

double check_sumDP(double sum, double sum_ref)
{
	return fabs(sum - sum_ref);
}

//Matrix Vector mulitplication functions
void mat_VecCPU(double * A, double* B, double* sum_ref, long r, long c)
{
	for(int i = 0; i < r; i++)
		for(int j = 0; j < c; j++)
			sum_ref[i] += A[i*c + j] * B[j];
}

double check_sum(double *sum, double *sum_ref, long r, long c)
{
	double error = 0.0;
	for(long i = 0; i < r; i++)
		error += fabs(sum[i * c] - sum_ref[i]);
	return error;
}

double check_sum_(double *sum, double *sum_ref, long r, long c)
{
	double error = 0.0;
	for(long i = 0; i < r; i++)
		error += fabs(sum[i] - sum_ref[i]);
	return error;
}

void test_matVec_2D()
{
	printf("Testing Matrix Vector Product: CPU vs GPU (Blocking)\n");
	long end_size = 6000;
	long iter_size = 1000;
	long size_r, size_c;
	for(long start_size = 1000; start_size < end_size; start_size += iter_size)
	{
		size_r = start_size;
		size_c = start_size;

		double *A, *B, *sum, *sum_ref;
		cudaMallocHost(&A, size_r * size_c * sizeof(double)); 
		cudaMallocHost(&B, size_c * sizeof(double));
		cudaMallocHost(&sum, size_r * size_c * sizeof(double));
		cudaMallocHost(&sum_ref, size_r * sizeof(double));

		//initialize the matrix
		for(long i = 0; i < size_r; i++)
			for(long j = 0; j < size_c; j++)
				A[i * size_c + j] = drand48();

		for(long i = 0; i < size_c; i++)
			B[i]= drand48();


		//Allocate memory for the GPU device
		double *A_d, *B_d, *sum_d;
		cudaMalloc((void**)&A_d, size_r * size_c * sizeof(double));
		cudaMalloc((void**)&B_d, size_c * sizeof(double));
		cudaMalloc((void**)&sum_d, size_r * size_c * sizeof(double));

		cudaMemcpyAsync(A_d, A, size_r * size_c * sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpyAsync(B_d, B, size_c * sizeof(double),cudaMemcpyHostToDevice);

		//Calling the CPU function
		double tt = omp_get_wtime();
		mat_VecCPU(A, B, sum_ref, size_r, size_c);
		double cputime = omp_get_wtime() - tt;
		//Calling the GPu function
		long nthreads = NT;
		long nthreads_r = nthreads;
		long nthreads_c = nthreads;
		long nblock_r = ceil(1.0 * size_r/nthreads_r);
		long nblock_c = ceil(1.0 * size_c/nthreads_c);
		dim3 blocks(nblock_c, nblock_r);
		//dim3 blocks_P(1, nblock_r);
		dim3 threads(nthreads_r, nthreads_c);
		//printf("Running GPU kernel\n");
		//printf("blocks: %ld x %ld, threads: %ld x %ld\n", blocks.x, blocks.y, threads.x, threads.y);
		
		tt = omp_get_wtime();
		mat_VecGPU<<<blocks, threads>>>(A_d, B_d, sum_d, size_r, size_c);
		cudaDeviceSynchronize();
		long t = ceil(logf(nblock_c) / logf(nthreads));
		//printf("t: %d \n", t);
		for(int i = 0; i < t; i++)
		{
			//printf("nblock_c: %d \n", nblock_c);
			dim3 blocks_P(nblock_c, nblock_r);
			addPartialSumsGPU_<<<blocks_P, threads>>>(sum_d, size_r, size_c);	
			cudaDeviceSynchronize();
			nblock_c = (nblock_c + nthreads_c - 1)/nthreads_c;
		}
		cudaDeviceSynchronize();
		double gputime = omp_get_wtime() - tt;
		cudaMemcpyAsync(sum, sum_d, size_r * size_c * sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		
		// for(long i = 0; i < size_r; i++)
		// {
		// 	printf("i = %ld  SUM_REF: %f, SUM: %f \n", i, sum_ref[i], sum[i * size_c + 0]);
		// }

		//Find error
		double error = check_sum(sum, sum_ref, size_r, size_c);
		printf("Size: %ld * %ld Error: %e CPU Time: %f GPU Time: %f Bandwidth: %f Gb/s Speedup: %f\n", size_r, size_c, error, cputime, gputime, (2*size_c*size_r + size_c)/gputime/1e9, cputime/gputime);

		cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(sum); cudaFreeHost(sum_ref);
		cudaFree(A_d); cudaFree(B_d); cudaFree(sum_d);


	}
}

void test_matVec()
{
	printf("Testing Matrix Vector Product: CPU vs GPU\n");
	long end_size = 20000;
	long iter_size = 2000;
	long size_r, size_c;
	for(long start_size = 1000; start_size < end_size; start_size += iter_size)
	{
		size_r = start_size;
		size_c = start_size;


		double *A, *B, *sum, *sum_ref;
		cudaMallocHost(&A, size_r * size_c * sizeof(double)); 
		cudaMallocHost(&B, size_c * sizeof(double));
		cudaMallocHost(&sum, size_r * sizeof(double));
		cudaMallocHost(&sum_ref, size_r * sizeof(double));


		//initialize the matrix
		for(long i = 0; i < size_r; i++)
			for(long j = 0; j < size_c; j++)
				A[i * size_c + j] = drand48();

		for(long i = 0; i < size_c; i++)
			B[i]= drand48();

		//Allocate memory for the GPU device
		double *A_d, *B_d, *sum_d;
		cudaMalloc((void**)&A_d, size_r * size_c * sizeof(double));
		cudaMalloc((void**)&B_d, size_c * sizeof(double));
		cudaMalloc((void**)&sum_d, size_r * sizeof(double));


		//Calling the CPU function
		//printf("Testing Matrix Vector Product: CPU\n");
		double tt = omp_get_wtime();
		mat_VecCPU(A, B, sum_ref, size_r, size_c);
		double cputime = omp_get_wtime() - tt;

		//printf("Testing Matrix Vector Product:GPU\n");
		long nblocks = (size_r + BLOCKSIZE - 1)/BLOCKSIZE;
		//printf("Blocks:  %d, Threads: %d\n", nblocks, BLOCKSIZE);
		
		cudaMemcpyAsync(A_d, A, size_r * size_c * sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpyAsync(B_d, B, size_c * sizeof(double),cudaMemcpyHostToDevice);
		tt = omp_get_wtime();
		mat_VecGPU_<<<nblocks, BLOCKSIZE>>>(A_d, B_d, sum_d, size_r, size_c);
		cudaDeviceSynchronize();
		double gputime = omp_get_wtime() - tt;
		cudaMemcpyAsync(sum, sum_d, size_r * sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		

		// double *A, *B, *sum, *sum_ref;
		// cudaMallocHost(&A, size_r * size_c * sizeof(double)); 
		// cudaMallocHost(&B, size_r * sizeof(double));
		// cudaMallocHost(&sum, size_r * size_c * sizeof(double));
		// cudaMallocHost(&sum_ref, size_r * sizeof(double));

		// //initialize the matrix
		// for(long i = 0; i < size_r; i++)
		// 	for(long j = 0; j < size_c; j++)
		// 		A[i * size_c + j] = drand48();

		// for(long i = 0; i < size_r; i++)
		// 	B[i]= drand48();

		// //Allocate memory for the GPU device
		// double *A_d, *B_d, *sum_d;
		// cudaMalloc((void**)&A_d, size_r * size_c * sizeof(double));
		// cudaMalloc((void**)&B_d, size_r * sizeof(double));
		// cudaMalloc((void**)&sum_d, size_r * size_c * sizeof(double));

		// cudaMemcpyAsync(A_d, A, size_r * size_c * sizeof(double),cudaMemcpyHostToDevice);
		// cudaMemcpyAsync(B_d, B, size_r * sizeof(double),cudaMemcpyHostToDevice);

		// //Calling the CPU function
		// double tt = omp_get_wtime();
		// mat_VecCPU(A, B, sum_ref, size_r, size_c);
		// double cputime = omp_get_wtime() - tt;
		// //Calling the GPu function
		// // long nthreads = NT;
		// // long nthreads_r = nthreads;
		// // long nthreads_c = nthreads;
		// // long nblock_r = ceil(1.0 * size_r/nthreads_r);
		// // long nblock_c = ceil(1.0 * size_c/nthreads_c);
		// // dim3 blocks(nblock_c, nblock_r);
		// // //dim3 blocks_P(1, nblock_r);
		// // dim3 threads(nthreads_r, nthreads_c);
		// //printf("Running GPU kernel\n");
		// //printf("blocks: %ld x %ld, threads: %ld x %ld\n", blocks.x, blocks.y, threads.x, threads.y);
		
		// long nblocks = (size_r * size_c + BLOCKSIZE - 1)/BLOCKSIZE;
		// tt = omp_get_wtime();
		// mat_VecGPU_<<<nblocks, BLOCKSIZE>>>(A_d, B_d, sum_d, size_r, size_c);
		// //mat_VecGPU<<<blocks, threads>>>(A_d, B_d, sum_d, size_r, size_c);
		// cudaDeviceSynchronize();
		// //long t = ceil(logf(nblock_c) / logf(nthreads));
		// //printf("t: %d \n", t);
		// // for(int i = 0; i < t; i++)
		// // {
		// // 	//printf("nblock_c: %d \n", nblock_c);
		// // 	dim3 blocks_P(nblock_c, nblock_r);
		// // 	addPartialSumsGPU_<<<blocks_P, threads>>>(sum_d, size_r, size_c);	
		// // 	cudaDeviceSynchronize();
		// // 	nblock_c = (nblock_c + nthreads_c - 1)/nthreads_c;
		// // }
		// // dim3 blocks_P(1, nblock_r);
		// // addPartialSumsGPU_<<<blocks_P, threads>>>(sum_d, size_r, size_c);
		// cudaDeviceSynchronize();
		// double gputime = omp_get_wtime() - tt;
		// cudaMemcpyAsync(sum, sum_d, size_r * size_c * sizeof(double), cudaMemcpyDeviceToHost);
		// cudaDeviceSynchronize();
		// 
		// for(long i = 0; i < size_r; i++)
		// {
		// 	printf("i = %ld  SUM_REF: %f, SUM: %f \n", i, sum_ref[i], sum[i]);
		// }

		//Find error
		double error = check_sum_(sum, sum_ref, size_r, size_c);
		printf("Size: %ld * %ld Error: %e CPU Time: %f GPU Time: %f Bandwidth: %f Gb/s Speedup: %f\n", size_r, size_c, error, cputime, gputime, (2*size_c*size_r + size_c)/gputime/1e9, cputime/gputime);

		cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(sum); cudaFreeHost(sum_ref);
		cudaFree(A_d); cudaFree(B_d); cudaFree(sum_d);
    } 


}
