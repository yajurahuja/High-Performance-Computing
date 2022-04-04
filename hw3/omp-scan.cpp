 #include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define nthreads 8
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i <= n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;
  prefix_sum[0] = 0;
  long size_task = ceil((1.0) * n / nthreads);
  #pragma omp parallel shared(prefix_sum, A, n)
  { 
    #pragma omp single nowait
    {
      for(int i = 0; i < nthreads; i++)
      {
        #pragma omp task
        {
          long start = i * size_task + 1;
          long end = std::min((i + 1) * size_task, n) + 1;
          for(int j = start; j < end; j++)
          {
            if(j == start)
              prefix_sum[j] = A[j-1];
            else
              prefix_sum[j] = prefix_sum[j-1] + A[j-1];

          }
        }
      } 
    }
  }

  for(int i = 1; i < nthreads; i++)
  {
    long start = i * size_task + 1;
    long end = std::min((i + 1) * size_task, n) + 1;
    long offset = prefix_sum[start - 1];
    #pragma omp parallel for 
    for(int j = start; j < end; j++)
    {
      prefix_sum[j] += offset;
    } 
  }

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc((N+1) * sizeof(long));
  long* B1 = (long*) malloc((N+1) * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N+1);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N+1);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N + 1; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
