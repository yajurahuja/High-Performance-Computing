/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 500

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
double a[N][N];
//FIX: The issue is the thread stack size is limited hence adding a private variable matrix a with a larger size than the thread stack will lead to an error. There are two fixes. The easy one is to decrease the size of N. Or other way is the increase the size of the thread stack that can be done. I have done the easier fix. 
/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
  {

  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);
  /* Each thread works on its own private copy of the array */
  #pragma omp barrier
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i][j] = tid + i + j;


  printf("Thread %d done. Last element= %f\n",tid, a[N-1][N-1]);
  
}  /* All threads join master thread and disband */

}

