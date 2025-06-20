#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define POWMIN 3
#define POWMAX 10

#define ind2d(i,j) (i)*(tam+2)+j

double wall_time(void) {
  struct timeval tv;
  struct timezone tz;

  gettimeofday(&tv, &tz);
  return(tv.tv_sec + tv.tv_usec/1000000.0);
} /* fim-wall_time */

__global__ void UmaVidaKernel(int* tabulIn, int* tabulOut, int tam) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if (i <= tam && j <= tam) {
    int vizviv = 	tabulIn[ind2d(i-1,j-1)] + tabulIn[ind2d(i-1,j  )] +
          tabulIn[ind2d(i-1,j+1)] + tabulIn[ind2d(i  ,j-1)] +
          tabulIn[ind2d(i  ,j+1)] + tabulIn[ind2d(i+1,j-1)] +
          tabulIn[ind2d(i+1,j  )] + tabulIn[ind2d(i+1,j+1)];
    if (tabulIn[ind2d(i,j)] && vizviv < 2)
      tabulOut[ind2d(i,j)] = 0;
    else if (tabulIn[ind2d(i,j)] && vizviv > 3)
      tabulOut[ind2d(i,j)] = 0;
    else if (!tabulIn[ind2d(i,j)] && vizviv == 3)
      tabulOut[ind2d(i,j)] = 1;
    else
      tabulOut[ind2d(i,j)] = tabulIn[ind2d(i,j)];
  }
}

void DumpTabul(int * tabul, int tam, int first, int last, char* msg){
  int i, ij;

  printf("%s; Dump posicoes [%d:%d, %d:%d] de tabuleiro %d x %d\n", \
	 msg, first, last, first, last, tam, tam);
  for (i=first; i<=last; i++) printf("="); printf("=\n");
  for (i=ind2d(first,0); i<=ind2d(last,0); i+=ind2d(1,0)) {
    for (ij=i+first; ij<=i+last; ij++)
      printf("%c", tabul[ij]? 'X' : '.');
    printf("\n");
  }
  for (i=first; i<=last; i++) printf("="); printf("=\n");
} /* fim-DumpTabul */


void InitTabul(int* tabulIn, int* tabulOut, int tam){
  int ij;

  for (ij=0; ij<(tam+2)*(tam+2); ij++) {
    tabulIn[ij] = 0;
    tabulOut[ij] = 0;
  } /* fim-for */

  tabulIn[ind2d(1,2)] = 1; tabulIn[ind2d(2,3)] = 1;
  tabulIn[ind2d(3,1)] = 1; tabulIn[ind2d(3,2)] = 1;
  tabulIn[ind2d(3,3)] = 1;
} /* fim-InitTabul */


int Correto(int* tabul, int tam){
  int ij, cnt;

  cnt = 0;
  for (ij=0; ij<(tam+2)*(tam+2); ij++)
    cnt = cnt + tabul[ij];
  return (cnt == 5 && tabul[ind2d(tam-2,tam-1)] &&
      tabul[ind2d(tam-1,tam  )] && tabul[ind2d(tam  ,tam-2)] &&
      tabul[ind2d(tam  ,tam-1)] && tabul[ind2d(tam  ,tam  )]);
} /* fim-Correto */

int main(void) {
  int pow;
  int i, tam;
  int *tabulIn, *tabulOut;
  int *d_tabulIn, *d_tabulOut; // Device pointers
  double t0, t1, t2, t3;

  for (pow=POWMIN; pow<=POWMAX; pow++) {
    tam = 1 << pow;
    size_t size = (tam+2)*(tam+2)*sizeof(int);

    t0 = wall_time();
    // Allocate host memory
    tabulIn  = (int *) malloc(size);
    tabulOut = (int *) malloc(size);
    InitTabul(tabulIn, tabulOut, tam);

    // Allocate device memory
    cudaMalloc((void**)&d_tabulIn, size);
    cudaMalloc((void**)&d_tabulOut, size);

    // Copy initial data from host to device
    cudaMemcpy(d_tabulIn, tabulIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabulOut, tabulOut, size, cudaMemcpyHostToDevice);

    t1 = wall_time();

    dim3 blockSize(16, 16);
    dim3 gridSize((tam + blockSize.x - 1) / blockSize.x, (tam + blockSize.y - 1) / blockSize.y);

    for (i=0; i<2*(tam-3); i++) {
      UmaVidaKernel<<<gridSize, blockSize>>>(d_tabulIn, d_tabulOut, tam);
      int* temp = d_tabulIn;
      d_tabulIn = d_tabulOut;
      d_tabulOut = temp;
    }
    cudaDeviceSynchronize();

    t2 = wall_time();

    // Determine which buffer has the final result and copy it back to host
    if ((2*(tam-3)) % 2 != 0) { // Odd number of iterations
        cudaMemcpy(tabulIn, d_tabulOut, size, cudaMemcpyDeviceToHost);
    } else { // Even number of iterations
        cudaMemcpy(tabulIn, d_tabulIn, size, cudaMemcpyDeviceToHost);
    }


    if (Correto(tabulIn, tam))
      printf("**RESULTADO CORRETO**\n");
    else
      printf("**RESULTADO ERRADO**\n");

    t3 = wall_time();
    printf("tam=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f \n",
	   tam, t1-t0, t2-t1, t3-t2, t3-t0);

    // Free memory
    free(tabulIn);
    free(tabulOut);
    cudaFree(d_tabulIn);
    cudaFree(d_tabulOut);
  }
  return 0;
} /* fim-main */
