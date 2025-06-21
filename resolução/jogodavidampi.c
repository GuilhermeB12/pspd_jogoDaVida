#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define POWMIN 3
#define POWMAX 10
#define ind2d(i,j) ((i)*(tam+2)+(j))

double wall_time(void) {
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return (tv.tv_sec + tv.tv_usec/1000000.0);
}

void UmaVida(int* tabulIn, int* tabulOut, int tam, int ini, int fim) {
  int i, j, vizviv;
  for (i=ini; i<=fim; i++) {
    for (j=1; j<=tam; j++) {
      vizviv = tabulIn[ind2d(i-1,j-1)] + tabulIn[ind2d(i-1,j)] + tabulIn[ind2d(i-1,j+1)] +
               tabulIn[ind2d(i  ,j-1)] +                         tabulIn[ind2d(i  ,j+1)] +
               tabulIn[ind2d(i+1,j-1)] + tabulIn[ind2d(i+1,j)] + tabulIn[ind2d(i+1,j+1)];

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
}

void InitTabul(int* tabulIn, int* tabulOut, int tam) {
  int ij;
  for (ij = 0; ij < (tam+2)*(tam+2); ij++) {
    tabulIn[ij] = 0;
    tabulOut[ij] = 0;
  }

  tabulIn[ind2d(1,2)] = 1; tabulIn[ind2d(2,3)] = 1;
  tabulIn[ind2d(3,1)] = 1; tabulIn[ind2d(3,2)] = 1;
  tabulIn[ind2d(3,3)] = 1;
}

int Correto(int* tabul, int tam) {
  int ij, cnt = 0;
  for (ij = 0; ij < (tam+2)*(tam+2); ij++)
    cnt += tabul[ij];
  return (cnt == 5 && tabul[ind2d(tam-2,tam-1)] &&
                   tabul[ind2d(tam-1,tam  )] &&
                   tabul[ind2d(tam  ,tam-2)] &&
                   tabul[ind2d(tam  ,tam-1)] &&
                   tabul[ind2d(tam  ,tam  )]);
}

int main(int argc, char** argv) {
  int rank, size, pow, tam, *tabIn, *tabOut;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Rank %d: iniciou execução\n", rank);
  fflush(stdout);


  for (pow = POWMIN; pow <= POWMAX; pow++) {
    tam = 1 << pow;

    tabIn  = (int *) malloc((tam+2)*(tam+2)*sizeof(int));
    tabOut = (int *) malloc((tam+2)*(tam+2)*sizeof(int));

    if (rank == 0)
      InitTabul(tabIn, tabOut, tam);

    MPI_Bcast(tabIn, (tam+2)*(tam+2), MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: terminou MPI_Bcast\n", rank);
    fflush(stdout);


    // Divisão das linhas
    int linhas = tam / size;
    int resto = tam % size;

    int ini = rank * linhas + 1 + (rank < resto ? rank : resto);
    int fim = ini + linhas - 1 + (rank < resto ? 1 : 0);
    int blocos = fim - ini + 1;

    double t0 = wall_time();

    for (int it = 0; it < 2*(tam-3); it++) {
      if (it == 0 && rank == 0) {
        printf("Rank %d: começando primeira iteração\n", rank);
        fflush(stdout);
    }

      // Troca de bordas entre processos
        if (rank > 0)
        MPI_Sendrecv(&tabIn[ind2d(ini,1)], tam, MPI_INT, rank-1, 0,
                    &tabIn[ind2d(ini-1,1)], tam, MPI_INT, rank-1, 1,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Envia para baixo, recebe de baixo
        if (rank < size-1)
        MPI_Sendrecv(&tabIn[ind2d(fim,1)], tam, MPI_INT, rank+1, 1,
                    &tabIn[ind2d(fim+1,1)], tam, MPI_INT, rank+1, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      UmaVida(tabIn, tabOut, tam, ini, fim);
      int* tmp = tabIn; tabIn = tabOut; tabOut = tmp;
    }

    // Preparação para o Gatherv
    int* recvbuf = NULL;
    int* recvcounts = NULL;
    int* displs = NULL;

    if (rank == 0) {
      recvbuf = (int*) malloc(tam * tam * sizeof(int));
      recvcounts = (int*) malloc(size * sizeof(int));
      displs = (int*) malloc(size * sizeof(int));

      int offset = 0;
      for (int i = 0; i < size; i++) {
        int ini_i = i * linhas + 1 + (i < resto ? i : resto);
        int fim_i = ini_i + linhas - 1 + (i < resto ? 1 : 0);
        int blocos_i = fim_i - ini_i + 1;
        recvcounts[i] = blocos_i * tam;
        displs[i] = offset;
        offset += recvcounts[i];
      }
    }

    MPI_Gatherv(&tabIn[ind2d(ini,1)], blocos * tam, MPI_INT,
                recvbuf, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
      // Copia os dados de volta pro tabuleiro completo
      for (int i = 0; i < tam * tam; i++)
        tabIn[ind2d(1 + i / tam, 1 + i % tam)] = recvbuf[i];

      if (Correto(tabIn, tam))
        printf("**RESULTADO CORRETO**\n");
      else
        printf("**RESULTADO ERRADO**\n");

      double t3 = wall_time();
      printf("tam=%d; tempo total: %7.7f s\n", tam, t3 - t0);

      free(recvbuf);
      free(recvcounts);
      free(displs);
    }

    free(tabIn);
    free(tabOut);
  }

  MPI_Finalize();
  return 0;
}
