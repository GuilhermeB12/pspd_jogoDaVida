#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define TAMANHO_MIN_POTENCIA 3
#define TAMANHO_MAX_POTENCIA 10
#define INDICE(i, j, tamanho_tabuleiro) ((i)*(tamanho_tabuleiro + 2) + (j))



// Protótipos
__device__ int indice(int linha, int coluna, int tamanho_tabuleiro);
__global__ void jogar_uma_geracao(int* tabuleiro_atual, int* proximo_tabuleiro, int tamanho_tabuleiro);
double tempo_em_segundos(void);
void inicializar_tabuleiros(int* tabuleiro_atual, int* proximo_tabuleiro, int tamanho);
int verificar_resultado(int* tabuleiro, int tamanho);



int main(void) {
    for (int pot = TAMANHO_MIN_POTENCIA; pot <= TAMANHO_MAX_POTENCIA; pot++) {
        int tamanho = 1 << pot;
        int total_celulas = (tamanho + 2) * (tamanho + 2);

        double tempo0 = tempo_em_segundos();

        int *tabuleiro_atual  = (int*) malloc(total_celulas * sizeof(int));
        int *proximo_tabuleiro = (int*) malloc(total_celulas * sizeof(int));
        inicializar_tabuleiros(tabuleiro_atual, proximo_tabuleiro, tamanho);

        double tempo1 = tempo_em_segundos();

        int *d_tabuleiro_atual, *d_proximo_tabuleiro;
        cudaMalloc((void**)&d_tabuleiro_atual,  total_celulas * sizeof(int));
        cudaMalloc((void**)&d_proximo_tabuleiro, total_celulas * sizeof(int));

        cudaMemcpy(d_tabuleiro_atual,  tabuleiro_atual,  total_celulas * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_proximo_tabuleiro, proximo_tabuleiro, total_celulas * sizeof(int), cudaMemcpyHostToDevice);

        dim3 bloco(16, 16);
        dim3 grade((tamanho + 15) / 16, (tamanho + 15) / 16);

        for (int geracao = 0; geracao < 4 * (tamanho - 3); geracao++) {
            jogar_uma_geracao<<<grade, bloco>>>(d_tabuleiro_atual, d_proximo_tabuleiro, tamanho);
            int* temp = d_tabuleiro_atual;
            d_tabuleiro_atual = d_proximo_tabuleiro;
            d_proximo_tabuleiro = temp;
        }

        cudaMemcpy(tabuleiro_atual, d_tabuleiro_atual, total_celulas * sizeof(int), cudaMemcpyDeviceToHost);

        double tempo2 = tempo_em_segundos();

        if (verificar_resultado(tabuleiro_atual, tamanho))
            printf("Tamanho %d: RESULTADO CORRETO\n", tamanho);
        else
            printf("Tamanho %d: RESULTADO INCORRETO\n", tamanho);

        double tempo3 = tempo_em_segundos();

        printf("Tamanho=%d; tempos: init=%.6f, execucao=%.6f, final=%.6f, total=%.6f\n",
               tamanho, tempo1 - tempo0, tempo2 - tempo1, tempo3 - tempo2, tempo3 - tempo0);

        cudaFree(d_tabuleiro_atual);
        cudaFree(d_proximo_tabuleiro);
        free(tabuleiro_atual);
        free(proximo_tabuleiro);
    }

    return 0;
}

// Converte coordenadas (linha, coluna) para índice linear
__device__ int indice(int linha, int coluna, int tamanho_tabuleiro) {
    return linha * (tamanho_tabuleiro + 2) + coluna;
}

// Kernel CUDA: Aplica uma geração do Jogo da Vida
__global__ void jogar_uma_geracao(int* tabuleiro_atual, int* proximo_tabuleiro, int tamanho_tabuleiro) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i <= tamanho_tabuleiro && j <= tamanho_tabuleiro) {
        //int idx = i * (tamanho_tabuleiro + 2) + j;
        int idx = INDICE(i, j, tamanho_tabuleiro);

        int vizinhos_vivos =
            tabuleiro_atual[idx - (tamanho_tabuleiro + 3)] +
            tabuleiro_atual[idx - (tamanho_tabuleiro + 2)] +
            tabuleiro_atual[idx - (tamanho_tabuleiro + 1)] +
            tabuleiro_atual[idx - 1] +
            tabuleiro_atual[idx + 1] +
            tabuleiro_atual[idx + (tamanho_tabuleiro + 1)] +
            tabuleiro_atual[idx + (tamanho_tabuleiro + 2)] +
            tabuleiro_atual[idx + (tamanho_tabuleiro + 3)];

        if (tabuleiro_atual[idx] && vizinhos_vivos < 2)
            proximo_tabuleiro[idx] = 0;
        else if (tabuleiro_atual[idx] && vizinhos_vivos > 3)
            proximo_tabuleiro[idx] = 0;
        else if (!tabuleiro_atual[idx] && vizinhos_vivos == 3)
            proximo_tabuleiro[idx] = 1;
        else
            proximo_tabuleiro[idx] = tabuleiro_atual[idx];
    }
}

// Retorna tempo atual em segundos
double tempo_em_segundos(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Inicializa tabuleiros com todas as células mortas e insere um glider no canto superior esquerdo
void inicializar_tabuleiros(int* tabuleiro_atual, int* proximo_tabuleiro, int tamanho){
    int total = (tamanho + 2) * (tamanho + 2);
    for (int i = 0; i < total; i++) {
        tabuleiro_atual[i] = 0;
        proximo_tabuleiro[i] = 0;
    }

    tabuleiro_atual[(1)*(tamanho+2)+2] = 1;
    tabuleiro_atual[(2)*(tamanho+2)+3] = 1;
    tabuleiro_atual[(3)*(tamanho+2)+1] = 1;
    tabuleiro_atual[(3)*(tamanho+2)+2] = 1;
    tabuleiro_atual[(3)*(tamanho+2)+3] = 1;
}

// Verifica se o glider chegou ao canto inferior direito
int verificar_resultado(int* tabuleiro, int tamanho){
    return (
        tabuleiro[(tamanho-2)*(tamanho+2)+(tamanho-1)] &&
        tabuleiro[(tamanho-1)*(tamanho+2)+(tamanho  )] &&
        tabuleiro[(tamanho  )*(tamanho+2)+(tamanho-2)] &&
        tabuleiro[(tamanho  )*(tamanho+2)+(tamanho-1)] &&
        tabuleiro[(tamanho  )*(tamanho+2)+(tamanho  )]
    );
}