#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "omp.h"
#define N 5
#define M 5


//----------
int matrix[N][M];
int transpose[N][M];
FILE *fp;
//----------


void printMatrix(int matrix[N][M]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf(" [%d]", matrix[i][j]);
    }
    printf("\n");
  }
}


void generateRandomMatrix(){
  printf("-----------ORIGINAL  MATRIX----------\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      matrix[i][j] = rand() % N;
    }
  }
  printMatrix(matrix);
  printf("-------------------------------------\n");
}


void transposeMatrix(){
  printf("\n----------TRANSPOSED MATRIX----------\n");
  int i, j;
  clock_t t = clock();


  //Chucu chucu
  #pragma omp parallel for private(j)
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      transpose[i][j] = matrix[j][i];
    }
  }


  t = clock() - t;
  double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
  
  printMatrix(transpose);

  printf("****** Tranposing took [%f] seconds to execute ******\n", time_taken);
  fprintf(fp, "%dx%d: %f\n", N, M, time_taken);
  fclose(fp);
  printf("-------------------------------------\n");
}

int main(int argc, char const *argv[]) {
  fp = fopen("./runs.txt", "a+");
  generateRandomMatrix();
  transposeMatrix();
}
