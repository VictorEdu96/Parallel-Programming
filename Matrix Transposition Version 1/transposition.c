#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define N 700
#define M 700


//----------
int matrix[N][M];
int transpose[N][M];
FILE *fp;
//----------


void printMatrix(int matrix[N][M]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf("%d\t", matrix[i][j]);
    }
    printf("\n");
  }
}


void generateRandomMatrix(){
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      matrix[i][j] = rand() % N;
    }
  }
  printMatrix(matrix);
}


void transposeMatrix(){

  clock_t t = clock();


  //Chucu chucu
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      transpose[i][j] = matrix[j][i];
    }
  }
  t = clock() - t;
  double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds


  printf("Tranposing took %f seconds to execute \n", time_taken);
  fprintf(fp, "%dx%d: %f\n", N, M, time_taken);
  fclose(fp);
  printMatrix(transpose);
}

int main(int argc, char const *argv[]) {
  fp = fopen("./runs.txt", "a+");
  generateRandomMatrix();
  transposeMatrix();
}
