#include <stdio.h>
#include <time.h>
#define N 5
#define M 3

void showMatrix(int matrix[N][M]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf("%d\t", matrix[i][j]);
    }
    printf("\n");
  }
}

int main(int argc, char const *argv[]) {
  clock_t t;
  int matrix[N][M];
  int transpose[N][M];
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      matrix[i][j] = rand() % N;
    }
  }

  showMatrix(matrix);
  t = clock();

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      transpose[i][j] = matrix[j][i];
    }
    printf("\n");
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

  printf("Tranposing) took %f seconds to execute \n", time_taken);


  showMatrix(transpose);

}
