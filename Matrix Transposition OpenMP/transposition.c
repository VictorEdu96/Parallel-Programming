#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "omp.h"

//----------
int main (int argc, char **argv) {

  int rows, columns, proccesors, total_threads, i, j;
  float **Matrix, **Trans;
  FILE *fp;
  fp = fopen("./runs.txt", "a+");

//Asignar numero de procesadores, columnas y filas.
  proccesors = atoi(argv[1]);
  rows = atoi(argv[2]);
  columns = atoi(argv[3]);

//Llenar la matrix con numeros aleatorios y alocando memoria.
  Matrix = (float **) malloc(sizeof(float *) * rows);
  for(i = 0; i < rows; i++) {
    Matrix[i] = (float *) malloc(sizeof(float) * columns);
    for(j = 0; j < columns; j++) {
      Matrix[i][j] = rand() % 200;
    }
  }

  //Inicializar la matriz transpuesta a 0's alocando memoria.
  Trans = (float **) malloc(sizeof(float *) * rows);
  for(i = 0; i < rows; i++) {
    Trans[i] = (float *) malloc(sizeof(float) * columns);
    for(j = 0; j < columns; j++) {
      Trans[i][j] = 0;
    }
  }

  omp_set_num_threads(proccesors);
  clock_t t = clock();

  #pragma omp parallel for private(j)
  for (i = 0; i < rows; i++) {
    total_threads = omp_get_num_threads();

  
    for (j = 0; j < columns; j++) {
      Trans[i][j] = Matrix[j][i];
    }
  }

  t = clock() - t;
  double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
  
  printf("****** Tranposing took [%f] seconds to execute ******\n", time_taken);
  fprintf(fp, "Processors: %d, Matrix %dx%d: %f\n", proccesors, rows, columns, time_taken);
  fclose(fp);

// printf("------------ORIGINAL------------\n");
//     for (int i = 0; i < rows; i++) {
//       for (int j = 0; j < columns; j++) {
//         printf(" [%f]", Matrix[i][j]);
//       }
//       printf("\n");
//   }


// printf("------------TRANSPUESTA------------\n");
//     for (int i = 0; i < rows; i++) {
//       for (int j = 0; j < columns; j++) {
//         printf(" [%f]", Trans[i][j]);
//       }
//       printf("\n");
//   }

  free(Matrix);
  free(Trans);

  return 0;
}
