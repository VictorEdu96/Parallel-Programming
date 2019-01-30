#include <stdio.h>
#include <time.h>

//Funcion para solo imprimir matriz.
// void showMatrix(double *a, int w, int l) {
//   for (int i = 0; i < w; i++) {
//     for (int j = 0; j < l; j++) {
//       printf("%g%c", a[i][j], j == 2 ? '\n' : ' ');
//     }
//     puts("\n");
//   }
// }

//Transponer la matriz
void transposition(void *original, void *destino, int w, int l) {

double (*d)[w] = original;
double (*s)[l] = destino;
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < l; j++) {
      d[j][i] = s[i][j];
    }
  }
  // showMatrix(*d,3,3);
}

int main(int argc, char const *argv[]) {
  int i, j;
  clock_t t;
  double threeByThree[10][10] = {{1,2,3,5,3,6,7,5,4,3},
                                {4,5,6,6,4,7,8,6,3,6},
                                {7,8,9,6,1,4,5,8,3,9},
                                {7,5,3,7,9,0,5,2,7,8},
                                {6,2,3,3,4,5,7,8,9,10},
                                {1,2,3,5,3,6,7,5,4,3},
                                {4,5,6,6,4,7,8,6,3,6},
                                {7,8,9,6,1,4,5,8,3,9},
                                {7,5,3,7,9,0,5,2,7,8},
                                {6,2,3,3,4,5,7,8,9,10}};

  double transposedMatrix[10][10];
  transposition(transposedMatrix, threeByThree, 10, 10);
  for (i = 0; i < 10; i++)
		for (j = 0; j < 10; j++)
			printf("%g%c", transposedMatrix[i][j], j == 9 ? '\n' : ' ');
  return 0;
}
