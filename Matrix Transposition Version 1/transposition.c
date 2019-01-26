#include <stdio.h>
#define SIZE(x)  (sizeof(x) / sizeof((x)[0]))

//Funcion para solo imprimir matriz.
void showMatrix(int *a) {
  size_t w = SIZE(a);
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < w; j++) {
      printf("%d ", *a++);
    }
    puts("\n");
  }
}

//Transponer la matriz
void transposition(int *a) {

  /* TO DO:
  1.- Regresar puntero a matriz transpuesta
  2.- Ver si se puede usar doble puntero
  */

  size_t w = SIZE(a);
  int transpuesta[w][w];
  printf("Matriz Transpuesta: \n");
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < w; j++) {
      transpuesta[j][i] = *a++;
    }
  }
  showMatrix(*transpuesta);
}

int main(int argc, char const *argv[]) {
int twoByTwo[2][2] = {1,2,
                      3,4};
// showMatrix(*twoByTwo);
transposition(*twoByTwo);
}
