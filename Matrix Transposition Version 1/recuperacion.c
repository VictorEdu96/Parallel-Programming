#include <stdio.h>
#include <stdlib.h>
int W,L;

typedef struct matrix  {
    int ** p;
}Matrix;

int dim,i,j ;
Matrix M;

void matrices(int W, int L){

    int dim = W * L;
    M.p=(int**) malloc (dim*sizeof(int *));

    for (W=0;W<dim;W++)  {
        M.p[W] = (int*) malloc (dim*sizeof(int));
        for (L=0; L<dim;L++)   {
            printf("Introduce el elemento [%d,%d] :  ", W+1, L+1 );
            scanf("%d",  &M.p [W][L]);
        }
    }

    printf("Matriz Original: \n");
    for (W=0; W<dim; W++)  {
        for (L=0; L<dim; L++)  {
            printf("%d  ", M.p[W][L]);
        }
        puts("\n");
    }
    printf("Matriz Transpuesta: \n");
    for (W=0; W<dim; W++)  {
        for (L=0; L<dim; L++)  {
            printf("%d  ", M.p[L][W]);
        }
        puts("\n");
    }

}

int main(int argc, char **argv){

    printf("Introduce las dimensiones de la matriz: ");
    scanf("%d", &W);
    scanf("%d", &L);
    matrices(W,L);

}
