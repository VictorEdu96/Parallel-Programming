/*
 * PARA CORRERLO:
 *   $ export LD_LIBRARY_PATH=/usr/local/cuda/lib
 *   $ export PATH=$PATH:/usr/local/cuda/bin
 *   $ nvcc -o matrixTrans matrixTrans.cu -O2 -lc -lm
 *   $ ./matrixTrans n
*/

/*
 * UNSIGNED INT --> Tipo de dato para enteros, números sin punto decimal. 
 *                  Los enteros sin signo pueden ser tan grandes como 65535 
 *                  y tan pequeños como 0. 
 *                  Son almacenados como 16 bits de información.
 *
 * SIZE_T --> is an unsigned integer type guaranteed to support the longest 
 *            object for the platform you use. It is also the result of the 
 *            sizeof operator.sizeof returns the size of the type in bytes.
 *            So in your context of question in both cases you pass a 
 *            size_t to malloc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h> 
#define NUMBER_THREADS 32

float elapsed_time_ms;  
int gpudev = 0;

char *dev_mat_in, *dev_mat_out;

//---------------------------------------------------------------------------

void printMatrix(unsigned int rows, unsigned int cols, char *matrix){
   for (unsigned int i = 0; i < rows; ++i) {
       for (unsigned int j = 0; j < cols; ++j) {
           printf("%d\t", matrix[i * cols + j]);
       }
       printf("\n");
   }
}

void printTimeOnFile(unsigned int rows, unsigned int cols, float elapsed_time_ms){
   FILE *fp;
   fp = fopen("./times.txt", "a+");
   fprintf(fp, "[%d x %d] = %f ms.\n\n", rows, cols, elapsed_time_ms);
   fclose(fp);
}

__global__ void kernelTransposeMatrix(const char *mat_in, char *mat_out, unsigned int rows, unsigned int cols){
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
 
    if (idx < cols && idy < rows) {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}

void transponerMatrix(char *h_mat_in, char *h_mat_out, unsigned int rows, unsigned int cols, size_t size){
   unsigned int g_row = (unsigned int)ceil((float)size/32.0);
   unsigned int g_col = (unsigned int)ceil((float)size/32.0);
   dim3 bloques(g_col, g_row);
   dim3 hilos(NUMBER_THREADS, NUMBER_THREADS);
   cudaEvent_t start, stop;

   cudaSetDevice(gpudev);

   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   
   // 1. Asignar memoria
   cudaMalloc(&dev_mat_in, size);
   cudaMalloc(&dev_mat_out, size);

   // 2. Copiar datos del Host al Device
   cudaMemcpy(dev_mat_in, h_mat_in, size, cudaMemcpyHostToDevice);

   // 3. Ejecutar kernel
   kernelTransposeMatrix<<<bloques, hilos>>>(dev_mat_in, dev_mat_out, rows, cols);

   // 4. Copiar datos del device al Host
   cudaMemcpy(h_mat_out, dev_mat_out, size, cudaMemcpyDeviceToHost);

   // 5. Liberar Memoria
   cudaFree(dev_mat_in);
   cudaFree(dev_mat_out);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed_time_ms, start, stop);
   cudaEventDestroy(start); 
   cudaEventDestroy(stop); 
}
//---------------------------------------------------------------------------
int main(int argc, char **argv) {
   if (argc != 2) {
       fprintf(stderr, "Must be executed as: %s matrix_size\n", argv[0]);
       return EXIT_FAILURE;
   }

   unsigned int rows = atoi(argv[1]);
   unsigned int cols = rows;

   char *h_mat_in, *h_mat_out;
   size_t size = rows * cols * sizeof(char);
   h_mat_in = (char *)malloc(size);
   h_mat_out = (char *)malloc(size);

   for (unsigned int i = 0; i < rows; ++i) {
       for (unsigned int j = 0; j < cols; ++j) {
           h_mat_in[i * cols + j] = rand() % (rows * cols);
       }
   }
   //printMatrix(rows, cols, h_mat_in);

   transponerMatrix(h_mat_in, h_mat_out, rows, cols, size);

   printf("\n***** Time to transpose a matrix of [%dx%d] on GPU: [%f] ms. *****\n\n", rows, cols, elapsed_time_ms);
   printTimeOnFile(rows, cols, elapsed_time_ms);
   //printMatrix(rows, cols, h_mat_out);
   free(h_mat_in); free(h_mat_out); cudaFree(dev_mat_in); cudaFree(dev_mat_out);
}