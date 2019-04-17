/*
 * PARA CORRERLO
 *   export LD_LIBRARY_PATH=/usr/local/cuda/lib
 *   export PATH=$PATH:/usr/local/cuda/bin
 *   nvcc -o prueba prueba.cu -O2 -lc -lm
 *   ./prueba n
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
 
 #define THREADS_PER_BLOCK_2D 16
  
 /* Kernel code */
 __global__ void transpose_gpu(const char *mat_in, char *mat_out, unsigned int rows, unsigned int cols){
     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  
     if (idx < cols && idy < rows) {
         unsigned int pos = idy * cols + idx;
         unsigned int trans_pos = idx * rows + idy;
  
         mat_out[trans_pos] = mat_in[pos];
     }
}


void transponerMatrix(char *dmi, char *dmo, unsigned int r, unsigned int c){
    
}

void printMatrix(unsigned int rows, unsigned int cols, char *matrix){
    for (int i = 0; i < rows * cols; i++) {
       printf("%i, ", matrix[i]);
    }
}


int main(int argc, char **argv) {
 
    /* Process command-line arguments */
    if (argc != 2) {
        fprintf(stderr, "Usage: %s matrix_size\n", argv[0]);
        fprintf(stderr, "       matrix_size is the number of rows and cols of the matrix\n");
        return EXIT_FAILURE;
    }
 
    cudaEvent_t start, stop;                                                                // --> Va en funcion transponerMatrix
    float elapsed_time_ms;                                                                  // --> Va en funcion transponerMatrix
 
    unsigned int rows = atoi(argv[1]);
    unsigned int cols = atoi(argv[1]);
 
    //Para poner tiempos en matriz
    FILE *fp;
    fp = fopen("./times.txt", "a+");
 
    /* Pointer for host memory */
    char *h_mat_in, *h_mat_out;                                                             // --> Va en funcion transponerMatrix
    size_t mat_size = rows * cols * sizeof(char);                                           // --> Va en funcion transponerMatrix
 
    /* Pointer for device memory */
    char *dev_mat_in, *dev_mat_out;
 
    /* Allocate host and device memory */
    h_mat_in = (char *)malloc(mat_size);
    h_mat_out = (char *)malloc(mat_size);
 
    cudaMalloc(&dev_mat_in, mat_size);                                                      // --> Va en funcion transponerMatrix
    cudaMalloc(&dev_mat_out, mat_size);                                                     // --> Va en funcion transponerMatrix
 
    /* Fixed seed for illustration */
    srand(2047);                        // --> Va en funcion transponerMatrix
 
    /* Initialize host memory */
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            h_mat_in[i * cols + j] = rand() % (rows * cols);
            //printf("%d\t", h_mat_in[i * cols + j]);
        }
        //printf("\n");
    }
 
    // PRINTING ORIGINAL MATRIX
    // printMatrix(rows, cols, h_mat_in);
 
    cudaEventCreate(&start); cudaEventCreate(&stop);                                        // --> Va en funcion transponerMatrix
 
    /*------------------------ COMPUTATION ON GPU ----------------------------*/
 
    /* Host to device memory copy */
    cudaMemcpy(dev_mat_in, h_mat_in, mat_size, cudaMemcpyHostToDevice);                     // --> Va en funcion transponerMatrix
 
    /* Set grid and block dimensions properly */
    unsigned int grid_rows = (rows + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D;      // --> Va en funcion transponerMatrix
    unsigned int grid_cols = (cols + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D;      // --> Va en funcion transponerMatrix
    dim3 dimGrid(grid_cols, grid_rows);                                                     // --> Va en funcion transponerMatrix
    dim3 dimBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);                              // --> Va en funcion transponerMatrix
 
    cudaEventRecord(start, 0);                                                              // --> Va en funcion transponerMatrix
 
    /* Launch kernel */
    transpose_gpu<<<dimGrid, dimBlock>>>(dev_mat_in, dev_mat_out, rows, cols);              // --> Va en funcion transponerMatrix
 
    cudaEventRecord(stop, 0);                                                               // --> Va en funcion transponerMatrix
    cudaEventSynchronize(stop);                                                             // --> Va en funcion transponerMatrix
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);                                    // --> Va en funcion transponerMatrix
 
    /* device to host copy */
    cudaMemcpy(h_mat_out, dev_mat_out, mat_size, cudaMemcpyDeviceToHost);
 
    printf("Time to transpose a matrix of %dx%d on GPU: %f ms.\n\n", rows, cols, elapsed_time_ms);
 
    //IMPRIMIR TIEMPOS EN ARCHIVO times.txt
    fprintf(fp, "[%d x %d] = %f ms.\n\n", rows, cols, elapsed_time_ms);
    fclose(fp);
 
    // PRINTING TRANSPOSED MATRIX
    // printMatrix(rows, cols, h_mat_out);

    /* Free host and device memory */
    free(h_mat_in); free(h_mat_out); cudaFree(dev_mat_in); cudaFree(dev_mat_out);
}