/*
 * PARA CORRERLO
 *   export LD_LIBRARY_PATH=/usr/local/cuda/lib
 *   export PATH=$PATH:/usr/local/cuda/bin
 *   nvcc -o prueba prueba.cu -O2 -lc -lm
 *   ./prueba n n
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <assert.h>
 
 #define THREADS_PER_BLOCK_2D 16
 
 /* Kernel code */
 __global__ void transpose_gpu(const char *mat_in, char *mat_out, unsigned int rows, unsigned int cols)
 {
     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
 
     if (idx < cols && idy < rows)
     {
         unsigned int pos = idy * cols + idx;
         unsigned int trans_pos = idx * rows + idy;
 
         mat_out[trans_pos] = mat_in[pos];
     }
 }
 
 int main(int argc, char **argv)
 {
 
     /* Process command-line arguments */
     if (argc != 3)
     {
         fprintf(stderr, "Usage: %s rows columns\n", argv[0]);
         fprintf(stderr, "       rows is the number of rows of the input matrix\n");
         fprintf(stderr, "       columns is the number of columns of the input matrix\n");
         return EXIT_FAILURE;
     }
 
     cudaEvent_t start, stop;
     float elapsed_time_ms;
 
     unsigned int rows = atoi(argv[1]);
     unsigned int cols = atoi(argv[2]);
 
     //Para poner tiempos en matriz
     FILE *fp;
     fp = fopen("./times.txt", "a+");
 
     /* Pointer for host memory */
     char *h_mat_in, *h_mat_out;
     size_t mat_size = rows * cols * sizeof(char);
 
     /* Pointer for device memory */
     char *dev_mat_in, *dev_mat_out;
 
     /* Allocate host and device memory */
     h_mat_in = (char *)malloc(mat_size);
     h_mat_out = (char *)malloc(mat_size);
 
     cudaMalloc(&dev_mat_in, mat_size);
     cudaMalloc(&dev_mat_out, mat_size);
 
     /* Fixed seed for illustration */
     srand(2047);
 
     /* Initialize host memory */
     for (unsigned int i = 0; i < rows; ++i)
     {
         for (unsigned int j = 0; j < cols; ++j)
         {
             h_mat_in[i * cols + j] = rand() % (rows * cols);
             //printf("%d\t", h_mat_in[i * cols + j]);
         }
         //printf("\n");
     }
 
     // PRINTING ORIGINAL MATRIX
     //for (int i = 0; i < rows * cols; i++)
     //{
     //    printf("%i, ", h_mat_in[i]);
     //}
 
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
 
     /*------------------------ COMPUTATION ON GPU ----------------------------*/
 
     /* Host to device memory copy */
     cudaMemcpy(dev_mat_in, h_mat_in, mat_size, cudaMemcpyHostToDevice);
 
     /* Set grid and block dimensions properly */
     unsigned int grid_rows = (rows + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D;
     unsigned int grid_cols = (cols + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D;
     dim3 dimGrid(grid_cols, grid_rows);
     dim3 dimBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);
 
     cudaEventRecord(start, 0);
 
     /* Launch kernel */
     transpose_gpu<<<dimGrid, dimBlock>>>(dev_mat_in, dev_mat_out, rows, cols);
 
     cudaEventRecord(stop, 0);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&elapsed_time_ms, start, stop);
 
     /* device to host copy */
     cudaMemcpy(h_mat_out, dev_mat_out, mat_size, cudaMemcpyDeviceToHost);
 
     printf("Time to transpose a matrix of %dx%d on GPU: %f ms.\n\n", rows, cols, elapsed_time_ms);
 
     //IMPRIMIR TIEMPOS EN ARCHIVO times.txt
     fprintf(fp, "[%d x %d] = %f ms.\n\n", rows, cols, elapsed_time_ms);
     fclose(fp);
 
     // PRINTING TRANSPOSED MATRIX
     //for (int i = 0; i < rows * cols; i++)
     //{
     //  printf("%i, ", h_mat_out[i]);
     //}
 
     /* Free host and device memory */
     free(h_mat_in);
     free(h_mat_out);
     cudaFree(dev_mat_in);
     cudaFree(dev_mat_out);
 }