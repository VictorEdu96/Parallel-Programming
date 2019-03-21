#include <stdio.h>
#include <mpi.h>

void trans(double *a, int n)
{
    int i, j;
    int ij, ji, l;
    double tmp;
    ij = 0;
    l = -1;
    for (i = 0; i < n; i++)
    {
        l += n + 1;
        ji = l;
        ij += i + 1;
        for (j = i + 1; j < n; j++)
        {
            tmp = a[ij];
            a[ij] = a[ji];
            a[ji] = tmp;
            ij++;
            ji += n;
        }
    }
}

int main(int argc, char *argv[])
{

    int i, j, nprocs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ps = 128 / nprocs;
    double a[128][ps];
    double b[128][ps];
    double time_start, time_end;

    for (i = 0; i < 128; i++)
    {
        for (j = 0; j < ps; j++)
        {
            a[i][j] = 1000 * i + j + ps * rank;
        }
    }

    time_start = MPI_Wtime();

    MPI_Alltoall(&a[0][0],
                 ps * ps,
                 MPI_DOUBLE,
                 &b[0][0],
                 ps * ps,
                 MPI_DOUBLE,
                 MPI_COMM_WORLD);

    for (i = 0; i < nprocs; i++)
        trans(&b[i * ps][0], ps);

    for (i = 0; i < 128; i++)
    {
        for (j = 0; j < ps; j++)
        {
            if (b[i][j] != 1000 * (j + ps * rank) + i)
            {
                printf("process %d found b[%d][%d] = %f, but %f was expected\n",
                       rank, i, j, b[i][j], (double)(1000 * (j + ps * rank) + i));
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }

    if (rank == 0)
    {
        printf("Transpose seems ok\n");
    }

    time_end = MPI_Wtime();
    printf("Time took transposing: %f\n", time_end - time_start);

    MPI_Finalize();
    return 0;
}