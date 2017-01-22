#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
// подключене библиотеки cuBLAS
#include <cublas_v2.h>

// макрос для работы с индексами в стиле FORTRAN
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main()
{
    const int N = 4;
    cublasHandle_t handle;
    float *dev_x, *dev_A, *dev_b;
    int *dev_info;
    int *info;
    float *x, *A, *b;

    x = (float *)malloc (N * sizeof(*x));
    b = (float *)malloc (N * sizeof(*b));
    A = (float *)malloc (N * N * sizeof(*A));
    info = (int *)malloc (1 * sizeof(*info));

    // инициализация матрицы и вектора правой части
    int ind = 11;
    srand(time(NULL));
    for(int j = 0; j < N; j++)
    {
        for(int i = 0; i < N; i++)
        {
            { 
                A[IDX2C(i,j,N)]= (float) (rand() % 100);
            }
             A[IDX2C(i,i,N)] += 100;
        }
        b[j] = 1.0f;
    }

    // выделяем память на GPU соответствующего размера для каждой переменной
    cudaMalloc((void**)&dev_x, N * sizeof(*x));
    cudaMalloc((void**)&dev_b, N * sizeof(*x));
    cudaMalloc((void**)&dev_A, N * N * sizeof(*A));

    cudaMalloc((void**)&dev_info, N * sizeof(*A));

    // инициализируем контекст cuBLAS
    cublasCreate(&handle);

    // копируем вектор и матрицу из CPU в GPU
    cublasSetVector(N, sizeof(*b), b, 1, dev_b, 1);
    cublasSetMatrix(N, N, sizeof(*A), A, N, dev_A, N);


    cublasSgetrfBatched(handle, N, &dev_A, N, NULL, dev_info, 1);
    
    // решаем нижнюю треугольню матрицу
    cublasStrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, dev_A, N, dev_b, 1);

    // копируем результат из GPU в CPU
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%3.0f ", A[IDX2C(i,j,N)]);
        }
        printf("\n");
    }
    cublasGetVector(N, sizeof(*x), dev_b, 1, x, 1);
    cublasGetMatrix(N, N, sizeof(*A), dev_A, N, A, N);
    cublasGetVector(1, sizeof(*info), dev_info, 1, info, 1);

    printf("\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%3.2f ", A[IDX2C(i,j,N)]);
        }
        printf("\n");
    }

    printf("\n info = %d\n", info[0]);
    
    // освобождаем память в GPU
    cudaFree (dev_x);
    cudaFree (dev_b);
    cudaFree (dev_A);

    // уничтожаем контекс cuBLAS
    cublasDestroy(handle);

    // освобождаем память в CPU
    free(x);
    free(b);
    free(A);

    return EXIT_SUCCESS;
}
