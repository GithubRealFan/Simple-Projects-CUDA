#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <typeinfo>
#include <cuda_runtime.h>

using std::generate;

typedef float DataType;

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here

  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (!(row < numARows && col < numBColumns))
    return;

  // Iterate over row, and down column
  DataType tmp = 0;
  for (int i = 0; i < numAColumns; ++i) {
    // Accumulate results for a single element
    tmp += A[row * numAColumns + i] * B[i * numBColumns + col];
  }
  C[row * numBColumns + col] = tmp;
}

// Check result on the CPU
void gemm__(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns) {
  // For every row...
  for (int i = 0; i < numARows; ++i) {
    // For every column...
    for (int j = 0; j < numBColumns; ++j) {
      // For every element in the row-column pair
      DataType tmp = 0;
      for (int k = 0; k < numAColumns; ++k) {
        // Accumulate the partial results
        tmp += A[i * numAColumns + k] * B[k * numBColumns + j];
      }
      C[i * numBColumns + j] = tmp;
    }
  }
}

//@@ Insert code to implement timer start
clock_t st, en;
void timerStart() {
  st = clock();
}

//@@ Insert code to implement timer stop
void timerStop(char stepName[]) {
  en = clock();
  clock_t elapsed = en - st;
  printf("%s: %u ms elapsed.\n", stepName, elapsed);
}

int main(int argc, char **argv) {  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBColumns = atoi(argv[3]);
  numBRows = numAColumns;
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  printf("Data Type: %s\n", typeid(DataType).name());

  //@@ Insert code below to allocate Host memory for input and output
  size_t bytesA = numARows * numAColumns * sizeof(DataType);
  size_t bytesB = numBRows * numBColumns * sizeof(DataType);
  size_t bytesC = numCRows * numCColumns * sizeof(DataType);
  hostA = (DataType*)malloc(bytesA);
  hostB = (DataType*)malloc(bytesB);
  hostC = (DataType*)malloc(bytesC);
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  generate(hostA, hostA + numARows * numAColumns, []() { return rand() / (DataType)RAND_MAX; });
  generate(hostB, hostB + numBRows * numBColumns, []() { return rand() / (DataType)RAND_MAX; });

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, bytesA);
  cudaMalloc(&deviceB, bytesB);
  cudaMalloc(&deviceC, bytesC);

  //@@ Insert code to below to Copy memory to the GPU here
  timerStart();
  cudaMemcpy(deviceA, hostA, bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, bytesB, cudaMemcpyHostToDevice);
  timerStop("Host to Deice");

  //@@ Initialize the grid and block dimensions here
  int THREADS = 32;
  int BLOCKS_ROW = (numARows + THREADS - 1) / THREADS;
  int BLOCKS_COL = (numBColumns + THREADS - 1) / THREADS;
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_COL, BLOCKS_ROW);

  //@@ Launch the GPU Kernel here
  timerStart();
  gemm<<<blocks, threads>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  timerStop("Kernel");

  //@@ Copy the GPU memory back to the CPU here
  timerStart();
  cudaMemcpy(hostC, deviceC, bytesC, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  timerStop("Device To Host");

  //@@ Insert code below to compare the output with the reference
  resultRef = (DataType*)malloc(bytesC);
  gemm__(hostA, hostB, resultRef, numARows, numAColumns, numBRows, numBColumns);

  for (int i = 0; i < numCRows * numCColumns; ++i)
    if (fabs(hostC[i] - resultRef[i]) > 1e-4) {
      printf("Wrong\n");
      break;
    }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
