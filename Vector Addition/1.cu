#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <cuda_runtime.h>

using std::generate;

typedef double DataType;

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

  // Boundary check
  if (idx < len)
    out[idx] = in1[idx] + in2[idx];
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
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  size_t bytes = inputLength * sizeof(DataType);
  hostInput1 = (DataType*)malloc(bytes);
  hostInput2 = (DataType*)malloc(bytes);
  hostOutput = (DataType*)malloc(bytes);

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  generate(hostInput1, hostInput1 + inputLength, []() { return rand() / ((DataType)rand() + 1); });
  generate(hostInput2, hostInput2 + inputLength, []() { return rand() / ((DataType)rand() + 1); });

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, bytes);
  cudaMalloc(&deviceInput2, bytes);
  cudaMalloc(&deviceOutput, bytes);

  //@@ Insert code to below to Copy memory to the GPU here
  timerStart();
  cudaMemcpy(deviceInput1, hostInput1, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, bytes, cudaMemcpyHostToDevice);
  timerStop("Host to Deice");

  //@@ Initialize the 1D grid and block dimensions here
  int BLOCK_SIZE = 1 << 10;
  int GRID_SIZE = (inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE;

  //@@ Launch the GPU Kernel here
  timerStart();
  vecAdd<<<GRID_SIZE, BLOCK_SIZE>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  timerStop("Kernel");

  //@@ Copy the GPU memory back to the CPU here
  timerStart();
  cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  timerStop("Device To Host");

  //@@ Insert code below to compare the output with the reference
  resultRef = (DataType*)malloc(bytes);
  for (int i = 0; i < inputLength; ++i)
    resultRef[i] = hostInput1[i] + hostInput2[i];

  for (int i = 0; i < inputLength; ++i)
    if (fabs(hostOutput[i] - resultRef[i]) > 1e-6) {
      printf("Wrong\n");
      break;
    }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
