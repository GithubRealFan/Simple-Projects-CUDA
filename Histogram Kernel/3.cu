#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <typeinfo>
#include <cuda_runtime.h>

using std::generate;

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  //@@ Insert code below to compute histogram of input using shared memory and atomics
  __shared__ unsigned int _bins[NUM_BINS];

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    _bins[i] = 0;

  __syncthreads();

  // Calculate global thread ID
  for (int i = threadIdx.x; i < num_elements; i += blockDim.x)
    atomicAdd(&_bins[input[i]], 1);

  __syncthreads();

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    bins[i] = _bins[i];
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
  //@@ Insert code below to clean up bins that saturate at 127
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

  // Boundary check
  if (idx < num_bins)
    if (bins[idx] > 127)
      bins[idx] = 127;
}

int main(int argc, char **argv) {
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);

  printf("Data Type: %s\n", typeid(*hostInput).name());

  //@@ Insert code below to allocate Host memory for input and output
  size_t bytesInput = inputLength * sizeof(unsigned int);
  size_t bytesBins = NUM_BINS * sizeof(unsigned int);
  hostInput = (unsigned int*)malloc(bytesInput);
  hostBins = (unsigned int*)malloc(bytesBins);
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  generate(hostInput, hostInput + inputLength, []() { return rand() % NUM_BINS; });

  //@@ Insert code below to create reference result in CPU
  resultRef = (unsigned int*)malloc(bytesBins);
  
  memset(resultRef, 0, bytesBins);
  for (int i = 0; i < inputLength; ++i)
    resultRef[hostInput[i]] += 1;
  for (int i = 0; i < NUM_BINS; ++i)
    if (resultRef[i] > 127)
      resultRef[i] = 127;

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, bytesInput);
  cudaMalloc(&deviceBins, bytesBins);

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, bytesInput, cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, bytesBins);

  //@@ Initialize the grid and block dimensions here
  int THREADS_1 = 1024;
  int BLOCKS_1 = 1;//(inputLength + THREADS_1 - 1) / THREADS_1;

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<BLOCKS_1, THREADS_1>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
  int THREADS_2 = 1024;
  int BLOCKS_2 = (NUM_BINS + THREADS_2 - 1) / THREADS_2;

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<BLOCKS_2, THREADS_2>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, bytesBins, cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < NUM_BINS; ++i)
    if (hostBins[i] != resultRef[i]) {
      printf("Wrong\n");
      break;
    }

  //@@ Print histogram values
  FILE *fp = fopen("3.csv", "w");
  for (int i = 0; i < NUM_BINS; ++i)
    fprintf(fp, "%d\n", hostBins[i]);
  fclose(fp);

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);

  return 0;
}
