#include <cstdio>
#include <cuda_runtime.h>

__global__ void empty_kernel() { /* nothing */ }

int main() {
  const int ITERS = 10000;
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  // warmup
  empty_kernel<<<1,1>>>();
  cudaDeviceSynchronize();

  // time
  cudaEventRecord(t0);
  for (int i = 0; i < ITERS; i++) {
    empty_kernel<<<1,1>>>();
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float ms;
  cudaEventElapsedTime(&ms, t0, t1);
  printf("launch time (avg over %d): %.3f µs\n",
         ITERS, (ms*1000)/ITERS);
  return 0;
}
