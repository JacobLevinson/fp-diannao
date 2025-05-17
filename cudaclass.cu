#include <iostream>
#include <cuda_runtime.h>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 128  // Number of Output Layers
  #define Ni 224  // Number of Input  Layers
#endif

#ifndef Tii
  // Tiling Sizes
  #define Tnn 32  
  #define Tii 32
  //#define Tn 5
  //#define Ti 25
  #define Tn 16
  #define Ti 16
#endif

#define WEIGHT_BYTES    (Nn * Ni * sizeof(VTYPE))
#define INPUT_BYTES     (Ni * sizeof(VTYPE))
#define OUTPUT_BYTES    (Nn * sizeof(VTYPE))
#define OUTPUT_ELEMENTS (Nn)

/* A simple 1-D kernel, so for example */
#define BX 256
#define BY 1
#define BZ 1

//Arrays:
VTYPE synapse[Nn][Ni] __attribute__((aligned(64)));
VTYPE neuron_i[Ni] __attribute__((aligned(64)));
VTYPE neuron_n[Nn] __attribute__((aligned(64))),    neuron_n2[Nn] __attribute__((aligned(64)));

void fill_classifier(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
    VTYPE (&neuron_n)[Nn],   VTYPE (&neuron_n2)[Nn]) {
  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      synapse[n][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }
  for(int i = 0; i < Ni; ++i) {
    neuron_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  for(int n = 0; n < Nn; ++n) {
    neuron_n[n] = 0; //i;
    neuron_n2[n] = 0; //i;
  }
}

void classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
  for (int n = 0; n < Nn; n++) {
    VTYPE temp=0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}

__global__ void classifier_layer_kernel(VTYPE* synapse,
                                        VTYPE* neuron_i,
                                        VTYPE* neuron_n);

float run_kernel(int iters,
                 VTYPE* d_syn, VTYPE* d_in, VTYPE* d_out,
                 dim3 grid, dim3 block)
{
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float total_ms = 0.0f;

    for (int it = 0; it < iters; ++it)
    {
        cudaMemcpyAsync(d_in, neuron_i, 
                        INPUT_BYTES,  // = NYPAD*NXPAD*Ni*sizeof(VTYPE)  or  Ni*sizeof(VTYPE)
                        cudaMemcpyHostToDevice);

        cudaEventRecord(t0);
        classifier_layer_kernel<<<grid, block>>>(d_syn, d_in, d_out);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms;  cudaEventElapsedTime(&ms, t0, t1);
        total_ms += ms;
    }
    return total_ms / iters;
}
__global__ void classifier_layer_kernel(VTYPE* synapse, VTYPE* neuron_i, VTYPE* neuron_n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= Nn) return;       

  VTYPE temp=0;
  for (int i = 0; i < Ni; i++) {
    int synapse_index = tid * Ni + i;
    temp += synapse[synapse_index] * neuron_i[i];
  }
  neuron_n[tid] = temp > 0 ? temp : temp / 4; 
}

// void classifier_layer_cuda(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
//   // Allocate space for device copies for synapse, neuron_i, and neuron_n
//   VTYPE *d_synapse, *d_neuron_i, *d_neuron_n;
//   cudaMalloc((void**)&d_synapse, Nn*Ni*sizeof(VTYPE));
//   cudaMalloc((void**)&d_neuron_i, Ni*sizeof(VTYPE));
//   cudaMalloc((void**)&d_neuron_n, Nn*sizeof(VTYPE));
  
//   // Copy inputs to device
//   cudaMemcpy(d_synapse, synapse, Nn*Ni*sizeof(VTYPE), cudaMemcpyHostToDevice); 
//   cudaMemcpy(d_neuron_i, neuron_i, Ni*sizeof(VTYPE), cudaMemcpyHostToDevice);

//   // Launch the kernel
//   classifier_layer_kernel<<<Nn/256, 256>>>(d_synapse, d_neuron_i, d_neuron_n);

//   // Copy result back to host
//   cudaMemcpy(neuron_n, d_neuron_n, Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);

//   // Cleanup
//   cudaFree(d_synapse);
//   cudaFree(d_neuron_i);
//   cudaFree(d_neuron_n);
// }

void classifier_layer_blocked(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
                              VTYPE (&neuron_n)[Nn]) {
  VTYPE sum[Nn]={0};
  for (int nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
    for (int iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
      for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
        for (int ii = iii; ii < iii + Tii; ii += Ti) {
          // — Original code —
          for (int n = nn; n < nn + Tn; n++) {
            VTYPE sum_sc=0;
            for (int i = ii; i < ii + Ti; i++) {
              sum_sc += (synapse[n][i] * neuron_i[i]);
            }
            sum[n]+=sum_sc;
          }
        }
      }
    }
    for (int nn = nnn; nn < nnn + Tnn; nn++) {
      neuron_n[nn] = transfer(sum[nn]);
    }
  }
}

int main(int argc, char** argv) {
  cout << "initializing arrays\n";

  // ---------------- one-time setup ----------------
  fill_classifier(synapse, neuron_i, neuron_n, neuron_n2);

  VTYPE *d_syn, *d_in, *d_out;
  cudaMalloc(&d_syn, WEIGHT_BYTES);          // e.g. SYNAPSE_SIZE*sizeof(VTYPE)
  cudaMalloc(&d_in , INPUT_BYTES);           // size of one activation tensor
  cudaMalloc(&d_out, OUTPUT_BYTES);          // NYSCL*NXSCL*Nn*sizeof(VTYPE)

  cudaMemcpy(d_syn, synapse, WEIGHT_BYTES, cudaMemcpyHostToDevice);

  // pick grid/block that you already computed
  dim3 block(BX, 1, 1);
  dim3 grid((Nn + BX - 1) / BX, 1, 1);

  // ---------------- timed region ------------------
  int ITERS = 30;
  float ms = run_kernel(ITERS, d_syn, d_in, d_out, grid, block);
  printf("KERNEL time (avg of %d): %.3f ms\n", ITERS, ms);

  // optional correctness
  classifier_layer(synapse, neuron_i, neuron_n2);
  cudaMemcpy(neuron_n, d_out, OUTPUT_BYTES, cudaMemcpyDeviceToHost);
  compare(neuron_n, neuron_n2, OUTPUT_ELEMENTS);

  // cleanup
  cudaFree(d_syn); cudaFree(d_in); cudaFree(d_out);
}

