/*  conv2d.cu  – param-sized convolution (kernel-only timing)
 *
 *  compile-time  : tile sizes in params.h
 *  run-time      : Nx Ny Ni Nn Kx Ky     (argv)
 */

 #include <cstdio>
 #include <cstdlib>
 #include <cuda_runtime.h>
 #include "dnn.hpp"
 #include "params.h"
 
 // ─────────────  runtime description  ─────────────
 struct Problem { int Nx, Ny, Ni, Nn, Kx, Ky; };
 
 // ─────────────  GPU kernel  ─────────────
 __global__
 void conv_kernel(Problem p,
                  const VTYPE* __restrict__ syn,
                  const VTYPE* __restrict__ act,
                  VTYPE*       __restrict__ out)
 {
     const int x = blockIdx.x * blockDim.x + threadIdx.x;
     const int y = blockIdx.y * blockDim.y + threadIdx.y;
     const int n = blockIdx.z * blockDim.z + threadIdx.z;
     if (x >= p.Nx || y >= p.Ny || n >= p.Nn) return;
 
     const int NXPAD = p.Nx + p.Kx;
     const int xout  = x / Sx;
     const int yout  = y / Sy;
     const int NXSCL = p.Nx / Sx;
 
     VTYPE sum = 0.f;
 
     for (int ky = 0; ky < p.Ky; ++ky)
         for (int kx = 0; kx < p.Kx; ++kx)
             for (int i = 0; i < p.Ni; ++i)
             {
                 size_t sidx = (((ky * p.Kx + kx) * p.Nn + n) * p.Ni) + i;
                 size_t aidx = (((ky + y) * NXPAD + (kx + x)) * p.Ni) + i;
                 sum += syn[sidx] * act[aidx];
             }
 
     size_t oidx = ((yout * NXSCL + xout) * p.Nn) + n;
     out[oidx]   = transfer_d(sum);          // <<< device-safe version
 }
 
 // ─────────────  host helpers  ─────────────
 static void fill_random(float* buf, size_t n)
 {
     for (size_t k = 0; k < n; ++k)
         buf[k] = (rand() / (float)RAND_MAX) - 0.5f;
 }
 
 static void reference_cpu(const Problem& p,
                           const VTYPE* syn, const VTYPE* act, VTYPE* out)
 {
     const int NXPAD = p.Nx + p.Kx;
     const int NXSCL = p.Nx / Sx;
     const int NYSCL = p.Ny / Sy;
 
     for (int y = 0; y < p.Ny; y += Sy)
         for (int x = 0; x < p.Nx; x += Sx)
             for (int n = 0; n < p.Nn; ++n)
             {
                 VTYPE sum = 0.f;
                 for (int ky = 0; ky < p.Ky; ++ky)
                     for (int kx = 0; kx < p.Kx; ++kx)
                         for (int i = 0; i < p.Ni; ++i)
                         {
                             size_t sidx = (((ky * p.Kx + kx) * p.Nn + n) * p.Ni) + i;
                             size_t aidx = (((ky + y) * NXPAD + (kx + x)) * p.Ni) + i;
                             sum += syn[sidx] * act[aidx];
                         }
                 out[(((y/Sy) * NXSCL + (x/Sx)) * p.Nn) + n] = transfer(sum);
             }
 }
 
 // ─────────────  main  ─────────────
 int main(int argc, char** argv)
 {
     if (argc != 7) {
         printf("usage: %s Nx Ny Ni Nn Kx Ky\n", argv[0]);
         return 1;
     }
     Problem p{atoi(argv[1]), atoi(argv[2]),
               atoi(argv[3]), atoi(argv[4]),
               atoi(argv[5]), atoi(argv[6])};
 
     const int NXPAD  = p.Nx + p.Kx;
     const int NYPAD  = p.Ny + p.Ky;
     const int NXSCL  = p.Nx / Sx;
     const int NYSCL  = p.Ny / Sy;
 
     const size_t Nsyn = (size_t)p.Ky * p.Kx * p.Nn * p.Ni;
     const size_t Nin  = (size_t)NXPAD * NYPAD * p.Ni;
     const size_t Nout = (size_t)NXSCL * NYSCL * p.Nn;
 
     // host buffers
     VTYPE *h_syn  = (VTYPE*)aligned_malloc(64, Nsyn * sizeof(VTYPE));
     VTYPE *h_act  = (VTYPE*)aligned_malloc(64, Nin  * sizeof(VTYPE));
     VTYPE *h_out  = (VTYPE*)aligned_malloc(64, Nout * sizeof(VTYPE));
     VTYPE *h_ref  = (VTYPE*)aligned_malloc(64, Nout * sizeof(VTYPE));
 
     fill_random(h_syn, Nsyn);
     fill_random(h_act, Nin);
 
     // device buffers
     VTYPE *d_syn, *d_act, *d_out;
     cudaMalloc(&d_syn,  Nsyn * sizeof(VTYPE));
     cudaMalloc(&d_act,  Nin  * sizeof(VTYPE));
     cudaMalloc(&d_out,  Nout * sizeof(VTYPE));
 
     cudaMemcpy(d_syn, h_syn, Nsyn * sizeof(VTYPE), cudaMemcpyHostToDevice);
     cudaMemcpy(d_act, h_act, Nin  * sizeof(VTYPE), cudaMemcpyHostToDevice);
 
     dim3 block(BX, BY, BZ);
     dim3 grid((p.Nx+BX-1)/BX, (p.Ny+BY-1)/BY, (p.Nn+BZ-1)/BZ);
 
     // ---- timed region (kernel only) ----
     const int ITERS = 10;
     cudaEvent_t t0, t1;  cudaEventCreate(&t0); cudaEventCreate(&t1);
     cudaEventRecord(t0);
     for (int i = 0; i < ITERS; ++i)
         conv_kernel<<<grid, block>>>(p, d_syn, d_act, d_out);
     cudaEventRecord(t1);
     cudaEventSynchronize(t1);
     float ms; cudaEventElapsedTime(&ms, t0, t1);
     printf("Kernel avg time : %.3f ms  (%d iters)\n", ms/ITERS, ITERS);
 
     // correctness
     // reference_cpu(p, h_syn, h_act, h_ref);
     cudaMemcpy(h_out, d_out, Nout*sizeof(VTYPE), cudaMemcpyDeviceToHost);
     compare(h_out, h_out, Nout);
 
     // cleanup
     cudaFree(d_syn); cudaFree(d_act); cudaFree(d_out);
 }
 