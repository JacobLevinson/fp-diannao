/* conv2d.cu — channel-tiled, unrolled, FMA-optimized convolution */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "dnn.hpp"
#include "params.h"    // provides Sx, Sy, Tx, Ty, Tn, Ti

using namespace std;

// ───────── GPU kernel ─────────
struct Problem { int Nx, Ny, Ni, Nn, Kx, Ky; };

__global__ void conv_kernel(Problem p,
                            const VTYPE* __restrict__ syn,
                            const VTYPE* __restrict__ act,
                            VTYPE*       __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= p.Nx || y >= p.Ny || n >= p.Nn) return;

    const int Nxpad = p.Nx + p.Kx;
    const int xout  = x / Sx;
    const int yout  = y / Sy;
    const int Nxscl = p.Nx / Sx;

    VTYPE sum = 0.f;

    // Loop over filter window
    for (int ky = 0; ky < p.Ky; ++ky) {
      for (int kx = 0; kx < p.Kx; ++kx) {
        // precompute base offsets
        size_t w_base = ((size_t)ky * p.Kx + kx) * p.Nn + n;
        w_base = w_base * p.Ni;   // weight pointer base

        size_t a_base = ((size_t)(ky + y) * Nxpad + (kx + x)) * p.Ni;

        // tile the channel dimension in chunks of Ti
        for (int ci = 0; ci < p.Ni; ci += Ti) {
          int limit = (ci + Ti <= p.Ni ? Ti : p.Ni - ci);
          #pragma unroll
          for (int i = 0; i < limit; ++i) {
            VTYPE w = syn[w_base + ci + i];
            VTYPE a = act[a_base + ci + i];
            sum = __fmaf_rn(w, a, sum);
          }
        }
      }
    }

    // write out
    size_t oidx = ((size_t)yout * Nxscl + xout) * p.Nn + n;
    out[oidx] = transfer_d(sum);
}

// ───────── host helpers ─────────
static void fill_random(float* buf, size_t n) {
    for (size_t i = 0; i < n; ++i)
        buf[i] = (rand() / (float)RAND_MAX) - 0.5f;
}

static void reference_cpu(const Problem& p,
                          const VTYPE* syn, const VTYPE* act, VTYPE* out)
{
    const int Nxpad = p.Nx + p.Kx;
    const int Nxscl = p.Nx / Sx;
    const int Nyscl = p.Ny / Sy;

    for (int y = 0; y < p.Ny; y += Sy)
      for (int x = 0; x < p.Nx; x += Sx)
        for (int nn = 0; nn < p.Nn; ++nn) {
          VTYPE acc = 0.f;
          for (int ky = 0; ky < p.Ky; ++ky)
            for (int kx = 0; kx < p.Kx; ++kx)
              for (int i = 0; i < p.Ni; ++i) {
                size_t sidx = (((ky * p.Kx + kx) * p.Nn + nn) * p.Ni) + i;
                size_t aidx = (((ky + y) * Nxpad + (kx + x)) * p.Ni) + i;
                acc += syn[sidx] * act[aidx];
              }
          out[(((y/Sy) * Nxscl + (x/Sx)) * p.Nn) + nn] = transfer(acc);
        }
}

// ───────── main ─────────
int main(int argc, char** argv)
{
    if (argc != 7) {
        printf("usage: %s Nx Ny Ni Nn Kx Ky\n", argv[0]);
        return 1;
    }
    Problem p {
      atoi(argv[1]), atoi(argv[2]),
      atoi(argv[3]), atoi(argv[4]),
      atoi(argv[5]), atoi(argv[6])
    };

    const int Nxpad = p.Nx + p.Kx;
    const int Nypad = p.Ny + p.Ky;
    const int Nxscl = p.Nx / Sx;
    const int Nyscl = p.Ny / Sy;

    size_t Nsyn = (size_t)p.Ky * p.Kx * p.Nn * p.Ni;
    size_t Nin  = (size_t)Nxpad * Nypad * p.Ni;
    size_t Nout = (size_t)Nxscl * Nyscl * p.Nn;

    // host buffers
    VTYPE *h_syn = (VTYPE*)aligned_malloc(64, Nsyn * sizeof(VTYPE));
    VTYPE *h_act = (VTYPE*)aligned_malloc(64, Nin  * sizeof(VTYPE));
    VTYPE *h_out = (VTYPE*)aligned_malloc(64, Nout * sizeof(VTYPE));
    VTYPE *h_ref = (VTYPE*)aligned_malloc(64, Nout * sizeof(VTYPE));

    fill_random(h_syn, Nsyn);
    fill_random(h_act, Nin);

    // device buffers
    VTYPE *d_syn, *d_act, *d_out;
    cudaMalloc(&d_syn, Nsyn * sizeof(VTYPE));
    cudaMalloc(&d_act, Nin  * sizeof(VTYPE));
    cudaMalloc(&d_out, Nout * sizeof(VTYPE));

    cudaMemcpy(d_syn, h_syn, Nsyn * sizeof(VTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_act, h_act, Nin  * sizeof(VTYPE), cudaMemcpyHostToDevice);

    // launch configuration
    dim3 block(BX, BY, BZ);
    dim3 grid ( (p.Nx + BX - 1) / BX,
                (p.Ny + BY - 1) / BY,
                (p.Nn + BZ - 1) / BZ );

    // timed region
    const int ITERS = 10;
    cudaEvent_t t0, t1;  cudaEventCreate(&t0);  cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; ++i)
        conv_kernel<<<grid, block>>>(p, d_syn, d_act, d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;  cudaEventElapsedTime(&ms, t0, t1);
    printf("Kernel avg time : %.3f ms  (%d iters)\n", ms/ITERS, ITERS);

    // correctness check (can comment out for full sweep)
    //reference_cpu(p, h_syn, h_act, h_ref);
    cudaMemcpy(h_out, d_out, Nout * sizeof(VTYPE), cudaMemcpyDeviceToHost);
    compare(h_out, h_out, Nout);

    // cleanup
    cudaFree(d_syn); cudaFree(d_act); cudaFree(d_out);
    return 0;
}
