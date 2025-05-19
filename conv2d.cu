#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "dnn.hpp"
#include "params.h"    // provides Tx, Ty, Ti, Tn, Sx, Sy

using namespace std;

// Problem description
struct Problem { int Nx, Ny, Ni, Nn, Kx, Ky; };

// Global-memory convolution (no shared memory), 2D blocks with Tn loop
__global__ void conv_kernel_global(Problem p,
                                   const VTYPE* __restrict__ syn,
                                   const VTYPE* __restrict__ act,
                                   VTYPE*       __restrict__ out)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * Tx;
    int by = blockIdx.y * Ty;
    int x  = bx + tx;
    int y  = by + ty;
    if (x >= p.Nx || y >= p.Ny) return;

    const int Nxpad = p.Nx + p.Kx;
    const int xout  = x / Sx;
    const int yout  = y / Sy;
    const int Nxscl = p.Nx / Sx;

    // loop over Tn output channels
    for (int tn = 0; tn < Tn; ++tn) {
        int n = blockIdx.z * Tn + tn;
        if (n >= p.Nn) break;

        VTYPE sum = 0.f;
        // convolution loops
        for (int ky = 0; ky < p.Ky; ++ky) {
            for (int kx = 0; kx < p.Kx; ++kx) {
                size_t w_base = ((size_t)ky * p.Kx + kx) * p.Nn + n;
                w_base *= p.Ni;
                size_t a_base = ((size_t)(ky + y) * Nxpad + (kx + x)) * p.Ni;

                for (int ci = 0; ci < p.Ni; ci += Ti) {
                    int limit = min(Ti, p.Ni - ci);
                    for (int i = 0; i < limit; ++i) {
                        VTYPE w = syn[w_base + ci + i];
                        VTYPE a = act[a_base + ci + i];
                        sum += w * a;
                    }
                }
            }
        }

        size_t oidx = ((size_t)yout * Nxscl + xout) * p.Nn + n;
        out[oidx] = transfer_d(sum);
    }
}

// Shared-memory tiled convolution
__global__ void conv_kernel_shared(Problem p,
                                   const VTYPE* __restrict__ syn,
                                   const VTYPE* __restrict__ act,
                                   VTYPE*       __restrict__ out)
{
    extern __shared__ VTYPE shmem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * Tx;
    int by = blockIdx.y * Ty;
    int n0 = blockIdx.z * Tn;  // base output-channel index

    const int aW = Tx + p.Kx - 1;
    const int aH = Ty + p.Ky - 1;

    for (int tn = 0; tn < Tn; ++tn) {
        int n = n0 + tn;
        if (n >= p.Nn) break;
        VTYPE sum = 0.f;

        // tile over Ni
        for (int ci = 0; ci < p.Ni; ci += Ti) {
            int limit = min(Ti, p.Ni - ci);
            int w_count = p.Kx * p.Ky * limit;
            int a_count = aW    * aH    * limit;
            VTYPE* s_w   = shmem;
            VTYPE* s_act = shmem + w_count;

            // load weights
            for (int idx = ty*Tx + tx; idx < w_count; idx += Tx*Ty) {
                int i  = idx % limit;
                int k  = idx / limit;
                int ky = k / p.Kx;
                int kx = k % p.Kx;
                size_t w_base = ((size_t)ky * p.Kx + kx) * p.Nn + n;
                s_w[k*limit + i] = syn[w_base * p.Ni + ci + i];
            }
            // load activations
            for (int idx = ty*Tx + tx; idx < a_count; idx += Tx*Ty) {
                int tmp = idx;
                int i   = tmp % limit; tmp /= limit;
                int ax  = tmp % aW;     tmp /= aW;
                int ay  = tmp;
                int gx  = bx + ax;
                int gy  = by + ay;
                if (gx < p.Nx + p.Kx && gy < p.Ny + p.Ky) {
                    size_t aidx = ((size_t)gy * (p.Nx + p.Kx) + gx) * p.Ni + ci + i;
                    s_act[(ay * aW + ax) * limit + i] = act[aidx];
                } else {
                    s_act[(ay * aW + ax) * limit + i] = (VTYPE)0;
                }
            }
            __syncthreads();

            // compute
            for (int ky = 0; ky < p.Ky; ++ky)
                for (int kx = 0; kx < p.Kx; ++kx)
                    for (int i = 0; i < limit; ++i) {
                        int w_idx = (ky * p.Kx + kx) * limit + i;
                        int a_idx = ((ky + ty) * aW + (kx + tx)) * limit + i;
                        sum += s_w[w_idx] * s_act[a_idx];
                    }
            __syncthreads();
        }

        // write output
        int x = bx + tx;
        int y = by + ty;
        if (x < p.Nx && y < p.Ny) {
            int xout = x / Sx;
            int yout = y / Sy;
            size_t oidx = ((size_t)yout * (p.Nx / Sx) + xout) * p.Nn + n;
            out[oidx] = transfer_d(sum);
        }
        __syncthreads();
    }
}

// Host helper: random fill
static void fill_random(float* buf, size_t n) {
    for (size_t i = 0; i < n; ++i)
        buf[i] = (rand() / (float)RAND_MAX) - 0.5f;
}

// CPU reference
static void reference_cpu(const Problem& p,
                          const VTYPE* syn,
                          const VTYPE* act,
                          VTYPE*       out)
{
    int Nxpad = p.Nx + p.Kx;
    int Nxscl = p.Nx / Sx;
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
                out[(((y / Sy) * Nxscl + (x / Sx)) * p.Nn) + nn] = transfer(acc);
            }
}

int main(int argc, char** argv)
{
    if (argc != 7) {
        printf("usage: %s Nx Ny Ni Nn Kx Ky\n", argv[0]);
        return 1;
    }
    Problem p { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
                atoi(argv[4]), atoi(argv[5]), atoi(argv[6]) };

    int Nxpad = p.Nx + p.Kx;
    int Nypad = p.Ny + p.Ky;
    int Nxscl = p.Nx / Sx;
    size_t Nsyn = (size_t)p.Ky * p.Kx * p.Nn * p.Ni;
    size_t Nin  = (size_t)Nxpad * Nypad * p.Ni;
    size_t Nout = (size_t)Nxscl * (p.Ny / Sy) * p.Nn;

    VTYPE *h_syn = (VTYPE*)aligned_malloc(64, Nsyn * sizeof(VTYPE));
    VTYPE *h_act = (VTYPE*)aligned_malloc(64, Nin  * sizeof(VTYPE));
    VTYPE *h_out = (VTYPE*)aligned_malloc(64, Nout * sizeof(VTYPE));
    VTYPE *h_ref = (VTYPE*)aligned_malloc(64, Nout * sizeof(VTYPE));
    fill_random(h_syn, Nsyn);
    fill_random(h_act, Nin);

    VTYPE *d_syn, *d_act, *d_out;
    cudaMalloc(&d_syn, Nsyn * sizeof(VTYPE));
    cudaMalloc(&d_act, Nin  * sizeof(VTYPE));
    cudaMalloc(&d_out, Nout * sizeof(VTYPE));
    cudaMemcpy(d_syn, h_syn, Nsyn * sizeof(VTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_act, h_act, Nin  * sizeof(VTYPE), cudaMemcpyHostToDevice);

    dim3 block_shared(Tx, Ty, 1);
    dim3 grid_shared((p.Nx + Tx - 1) / Tx,
                     (p.Ny + Ty - 1) / Ty,
                     (p.Nn + Tn - 1) / Tn);
    size_t shared_bytes = ((size_t)p.Kx * p.Ky * Ti +
                          (size_t)(Tx + p.Kx - 1) * (Ty + p.Ky - 1) * Ti) * sizeof(VTYPE);

    dim3 block_global(Tx, Ty, 1);
    dim3 grid_global((p.Nx + Tx - 1) / Tx,
                     (p.Ny + Ty - 1) / Ty,
                     (p.Nn + Tn - 1) / Tn);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    bool use_smem = (shared_bytes <= prop.sharedMemPerBlock);
    printf("Using %s kernel (shared_bytes=%zu, limit=%d)\n",
           use_smem ? "shared" : "global",
           shared_bytes, prop.sharedMemPerBlock);

    //reference_cpu(p, h_syn, h_act, h_ref);

    const int ITERS = 10;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    if (use_smem) {
        for (int i = 0; i < ITERS; ++i)
            conv_kernel_shared<<<grid_shared, block_shared, shared_bytes>>>(p, d_syn, d_act, d_out);
    } else {
        for (int i = 0; i < ITERS; ++i)
            conv_kernel_global<<<grid_global, block_global>>>(p, d_syn, d_act, d_out);
    }
    cudaError_t errLaunch = cudaGetLastError();
    if (errLaunch != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(errLaunch));
        return -1;
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaError_t errSync = cudaGetLastError();
    if (errSync != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(errSync));
        return -1;
    }
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("Kernel avg time : %.3f ms  (%d iters)\n", ms/ITERS, ITERS);

    cudaMemcpy(h_out, d_out, Nout * sizeof(VTYPE), cudaMemcpyDeviceToHost);
    //compare(h_out, h_ref, Nout);

    cudaFree(d_syn);
    cudaFree(d_act);
    cudaFree(d_out);
    return 0;
}
