#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" __global__ void vib_fused_kernel(
  const float* __restrict__ fc_std_raw,
  const float* __restrict__ mu,
  float* __restrict__ z,
  float* __restrict__ kl_sum,
  curandState* __restrict__ states,
  int batch_size, int latent_dim,
  float* __restrict__ std_out = nullptr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * latent_dim;
  if (idx >= total) return;

  int n = idx / latent_dim;
  int d = idx % latent_dim;
  int offset = n * latent_dim + d;

  // 1. Stable softplus: softplus(x - 5) = log1p(exp(x-5))
  float x = fc_std_raw[offset] - 5.0f;
  float std = log1p(expf(fminf(x, 30.0f)));  // avoid overflow

  // 2. Reparameterize
  float eps = curand_normal(&states[idx]);
  float sample = mu[offset] + std * eps;
  z[offset] = sample;

  // 3. KL term (per dimension)
  float std2 = std * std;
  float log_std2 = 2.0f * logf(std + 1e-8f);
  float kl_term = 0.5f * (mu[offset]*mu[offset] + std2 - log_std2 - 1.0f);

  // 4. Optional: output std
  if (std_out) std_out[offset] = std;

  // 5. Atomic add to shared KL sum (or use reduction later)
  atomicAdd(kl_sum, kl_term);
}
