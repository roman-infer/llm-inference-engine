#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

// 主机端（float）参考实现。GPU kernel 以 atol/rtol = 1e-2 对齐验证。

namespace cputest {

// C[M,N] = A[M,K] * B[K,N]，行主序。
inline std::vector<float> gemm_ref(const std::vector<float> &A,
                                   const std::vector<float> &B, int M, int N,
                                   int K) {
  std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      const float a = A[m * K + k];
      for (int n = 0; n < N; ++n) {
        C[m * N + n] += a * B[k * N + n];
      }
    }
  }
  return C;
}

// GEMV：y[n] = sum_k x[k] * w[n*K+k]（w 为 [N,K] 布局）。
inline std::vector<float> gemv_ref(const std::vector<float> &x,
                                   const std::vector<float> &w, int N, int K) {
  std::vector<float> y(N, 0.0f);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) y[n] += x[k] * w[n * K + k];
  return y;
}

inline float gelu_ref(float v) {
  return 0.5f * v * (1.0f + std::erf(v * 0.70710678f));
}
inline float silu_ref(float v) { return v / (1.0f + std::exp(-v)); }

inline std::vector<float> rmsnorm_ref(const std::vector<float> &x,
                                      const std::vector<float> &w, int rows,
                                      int cols, float eps) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    float ss = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float v = x[r * cols + c];
      ss += v * v;
    }
    const float scale = 1.0f / std::sqrt(ss / cols + eps);
    for (int c = 0; c < cols; ++c)
      y[r * cols + c] = x[r * cols + c] * scale * w[c];
  }
  return y;
}

inline std::vector<float> layernorm_ref(const std::vector<float> &x,
                                        const std::vector<float> &w,
                                        const std::vector<float> *b, int rows,
                                        int cols, float eps) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) mean += x[r * cols + c];
    mean /= cols;
    float var = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float d = x[r * cols + c] - mean;
      var += d * d;
    }
    var /= cols;
    const float scale = 1.0f / std::sqrt(var + eps);
    for (int c = 0; c < cols; ++c) {
      float v = (x[r * cols + c] - mean) * scale * w[c];
      if (b) v += (*b)[c];
      y[r * cols + c] = v;
    }
  }
  return y;
}

inline std::vector<float> softmax_ref(const std::vector<float> &x, int rows,
                                      int cols) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    float mx = -INFINITY;
    for (int c = 0; c < cols; ++c) mx = std::max(mx, x[r * cols + c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      y[r * cols + c] = std::exp(x[r * cols + c] - mx);
      sum += y[r * cols + c];
    }
    for (int c = 0; c < cols; ++c) y[r * cols + c] /= sum;
  }
  return y;
}

// RoPE，作用于 [seq, heads, head_dim]。
inline void rope_ref(std::vector<float> &x, int seq, int heads, int head_dim,
                     int pos0, float theta_base) {
  for (int s = 0; s < seq; ++s) {
    for (int h = 0; h < heads; ++h) {
      for (int d2 = 0; d2 < head_dim / 2; ++d2) {
        const int p = pos0 + s;
        const float angle =
            p * std::pow(theta_base, -2.0f * d2 / head_dim);
        float *e =
            &x[(static_cast<size_t>(s) * heads + h) * head_dim + 2 * d2];
        const float x0 = e[0], x1 = e[1];
        e[0] = x0 * std::cos(angle) - x1 * std::sin(angle);
        e[1] = x0 * std::sin(angle) + x1 * std::cos(angle);
      }
    }
  }
}

}  // namespace cputest
