// -*- c++ -*-
// =============================================================================
// bindings.cpp：把 CUDA 内核暴露给 Python（PyTorch 扩展）。
//
// 编译后在 Python 中得到模块 lhengine_cuda，提供：
//   scan_forward(a, b) -> y
//       一阶递推并行扫描。a,b: (C, L) float32 CUDA，序列在最后一维。
//   ssd_chunkwise(X, log_dA, B, C, S, chunk) -> (Y, S_out)
//       Mamba2 SSD 分块计算。布局见 ssd_kernel.cu 顶部注释。
// =============================================================================
#include <torch/extension.h>
#include <tuple>

// scan_kernel.cu
torch::Tensor scan_forward_cuda(torch::Tensor a, torch::Tensor b);

// ssd_kernel.cu
std::tuple<torch::Tensor, torch::Tensor>
ssd_chunkwise_cuda(torch::Tensor X, torch::Tensor log_dA,
                   torch::Tensor B, torch::Tensor C,
                   torch::Tensor S, int chunk);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "linear-hybrid-engine CUDA kernels (parallel scan + SSD)";
    m.def("scan_forward", &scan_forward_cuda,
          "Parallel first-order recurrent scan (float32)",
          py::arg("a"), py::arg("b"));
    m.def("ssd_chunkwise", &ssd_chunkwise_cuda,
          "Mamba2 SSD chunkwise forward (float32)",
          py::arg("X"), py::arg("log_dA"), py::arg("B"),
          py::arg("C"), py::arg("S"), py::arg("chunk"));
}
