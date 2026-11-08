# DeepSeek-Serving：从算子到分布式服务的全栈推理引擎

一个面向 **DeepSeek-V3/R1 新架构**的自研推理引擎，按"**节点内核 → 节点引擎 → 分布式服务**"三层从零实现，并在单卡 RTX 4060 Laptop（Ada **sm_89**, 8GB）上给出**可复现的真实 benchmark** 与**诚实的收益边界**。

> 设计对齐 2025–2026 生产系统：vLLM / Mooncake / DistServe / Splitwise / NVIDIA Dynamo / DeepEP / EPLB。
> Kernel 路线：PyTorch ATen + `torch._scaled_mm`（FP8 Tensor Core），不依赖 nvcc / Triton；后端可替换为 CUDA / Triton kernel。

---

## 三层架构

```
┌──────────────────────────── 集群服务层（cluster/，~950 LOC，本仓新增） ────────────────────────────┐
│                                                                                                    │
│   ┌──────────────── GlobalScheduler ────────────────┐                                              │
│   │ 准入控制 · P/D 速率匹配 · SLO 放置 · 前缀感知路由 │                                              │
│   └───────┬──────────────────────────────────┬────────┘                                            │
│           ▼                                  △                                                     │
│  ┌──── Prefill 引擎集群 ────┐   KV Connector  ┌──── Decode 引擎集群 ────┐                       │
│  │ chunked prefill → latent KV → 切块入池  │ ─handshake+kv─▶ │ 导入 KV 块 → 导入首 token → 逐 token 解码 │
│  └─────────────┬───────────┘                └──────────────┬─────────────┘                       │
│                ▼                                            ▼                                       │
│   ┌──────────────── 分布式 KV 池（BlockPool）─────────────────┐                                    │
│   │ 物理块 + 块表 + 引用计数 + CoW 前缀共享                    │                                    │
│   │ TieredStore：GPU HBM → CPU DRAM → SSD（LRU 淘汰）         │                                    │
│   │ PrefixIndex：最长公共前缀块复用                           │                                    │
│   └──────────────────────────────────────────────────────────┘                                    │
│                                                                                                    │
│   宽专家并行（EP）：token 行按 origin rank → all-to-all dispatch → 本地 grouped SwiGLU            │
│   → all-to-all combine 加权求和；EPLB 风格 LoadBalancer 复制热点专家、均衡 rank 利用率              │
└──────────────────────────────────────┬─────────────────────────────────────────────────────────────┘
                                      │ 复用（不重写算法）
┌─────────────────────────────────────▼─────────────────────────────────────────────────────────────┐
│ 节点内核（node_backend/deepseek_infer/，~1000 LOC，唯一一份）                                       │
│                                                                                                    │
│   FP8 Linear（per-tensor / 1×128 block）  MLA（低秩 KV 压缩 + weight absorption）                  │
│   MoE（top-k 路由 + 共享专家 + grouped GEMM）  MTP（草稿头 + 单链/树状投机解码 + 拒绝采样）          │
│   模型：DeepSeek Block / RMSNorm / RoPE / SwiGLU / 采样 / latent KV cache                          │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**复用边界**：`node_backend/deepseek_infer/` 是唯一一份节点内算子库（包内全相对导入，搬位置无需改）；`cluster/` 只调用、不重写其算法，重写的是**组织方式**——稠密 KV buffer → 分页块池；单进程 MoE → 跨 rank EP；单机 Generator → PD 分离 + Connector。

---

## 实测数据（RTX 4060 Laptop, sm_89, torch 2.x / CUDA 12.x）

### 节点内核

| 模块 | 实测 | 工程判断（不堆名词） |
| --- | --- | --- |
| **FP8 GEMM** | M1024×K4096×N8192 达 **53.4 TFLOPS**，较 fp16（31.2）加速 **1.71×**，cos=0.9993 | 小 GEMM（M128）上 fp16 反快——launch-overhead 占主导；blockwise 真实价值是 **outlier 隔离**而非精度保护；sm_89 无硬件 block-scale，生产应优先单次 tensorwise |
| **MLA 注意力** | KV 体积压缩 **3.0×**（1536→512 B/token）；absorbed vs naive 误差 **<0.002** | weight absorption 不减 FLOPs，核心收益是**避免物化历史 k_nope/v 的显存与读带宽**；长上下文 / 大 d_c 场景延迟优势才显现 |
| **MoE grouped GEMM** | grouped 延迟对 token 数不敏感：N=32→256 稳定 **~1.03ms**；与逐 token 路径 Frobenius 误差 **0.00038** | naive 基线是 Python 逐 token 循环（24×–205× 差距主要是循环开销），仅作数值正确性对照 |
| **MTP 投机解码** | fp32 贪心 **8/8 逐 token 无损**；bf16 2/8 分歧定位为形状相关累加舍入（非图错误） | 定量拆解 **0.5B 单序列短输出是 MTP 非受益场景**；真正受益需 target compute-bound + head 高对齐 + batching/树状草稿 |

### 分布式服务层

| 模块 | 实测 |
| --- | --- |
| **PD 分离** | 解耦 PD 贪心输出与单机基线**逐 token 相等（无损）**；KV 经 Connector 握手+字节序列化收发逐元素一致 |
| **EP 宽专家并行** | 对单进程 MoE Frobenius 相对误差 **<0.02（实测 0.0）**；EPLB 把热点 rank 最大负载 180→100，利用率 1.21 |
| **分布式 KV 池** | 跨块写入 / 填充 / 定位正确；前缀共享 + CoW 正确；释放引用归零；GPU→CPU→SSD 分层 + LRU |
| **全局调度** | P 慢 D 快时速率匹配批次 (2,2,1,0)；第二请求命中 1 个前缀块、跳过计算 |

---

## 目录结构

```
deepseek-serving/
├─ README.md
├─ requirements.txt              # torch>=2.4
├─ config.py                     # ServingConfig（P/D 数、EP/DP、块大小、分层容量、SLO）
│
├─ node_backend/                 # 节点内算子（唯一一份，包内相对导入）
│  ├─ REUSE.md                   # 复用边界说明
│  └─ deepseek_infer/
│     ├─ fp8/                    # quantize（per-tensor / tokenwise / 1×128 block）+ FP8Linear 三路径
│     ├─ mla/                    # rope / attention / mla（prefill / decode_naive / decode_absorbed / forward_chunk）
│     ├─ moe/                    # grouped_gemm（dispatch / padded bmm / combine）+ moe（top-k + 共享专家）
│     ├─ mtp/                    # mtp_head + speculative（单链/树状草稿、贪心/拒绝采样验证）
│     ├─ model/                  # config / norms / model（DeepSeek Block / Model）
│     └─ engine/                 # kv_cache（latent）/ sampling / generator
│
├─ cluster/                      # 分布式服务层（本仓新增）
│  ├─ engine.py                  # DisaggregatedEngine：组装一切 + 单机基线 reference
│  ├─ comm/                      # transport（Loopback）+ kv_connector（握手 + KV 迁移）
│  ├─ kvpool/                    # block_pool（块表/引用/CoW）+ tiered_store（GPU→CPU→SSD/LRU）+ prefix_index
│  ├─ ep/                        # parallel_state / all_to_all（dispatch-local-combine）/ ep_moe / load_balancer
│  ├─ pd/                        # node_factory（同 seed 保证 PD 无损）/ prefill_engine / decode_engine
│  └─ scheduler/                 # request 生命周期 + global_scheduler（准入/速率匹配/前缀路由）
│
├─ benchmarks/                   # bench_kernels.py（FP8/MLA/MoE/E2E，落 JSON）
├─ tests/                        # 强校验测试，可直接 python 运行（无需 pytest）
│  ├─ 节点内核：test_fp8 / test_mla / test_moe / test_mtp / test_e2e
│  └─ 服务层：  test_block_pool / test_connector / test_ep / test_pd_e2e / test_scheduler
├─ examples/                     # demo_pd.py（PD 无损演示）/ demo_ep.py（宽 EP + 负载均衡）
├─ scripts/                      # env_check / wall_bench / qwen_mtp / eval_mtp / train_mtp / scan_outlier / run_tests.cmd / run_demos.cmd
├─ results/                      # benchmark JSON（bench_kernels / bench_mtp* / env_check）
└─ docs/
   └─ DeepSeek推理引擎图解.html   # 模块级图解（33 张卡片的合集）
```

---

## 快速开始

```bash
# 安装依赖
pip install torch>=2.4

# 一键跑全部测试（Windows）
scripts\run_tests.cmd

# 或手动跑（推荐先跑 CPU 可复现的服务层测试）
python tests\test_pd_e2e.py       # PD 逐 token 无损
python tests\test_block_pool.py    # 块池 / CoW / 前缀共享
python tests\test_connector.py     # KV 握手 + 字节序列化
python tests\test_ep.py           # EP 数值等价 + EPLB
python tests\test_scheduler.py    # 速率匹配 + 前缀复用

# 节点内核 GPU 测试（需 CUDA）
python tests\test_fp8.py
python tests\test_mla.py
python tests\test_moe.py
python tests\test_mtp.py

# 演示
python examples\demo_pd.py        # PD 分离：无损、KV 分层、传输字节数
python examples\demo_ep.py         # 宽 EP + 专家负载均衡

# Kernel benchmark（落 results/bench_kernels.json）
python benchmarks\bench_kernels.py
```

---

## 与生产系统的对应

| 本仓 | 生产对应 |
| --- | --- |
| Prefill/Decode 分离 + 控制面 | DistServe、Splitwise、Mooncake（PD disaggregation） |
| BlockPool + KVConnector + TieredStore | Mooncake KVCache pool / 全局 KV store；vLLM KV offload |
| EP all-to-all（dispatch / local compute / combine） | DeepEP、Tutel、NVIDIA Dynamo 的 EP 通信 |
| LoadBalancer（专家副本 / 均衡） | EPLB（Expert Parallel Load Balancer） |
| GlobalScheduler（速率匹配 / 前缀路由） | Dynamo scheduler、Mooncake Global Scheduler |

## 诚实边界（单卡消费级 GPU 的模拟口径）

本机为单张 8GB 消费卡，**无法实测真实多节点 RDMA / 宽 EP**：

- **已在 CPU/GPU 上验证**：全部数据结构、消息语义、PD 无损性、EP 数值等价性、调度与前缀复用、分层存储搬运与 LRU（SSD 为真实 `torch.save` 落盘）。
- **用 loopback 模拟**：网络 = 单进程消息队列（`LoopbackTransport`，可选字节序列化贴近真实编码）；多 rank = 单进程虚拟 rank。
- **留作扩展**：`Transport` / `ParallelState` / `TieredStore` 均为抽象接口，替换为 NCCL/RDMA（NIXL/Mooncake transfer engine）、每 rank 一个 worker、dispatch/compute/combine 的 CUDA stream 通信-计算重叠、远端 KV 按需拉取，即可从单进程模拟平滑走向多节点真机。
