# ArcTN

**简体中文** | [English](https://github.com/Quill-ArcLight/ArcTN/blob/main/README.en.md)

ArcTN 是用 Rust 编写的张量网络库，提供收缩序优化、切片和数值收缩，并通过 Python 接口接入 Quimb、Cotengra 和 opt_einsum。既可向其他工具提供收缩路径，也可直接执行张量网络收缩。

[中文文档](https://quantumquill.arclightquantum.com/docs/arctn/index.html)

<a id="algorithms-and-execution"></a>

## 主要功能

- 收缩序优化：贪心、随机贪心、子集动态规划、固定叶序动态规划和超图二分。
- 收缩树优化：子树重构、模拟退火和并行回火（parallel tempering）。
- 网络化简，以及从化简网络的收缩路径重建原网络的路径。
- 固定收缩路径的切片，以及允许局部改进路径的动态切片。
- 实数和复数稠密张量的 CPU 收缩，收缩路径编译、保存与重复执行，以及路径缓存。
- NumPy 和显式指定的外部数组后端；可选的 MPI 切片并行执行。

输入由各张量的 index、输出 index 和维度组成，数值执行时再提供对应数组。量子电路到张量网络的转换由 Quimb 等前端完成，ArcTN 接收转换后的张量网络。

<a id="source-and-availability"></a>

## 源码与 Light / Heavy

本仓库公开独立算法、网络与路径类型、切片、数值执行、Python 接口和命令行工具。公司有权授权的源码采用 **[Arclight 非商业源码可用许可证 1.0](LICENSE)**，禁止商用和闭源集成；这不是 OSI 标准开源许可。

**Light 和 Heavy 的实现闭源，本仓库保留调用接口。** 从本仓库源码构建的 Rust crate 和 Python wheel 不包含该引擎。Rust 独立算法、CLI 的 `--method greedy` 和张量数值收缩可直接使用；Python 可执行已有收缩路径、切片和网络化简。使用 `arctn_path`、`arctn_schedule`、`arctn_plan` 等接口调用 Light/Heavy，需要另行提供兼容且已获授权的动态库。

[CI](https://github.com/Quill-ArcLight/ArcTN/actions/workflows/test.yml) 在 Linux、macOS 和 Windows 上运行 Rust 与 CPython 3.13 测试，并在 Linux 上运行 Open MPI 多进程测试。MPI 是可选的切片并行执行功能，通过 `tnmpi` 按已保存的收缩路径计算分配给各进程的切片，并汇总结果。普通单机 Rust/Python 功能不需要 MPI。安装和运行要求见 [MPI 文档](docs/mpi.md)。

<a id="rust"></a>

## Rust 使用

源码构建需要 Rust 1.82 或更新版本：

```sh
git clone https://github.com/Quill-ArcLight/ArcTN.git
cd ArcTN
cargo build --release
cargo test
cargo run --example contraction
```

以下随机贪心示例不依赖 Light/Heavy 动态库：

```rust
use arctn::{random_greedy, TensorNetwork};

fn main() -> Result<(), String> {
    let net = TensorNetwork {
        name: "matrix_product".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        size_dict: [(0, 2), (1, 3), (2, 2)].into_iter().collect(),
    };
    let (path, stats) = random_greedy(&net, 16, 0)?;
    println!("path: {path:?}; log10 FLOPs: {}", stats.log10_flops);
    Ok(())
}
```

`RAYON_NUM_THREADS` 设置默认 Rayon 线程池的线程数，应用也可以使用自己的 Rayon 线程池。Cargo 的 `mt` feature 启用独立的矩阵乘法线程池；与切片并行同时使用时，应控制总线程数，避免超过可用 CPU 核数。

<a id="python"></a>

## Python 使用

### 安装

从本仓库源码安装需要 Python 3.9 或更新版本、Rust 1.83 或更新版本，操作见 [Python 源码安装](pybind/README.md#source-installation)。源码安装不附带 Light/Heavy 动态库。

如果已获得含引擎的完整 wheel，可以直接安装，无需 Rust。请选择与操作系统、CPU 架构和 Python 版本匹配的安装包，将下面的路径替换为实际文件路径：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install /path/to/arctn-...whl
```

以上激活命令适用于 macOS 和 Linux；Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。不同系统、CPU 架构和 Python 版本使用不同 wheel，具体以提供的安装包为准。

### 执行已有收缩路径

以下示例编译并执行矩阵乘法的收缩路径，不需要 Light/Heavy：

```python
import numpy as np
from arctn import ArcTNCompiledContraction

inputs, output = [["a", "b"], ["b", "c"]], ["a", "c"]
sizes = {"a": 2, "b": 3, "c": 2}
compiled = ArcTNCompiledContraction.compile(inputs, output, sizes, ssa_path=[(0, 1)])
a = np.arange(6, dtype=np.float64).reshape(2, 3)
b = np.arange(6, dtype=np.float64).reshape(3, 2)
np.testing.assert_allclose(compiled([a, b]), a @ b)
```

<a id="light-and-heavy-interface"></a>

### 调用 Light / Heavy

完整安装包会自动加载其中的动态库，不需要额外配置。沿用上面的网络定义：

```python
from arctn import arctn_schedule

result = arctn_schedule(inputs, output, sizes, preset="heavy", seed=0)
print(result["path"], result["log10_flops"])
```

使用 Light 时，将 `preset` 改为 `"light"`。返回信息包含最终路径、目标权重、路径指标、可选切片结果和收缩序优化耗时。其他调用方式及 Quimb 接入见 [Python 接口说明](pybind/README.md)。

`target_size` 限制每个 slice 中生成的单个中间张量的元素数，不是进程总内存上限。`max_time` 由搜索过程检查并退出，不是操作系统强制终止进程的超时限制。

### 单独配置 Light/Heavy 动态库

Rust 调用、Python 源码安装，或需要指定其他兼容动态库时，在首次调用 Light/Heavy 前设置绝对路径：

```sh
export ARCTN_ENGINE_LIBRARY=/absolute/path/to/libarctn_engine.so
```

macOS 使用对应的 `.dylib`；Windows 使用 `.dll`，PowerShell 的设置方式见 [动态库接口说明](docs/engine-interface.md)。显式设置的 `ARCTN_ENGINE_LIBRARY` 优先于 Python 包中附带的动态库。

动态库必须与操作系统、CPU 架构和 [ABI 版本](docs/engine-interface.md#c-abi-version-1) 匹配，并来自可信来源。没有可用动态库时，Light/Heavy 调用会报告安装错误，不影响独立算法或已有路径的执行。

<a id="license"></a>

## 许可证

公司代码适用 [Arclight 非商业源码可用许可证 1.0](LICENSE)。第三方依赖保留各自许可证。第三方声明及早期开发版本中保留的许可文本说明，见 [第三方及历史授权说明](THIRD_PARTY_NOTICES.md)。Light/Heavy 动态库单独许可，本许可证不授权其分发或使用。
