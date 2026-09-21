# ArcTN

**简体中文** | [English](https://github.com/Quill-ArcLight/ArcTN/blob/main/README.en.md)

ArcTN 是用 Rust 编写的张量网络库，提供收缩序优化、切片和数值收缩，并通过 Python 接口接入 Quimb、Cotengra 和 opt_einsum。既可向其他工具提供收缩路径，也可直接执行张量网络收缩。

[中文文档](https://quantumquill.arclightquantum.com/docs/arctn/index.html)

<a id="algorithms-and-execution"></a>

## 主要功能

- 收缩序优化：贪心、随机贪心、子集动态规划、固定叶序动态规划和超图二分。
- 收缩树优化：子树重构、模拟退火和并行回火（parallel tempering）。
- 网络化简，以及从化简网络的收缩路径重建原网络的路径。
- 固定收缩路径的切片，以及允许局部调整收缩路径的动态切片。
- 实数和复数稠密张量的 CPU 收缩。
- 收缩路径和切片信息的保存、路径编译与重复执行，以及路径缓存。
- NumPy 和显式指定的外部数组后端；可选的 MPI 切片并行执行。

网络结构由各输入张量的索引（index）、输出索引和相应维度定义；数值执行时另行提供对应的张量数组。量子电路到张量网络的转换由 Quimb 等前端完成，ArcTN 接收转换后的张量网络。

<a id="source-and-availability"></a>

## 源码与 Light / Heavy

本仓库公开独立算法、网络与路径类型、切片、数值执行、Python 接口和命令行工具。公司有权授权的源码采用 **[Arclight 非商业源码可用许可证 1.0](LICENSE)**，禁止商用和闭源集成；这不是 OSI 标准开源许可。

**Light 和 Heavy 的实现闭源，本仓库提供调用接口。** 从本仓库源码构建的 Rust crate 和 Python wheel 不包含该引擎。没有引擎时，仍可使用 Rust 独立算法、CLI 的 `--method greedy` 和张量数值收缩；Python 可执行已有收缩路径及给定的切片方案，并进行网络化简。

通过 `arctn_path`、`arctn_schedule`、`arctn_plan` 等接口使用 Light/Heavy，需要安装包含引擎的完整 wheel，或单独配置兼容且已获授权的动态库。安装方法见 [Python 使用](#python)。

[CI](https://github.com/Quill-ArcLight/ArcTN/actions/workflows/test.yml) 在 Linux、macOS 和 Windows 上运行 Rust 与 CPython 3.13 测试，并在 Linux 上运行 Open MPI 多进程测试。

MPI 是可选的切片并行执行功能。`tnmpi` 将切片分配给各进程，按已保存的收缩路径执行计算，最后对各进程的结果求和。普通单机 Rust/Python 功能不需要 MPI。安装和运行要求见 [MPI 文档](docs/mpi.md)。

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

`RAYON_NUM_THREADS` 设置默认 Rayon 线程池的线程数，应用也可以使用自己的 Rayon 线程池。Cargo 的 `mt` feature 启用独立的矩阵乘法线程池；与切片并行同时使用时，应控制总线程数，避免超过可用 CPU 核数。

<a id="python"></a>

## Python 使用

<a id="python-installation"></a>

### 安装

安装包含 Light/Heavy 引擎的 ArcTN 1.0.0 完整 wheel，无需 Rust。先获取与操作系统、CPU 架构和 Python 版本匹配的安装包，在该 wheel 文件所在的目录打开终端，然后运行：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --find-links=. arctn==1.0.0
```

`arctn==1.0.0` 指定安装版本；`--find-links=.` 让 pip 同时在当前目录查找兼容的 wheel，不需要手动填写文件名。NumPy 等依赖会自动安装。

以上激活命令适用于 macOS 和 Linux；Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。具体支持的组合以发布的安装包为准。

从本仓库源码安装需要 Python 3.9 或更新版本、Rust 1.83 或更新版本，操作见 [Python 源码安装](pybind/README.md#source-installation)。源码安装不包含 Light/Heavy 动态库；调用 Light/Heavy 时需要[单独配置动态库](#engine-library)。

<a id="light-and-heavy-interface"></a>

### 调用 Light / Heavy

先按[安装步骤](#python-installation)安装包含 Light/Heavy 引擎的完整 wheel，再在同一个 Python 环境中运行下面的代码。使用完整 wheel 不需要克隆源码或安装 Rust，也不需要手动加载动态库。

**导入的是 `arctn_path` 函数，Light 和 Heavy 由 `preset` 参数选择，不是两个需要单独导入的 Python 模块。** 导入时，Python 接口会自动识别安装包中的引擎动态库；第一次调用搜索函数时再加载并运行它。下面的例子使用 Light：

```python
from arctn import arctn_path

inputs = [["a", "b"], ["b", "c"], ["c", "d"]]
output = ["a", "d"]
size_dict = {"a": 2, "b": 3, "c": 4, "d": 2}

path = arctn_path(
    inputs, output, size_dict,
    preset="light",
    seed=0,
)
print(path)
```

**使用 Heavy 时，只需把 `preset="light"` 改成 `preset="heavy"`，不需要重新安装或更换导入语句。** 未指定 `preset` 时默认使用 Heavy。

`inputs` 按张量顺序列出各自的索引，`output` 指定结果保留的索引及顺序，`size_dict` 给出各索引的维度。这个例子表示三个矩阵的乘积，输入形状分别为 `(2, 3)`、`(3, 4)` 和 `(4, 2)`，输出形状为 `(2, 2)`。

`arctn_path` 只搜索收缩序，不执行数值收缩，因此不需要矩阵中的数值。返回值是一组二元组，表示每一步收缩哪两个张量；默认使用 opt_einsum 的 linear path 格式，设置 `use_ssa=True` 可返回 SSA path。

`seed` 指定搜索使用的随机种子。两种模式使用相同的优化目标参数，默认是 `flops_weight=1, read_write_weight=64`；设置 `read_write_weight=0` 可仅优化 FLOPs。

如果还需要路径指标，改用 `arctn_schedule`。沿用上面的网络定义：

```python
from arctn import arctn_schedule

result = arctn_schedule(inputs, output, size_dict, preset="heavy", seed=0)
print("Path:", result["path"])
print("log10 FLOPs:", result["log10_flops"])
print("log2 largest intermediate:", result["log2_max_size"])
```

`arctn_schedule` 返回字典，包含最终路径、优化目标的权重、路径指标、可选的切片结果和收缩序优化耗时。上面的 `log10_flops` 是 FLOPs 的以 10 为底的对数，`log2_max_size` 是最大中间张量元素数的以 2 为底的对数。

按需要选择调用入口，不需要依次调用：

| 需求 | 从 `arctn` 导入 | 返回值 |
| --- | --- | --- |
| 只搜索收缩序 | `arctn_path` | 一条收缩路径 |
| 搜索并查看路径指标，或生成切片方案 | `arctn_schedule` | 包含路径和指标的字典 |
| 搜索后直接执行收缩 | `arctn_contract` | 结果数组；设置 `return_info=True` 时返回 `(result, info)` |

这三个入口都通过 `preset` 选择 Light 或 Heavy。`arctn_contract` 还需要按 `inputs` 的顺序传入 `arrays`，默认使用 Rust CPU 执行器。其他调用方式及 Quimb 接入见 [Python 接口说明](pybind/README.md#interfaces)。

`arctn_schedule` 和 `arctn_contract` 接受 `target_size`，限制每个切片中生成的单个中间张量的元素数，不是进程总内存上限。`max_time` 的单位是秒，由搜索过程检查，不会由操作系统强制终止进程。

### 执行已有收缩路径

如果已经有收缩路径，可直接编译并执行，不需要调用 Light/Heavy。以下是一个独立的矩阵乘法示例：

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

<a id="engine-library"></a>

### 单独配置 Light/Heavy 动态库

从 Rust 调用 Light/Heavy、使用 Python 源码安装，或需要指定其他兼容动态库时，在首次调用前设置动态库的绝对路径：

```sh
export ARCTN_ENGINE_LIBRARY=/absolute/path/to/libarctn_engine.so
```

macOS 使用对应的 `.dylib`；Windows 使用 `.dll`，PowerShell 的设置方式见 [动态库接口说明](docs/engine-interface.md)。显式设置的 `ARCTN_ENGINE_LIBRARY` 优先于 Python 包中附带的动态库。

动态库必须与操作系统、CPU 架构和 [ABI 版本](docs/engine-interface.md#c-abi-version-1) 匹配，并来自可信来源。没有可用动态库时，Light/Heavy 调用会报告安装错误，不影响独立算法或已有路径的执行。

<a id="license"></a>

## 许可证

公司代码适用 [Arclight 非商业源码可用许可证 1.0](LICENSE)。第三方依赖保留各自许可证。第三方声明及早期开发版本中保留的许可文本说明，见 [第三方及历史授权说明](THIRD_PARTY_NOTICES.md)。Light/Heavy 动态库单独许可，本许可证不授权其分发或使用。
