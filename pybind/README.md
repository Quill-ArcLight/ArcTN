# ArcTN Python 接口

Python 包提供收缩序优化、路径保存与加载、切片和数值收缩接口。

从本仓库构建 Python 包，请参阅[源码安装](#source-installation)。该方式不包含 Light/Heavy 引擎，
但可以执行已有收缩路径及给定的切片方案，并进行网络化简。
使用 Light/Heavy，需要安装包含引擎的完整 wheel，或单独配置兼容且已获授权的动态库。

<a id="complete-wheel-installation"></a>

## 安装完整 wheel

以下步骤适用于已获得完整 wheel 的用户，安装时不需要 Rust。
请选择与操作系统、CPU 架构和 Python 版本匹配的文件。
源码支持 Python 3.9 及以上版本；预编译 wheel 支持的 Python 版本以实际提供的文件为准。
将下面的路径替换为实际的 wheel 文件路径：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install /path/to/arctn-...whl
```

在 Windows PowerShell 中，请改用 `.venv\Scripts\Activate.ps1` 激活虚拟环境。
Linux、macOS 和 Windows 的 wheel 不能跨平台使用；具体可安装的组合以实际提供的文件为准。

完整 wheel 在 `arctn/_lib` 中包含编译后的 Light/Heavy 动态库。
导入 `arctn` 时会自动识别该动态库，调用 Light/Heavy 时加载，无需额外设置环境变量。
`arctn_path`、`arctn_schedule`、`arctn_plan` 等高层接口的参数和返回类型保持一致。

源码和动态库分别授权，说明见[源码与 Light / Heavy](../README.md#source-and-availability)。

NumPy 会作为依赖自动安装。如需与 Quimb 和 Cotengra 配合使用，请安装：

```sh
python -m pip install quimb cotengra opt_einsum
```

<a id="source-installation"></a>

## 从源码安装

从本仓库构建 Python 包需要 Python 3.9 及以上版本、Rust 1.83 及以上版本。
激活虚拟环境后，在仓库根目录运行：

```sh
python -m pip install "maturin>=1.9.3,<2"
python -m maturin develop --release --manifest-path pybind/Cargo.toml
```

从本仓库源码构建的包不包含 Light/Heavy 动态库，但仍可执行给定的收缩路径与切片方案，
并进行网络化简。如需使用 Light 或 Heavy，需要提供兼容且已获授权的动态库，
并在首次调用搜索接口前，将 `ARCTN_ENGINE_LIBRARY` 设置为该库的绝对路径。
显式设置此变量时，会优先使用指定的动态库。
平台支持与 ABI 要求详见[动态库接口说明](../docs/engine-interface.md)。

<a id="interfaces"></a>

## 接口

| 接口 | 用途 |
| --- | --- |
| `arctn_path` | 使用 Light 或 Heavy 搜索并返回收缩路径。 |
| `arctn_schedule` | 返回最终路径、优化目标、指标，以及可选的切片索引。 |
| `arctn_plan` | 返回 `ArcTNExecutionPlan`，保存网络结构、收缩路径与切片信息。 |
| `arctn_tree` | 返回 Cotengra 的 `ContractionTree`。 |
| `arctn_contract` | 优化收缩路径并执行数值收缩。 |
| `ArcTNOptimizer` | 为 opt_einsum、Cotengra 和 Quimb 提供路径或收缩树搜索接口。 |
| `ArcTNCompiledContraction` | 编译给定路径，以便重复执行收缩。 |
| `ArcTNExecutionPlan` | 保存、加载和校验网络与收缩路径，并按其中的切片信息执行收缩。 |
| `arctn_simplify` | 返回张量网络的结构化简结果。 |

使用 Light 或 Heavy 搜索时，接口会调用随包提供的动态库，或由 `ARCTN_ENGINE_LIBRARY`
显式指定的动态库。按给定路径执行收缩和化简张量网络不需要该动态库。

<a id="objective-and-slicing"></a>

## 优化目标与切片

默认优化目标的权重为 `flops_weight=1, read_write_weight=64`。
设置 `read_write_weight=0` 可仅优化浮点运算量（FLOPs）；
设置 `flops_weight=0` 可仅优化读写量。

`target_size` 是每个切片中生成的单个中间张量的元素数量上限，必须为正整数。
`slicing_mode="fixed"` 保持已选定的收缩路径；`"dynamic"` 还允许局部调整收缩路径。
两种模式均不对输出索引切片。请求切片时，两种模式都需要提供 `target_size`；
未提供 `target_size` 时，不能使用 `"dynamic"`。

## Quimb

将 `ArcTNOptimizer(preset="heavy")` 传给 Quimb 的 `optimize` 参数即可使用。
Quimb 负责构建张量网络并控制预处理；ArcTN 提供收缩路径，
也可通过 `search()` 返回包含切片索引的 Cotengra 收缩树。
实际的数值收缩仍由调用方选择的执行器完成。

如需直接执行数值收缩，`arctn_contract` 支持通过 `backend="native"`
使用 Rust CPU 执行器，也支持显式指定外部数组后端。
输入数组必须已属于所选后端；ArcTN 不会自动将数组传输到 GPU。

[中文文档](https://quantumquill.arclightquantum.com/docs/arctn/index.html)
