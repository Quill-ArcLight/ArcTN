# ArcTN

[English](README.md) | **简体中文**

ArcTN 是一个用 Rust 编写、提供 Python 接口的张量网络库，用于搜索收缩路径、切片和数值收缩，也可接入 Quimb、Cotengra 和 opt_einsum。

[完整安装包](https://github.com/Quill-ArcLight/ArcTN/releases/tag/v1.0.1) · [中文文档](https://quantumquill.arclightquantum.com/docs/arctn/index.html)

<a id="algorithms-and-execution"></a>

## 主要功能

- 搜索收缩路径，提供 Light 和 Heavy 两种搜索预设。
- 化简张量网络，通过切片减小中间张量。
- 在 CPU 上执行实数和复数张量收缩。
- 保存、复用收缩路径，编译后重复执行计算。
- 接入其他 Python 张量网络工具；可选 MPI 切片并行执行。

<a id="python"></a>

## Python 使用

<a id="python-installation"></a>

### 安装

从 [ArcTN 1.0.1 下载页](https://github.com/Quill-ArcLight/ArcTN/releases/tag/v1.0.1)的 **Assets** 中，下载一份与系统和 Python 版本匹配的 `.whl`。完整包已包含 Light/Heavy 引擎，无需安装 Rust 或手动配置动态库。

| 系统与处理器 | 文件名中的平台标记 |
| --- | --- |
| macOS，Apple 芯片 | `macosx_..._arm64` |
| macOS，Intel 芯片 | `macosx_..._x86_64` |
| Windows，x86-64 | `win_amd64` |
| Linux，x86-64 | `manylinux_2_28_x86_64` |
| Linux，ARM64 | `manylinux_2_28_aarch64` |

每个平台均提供 CPython 3.12（`cp312`）和 3.13（`cp313`）安装包。用 `python --version` 查看版本。Linux 要求 glibc 2.28 或更新版本；macOS 最低版本见安装包文件名。

在下载的 wheel 所在目录打开终端，创建环境并安装：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=arctn --find-links=. arctn==1.0.1
```

Windows PowerShell 将第二行替换为 `.venv\Scripts\Activate.ps1`。NumPy 等依赖会自动安装，以下示例在同一环境中运行。

<a id="light-and-heavy-interface"></a>

### 搜索收缩路径

下面以三个矩阵相乘为例，输入形状依次为 `(2, 3)`、`(3, 4)` 和 `(4, 2)`，输出形状为 `(2, 2)`：

```python
from arctn import arctn_path

inputs = [["a", "b"], ["b", "c"], ["c", "d"]]
output = ["a", "d"]
size_dict = {"a": 2, "b": 3, "c": 4, "d": 2}

path = arctn_path(inputs, output, size_dict, preset="light", seed=0)
print(path)
```

`inputs` 列出各张量的索引，`output` 指定输出索引及顺序，`size_dict` 给出索引维度。返回的 `path` 是每一步收缩哪两个张量的列表；此调用只搜索路径，不需要张量数值。

**使用 Heavy，只需把 `preset="light"` 改成 `preset="heavy"`。** 两者使用相同的导入方式；不指定 `preset` 时默认使用 Heavy。

### 直接执行收缩

沿用上面的网络定义，按 `inputs` 的顺序提供浮点或复数数组：

```python
import numpy as np
from arctn import arctn_contract

arrays = [
    np.arange(6, dtype=float).reshape(2, 3),
    np.arange(12, dtype=float).reshape(3, 4),
    np.arange(8, dtype=float).reshape(4, 2),
]
result = arctn_contract(inputs, output, size_dict, arrays, preset="light", seed=0)
print(result)
```

`arctn_contract` 会搜索路径并执行计算，返回结果数组。上例等价于 `arrays[0] @ arrays[1] @ arrays[2]`。

根据需要选择入口，无需依次调用：

| 需求 | 函数 | 返回值 |
| --- | --- | --- |
| 只搜索路径 | `arctn_path` | 收缩路径 |
| 查看路径、计算量和切片方案 | `arctn_schedule` | 包含路径与指标的字典 |
| 搜索并执行收缩 | `arctn_contract` | 结果数组 |

### 常用设置

- `preset="light"` 或 `"heavy"`：选择搜索预设。
- `seed=0`：指定搜索的随机种子。
- `max_time=10.0`：设置搜索时间预算，单位秒，不是整个数值计算的强制超时。
- `read_write_weight=0`：只按 FLOPs 优化路径；默认也考虑读写成本。

`arctn_schedule` 和 `arctn_contract` 还支持 `target_size`，用于设置每个切片中单个中间张量的元素数上限，不是整个程序的内存上限。需要对输出索引切片时，同时设置 `allow_output_slicing=True`；执行后仍返回完整输出。

其他参数、路径复用和 Quimb 接入示例见 [Python 接口说明](pybind/README.md#interfaces)。MPI 的安装和使用见 [MPI 文档](docs/mpi.md)；目前仅支持内部索引切片。

<a id="rust"></a>

## Rust 使用

需要 Rust 1.82 或更新版本：

```sh
git clone https://github.com/Quill-ArcLight/ArcTN.git
cd ArcTN
cargo run --release --example contraction
```

<a id="source-and-availability"></a>
<a id="engine-library"></a>

本仓库公开独立算法、切片、数值执行和调用接口。Light/Heavy 引擎保持闭源，完整 wheel 已包含引擎；自行从源码构建时需[单独配置引擎](docs/engine-interface.md)。Python 源码安装见[构建说明](pybind/README.md#source-installation)。

<a id="license"></a>

## 许可证

本项目供非商业使用。公开源码适用 [Arclight 非商业源码可用许可证 1.0](LICENSE)，Light/Heavy 引擎适用单独的[引擎许可证](LICENSE-ARCTN-ENGINE.txt)。第三方依赖保留各自许可证，见[第三方声明](THIRD_PARTY_NOTICES.md)。
