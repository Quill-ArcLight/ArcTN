# MPI 执行

`tnmpi` 执行已经保存的收缩路径和切片集合。rank 0 读取并校验收缩序文件，再把网络结构、
路径和切片集合广播给所有进程；各进程计算不同切片，最后通过 MPI Allreduce
逐元素求和。路径搜索和切片选择在保存文件前完成，MPI 执行不会改变这些决定。

## 构建

先安装 MPI。在 Ubuntu 上运行：

```bash
sudo apt install libopenmpi-dev openmpi-bin
```

在 macOS 上运行：

```bash
brew install open-mpi
```

然后在仓库根目录构建：

```bash
cargo build --locked --release --features mpi --bin tnpath --bin tnmpi
```

`mpi` 是可选 feature；普通构建不会编译 `tnmpi`。

## 保存收缩序文件

先用仓库中的小网络生成包含路径与切片的收缩序文件：

```bash
target/release/tnpath tests/fixtures/demo_tiny6.net.json \
  --method greedy \
  --target-size 16 \
  --save-path plan.json
```

`--save-path` 的文件名须尚不存在。`--target-size` 是规划参数，限制每个切片中
产生的单个中间张量元素数，不等于进程实际内存占用。正式计算应根据网络和可用内存
在规划阶段选择目标。

已有 Python 网络时，也可以保存 `arctn_plan` 的结果：

```python
from arctn import arctn_plan

plan = arctn_plan(inputs, output, size_dict, preset="heavy", target_size=1048576)
plan.save("plan.json")
```

Python 规划的安装说明见 [Python 接口](../pybind/README.md)。两种方法都写入
`arctn-execution-plan` v2 JSON，包含网络结构、SSA 路径和明确的切片集合；
空切片集合表示执行完整网络。收缩序文件不保存输入张量的数值。

## 执行保存的路径与切片

先使用按固定种子生成的演示数据检查 MPI 环境：

```bash
mpirun -n 2 target/release/tnmpi --load-path plan.json --seed 1 --check
```

正式输入通过 `--data` 指定：

```bash
mpirun -n 8 target/release/tnmpi --load-path plan.json --data tensors.bin
```

`--dtype` 选择输入和收缩计算的数据类型，默认 `f64`：

| `--dtype` | 每个元素的表示 | 每个元素的字节数 |
| --- | --- | --- |
| `f32` | 32 位实数 | 4 |
| `f64` | 64 位实数 | 8 |
| `complex64` | 实部和虚部各为一个 `f32` | 8 |
| `complex128` | 实部和虚部各为一个 `f64` | 16 |

数据文件按收缩序文件中输入张量的顺序连续存放，各张量使用行优先、小端格式，
不带文件头；复数的每个元素先存实部，再存虚部。所有输入使用同一种数据类型，
且必须与 `--dtype` 一致。`complex128` 表示总共 128 位的复数，不是 128 位实数。
例如，读取 `complex128` 输入：

```bash
mpirun -n 8 target/release/tnmpi \
  --load-path plan.json --data tensors-complex128.bin --dtype complex128
```

每个进程都必须能访问相同的数据文件；未给 `--data` 时，每个进程按相同种子
生成所选类型的一致演示输入。`--seed` 只控制这些演示输入。
最终 Allreduce 使用与所选类型对应的 MPI 数据类型逐元素求和。

v2 收缩序文件已经包含网络，无需再给 `--net`。读取旧版 v1 文件时必须同时提供
`--net network.json`；对 v2 使用这个选项则会额外核对网络是否匹配。
当前执行器不接受无版本的旧路径文件。

rank 0 输出 JSON 统计。`load_wall_seconds_max` 包含文件读取、校验、广播和输入张量
准备；`execution_wall_seconds_max` 包含切片收缩与最终 Allreduce，不包含输入准备
或 `--check` 对照。两者都取各进程墙时的最大值，单位为秒。

Rayon 默认每进程 1 个线程，可用 `RAYON_NUM_THREADS` 指定；启用 `mt` feature 后，
矩阵乘法线程数由 `MATMUL_NUM_THREADS` 单独控制。

## 运行条件

- MPI 按切片分配工作；每片的收缩和矩阵计算在所属进程内完成。
- 每个进程保存完整输入张量。Allreduce 把最终输出交给每个进程，因此输入和输出
  内存不会随进程数增加而自动按比例缩小。
- 比较不同进程数时，复用同一收缩序文件和同一输入。切片数少于进程数时会有进程没有切片
  可算；调整并行工作量需要重新规划并保存新文件。
- 浮点求和顺序可能随 MPI 实现和进程数变化，不保证结果逐位相同。`--check` 只在
  输出和切片数量较小时执行单机对照；更大的问题会跳过对照。
- 作业时间上限通过 MPI 启动器或集群调度器设置。
