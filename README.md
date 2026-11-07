# csg_conflichek

一个用于检查 PHITS 构造实体几何（CSG）中重叠（overlap）和未定义（undefined）区域的命令行工具。

## 功能与范围

- 解析 PHITS 的 `[Title]`、`[Cell]`、`[Surface]`、`[Transform]` 和 `set:` 语句。
- 支持 PHITS 表面、宏体、cell 布尔表达式、`#` 引用和内联 `trcl`。
- 在指定的笛卡尔范围内生成网格，逐点判断 cell 归属。
- 输出两类冲突：
  - `overlap`：同一格点被两个或更多 cell 同时包含。
  - `undefined`：同一格点没有被任何 cell 包含。
- 将同一 cell 集合的相邻冲突点合并为块，并输出块内代表点。
- 提供 `--verify` 稳定性检查，用加密网格比较冲突体积和块数变化。
- 仅支持 PHITS，不实现 MCNP 输入解析。

## 安装

需要 Python >= 3.13。

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip install -e .
```

`requirements.txt` 只包含运行时依赖。安装后会提供命令行入口 `csgc`。也可以直接从仓库根目录运行模块：

```powershell
& '.\.venv\Scripts\python.exe' -m csg_conflichek.cli -i foo.inp -o bar.csv
```

运行时依赖：

- `numpy`
- `scipy`

开发和测试依赖：

- `pytest`
- `ruff`

## 快速开始

```powershell
& '.\.venv\Scripts\csgc.exe' -i examples\simple_phits.inp -o output\simple.csv
& '.\.venv\Scripts\csgc.exe' -i examples\overlap_phits.inp -o output\overlap.csv --size 200
& '.\.venv\Scripts\csgc.exe' -i examples\undefined_phits.inp -o output\undefined.csv --bounds -3 3 -3 3 -3 3
```

带稳定性检查：

```powershell
& '.\.venv\Scripts\csgc.exe' -i examples\overlap_phits.inp -o output\overlap.csv --size 200 --verify
```

手动指定验证网格：

```powershell
& '.\.venv\Scripts\csgc.exe' -i foo.inp -o bar.csv --size 200 --verify --verify-size 300
```

## 命令行参数

| 参数 | 说明 |
|---|---|
| `-i, --input` | 必填，PHITS 输入文件 |
| `-o, --output` | 输出 CSV，默认 `overlap.csv` |
| `--size` | 网格尺寸，单值或 `NX,NY,NZ`，默认 `200` |
| `--bounds` | 显式范围：`XMIN XMAX YMIN YMAX ZMIN ZMAX` |
| `--margin` | 自动范围向外扩展的量，默认 `0` |
| `--workers` | 并行进程数，`0` 表示大网格自动选择，默认 `0` |
| `--verify` | 额外运行一次加密网格并报告稳定性 |
| `--verify-size` | 验证网格尺寸，默认将主网格每轴加倍 |
| `--no-merge` | 不合并相邻冲突点，输出每个冲突格点 |
| `--fail-on-conflict` | 发现冲突时返回退出码 1 |
| `-q, --quiet` | 不打印摘要 |
| `-V, --version` | 显示版本 |

## CSV 输出

表头固定为：

```csv
no,type,x,y,z,cell1,cell2
```

规则：

- `no` 从 1 开始连续编号。
- `undefined` 行的 `cell1`、`cell2` 为空。
- 两个 cell 的 `overlap` 分别写入 `cell1`、`cell2`。
- 三个及以上 cell 的 `overlap` 为降低输出成本只写一行：所有 cell id 用 `;` 连接后写入 `cell1`，`cell2` 为空。
- 坐标为物理坐标，来自块内最接近质心的真实格点，不是质心，也不是网格索引。

示例：

```csv
no,type,x,y,z,cell1,cell2
1,overlap,0.5,0.5,0,1,2
2,overlap,1,1,1,1;2;3,
3,undefined,0,0,0,,
```

## 项目结构

```text
csg_conflichek/
    __init__.py
    shapes.py               几何表面及 is_positive()
    macro.py                RPP/RCC/BOX/HEX 等宏体
    cell.py                 CSG 布尔表达式与 Cell
    transforms.py           PHITS 坐标变换
    geometry.py             Geometry/CellEntry 与采样
    set_syntax.py           PHITS set: 变量
    _geom_syntax_core.py    共享解析核心
    phits_geom_syntax.py    PHITS 入口
    bounds.py               笛卡尔范围估计
    conflict.py             网格扫描、冲突合并、CSV、稳定性检查
    cli.py                  csgc 命令行入口
tests/                      单元测试和 CLI 测试
examples/                   小型 PHITS 输入示例
COPYING                     GPLv3 完整文本
requirements.txt            运行时依赖
pyproject.toml              构建和工具配置
README.md                   本文件
```

## 工作原理

1. 解析 PHITS 输入，构建 cell、surface、transform 和 `set:` 变量。
2. 根据显式 `--bounds` 或有限尺寸表面估计笛卡尔范围。
3. 在范围内生成 `NX × NY × NZ` 网格，计算每个格点的 cell 归属。
4. 归属 cell 数大于 1 记为 `overlap`，等于 0 记为 `undefined`。
5. 按 `(type, 归属 cell id 集合)` 分组。
6. 同一组内，网格索引曼哈顿距离为 1（6 邻域共享面）的冲突点合并为块。
7. 每个块输出一个块内格点作为代表坐标。

性能相关实现：

- 冲突点的 cell 归属使用位掩码编码；超过 64 个 cell 时使用多个 `uint64` 字。
- 冲突密集的组在局部包围盒上调用 `scipy.ndimage.label` 做 6 邻域连通分量标记。
- 稀疏组使用内存更省的 BFS。
- 大网格按 z 轴切分，可使用多进程扫描；主进程负责合并和写 CSV。

## 精度与注意事项

- 本程序采用 6 邻域面连通，作为“正体积/面连通”的离散近似。
- 这不是点集拓扑连通：拓扑意义上点/棱接触也算连通，但本程序按不同块处理。
- 对固定的正厚度连接或正宽度间隙，网格足够细时可以分辨。
- 对任意薄的连接、临界相切或夹点，不存在统一的“足够细”阈值，可能漏检或误合并。
- 合并块数会随网格变化，不能作为精度指标；应比较冲突体积随网格的收敛情况。
- 默认 `200³` 网格会产生 800 万个格点；更高分辨率会线性增加运行时间和冲突点内存占用。
- 代表点保证在冲突块内部，但不等于几何质心。
- `overlap` 和 `undefined` 的检测精度主要受网格离散化影响，而不是浮点精度。
- 只有平面、圆柱、圆锥等无界表面的模型无法自动确定范围，需要使用 `--bounds`。
- 不支持 `u=`、`lat=`、`fill=` 等尚未实现的 cell 参数。
- 只支持数值型 `set:` 变量，不支持字符型变量。

## 性能与并行

- 11 cell、`200³`：约 0.4 秒。
- 100 cell、`200³`：约 8 秒。
- 100 cell、`300³`：自动多进程约 8 秒。
- 100 cell、`500³`：8 进程约 80 秒。
- `--workers 0` 时，网格点数达到约 2000 万才自动启用多进程；较小网格保持单进程以避免启动开销。
- 可手动指定进程数：

```powershell
& '.\.venv\Scripts\csgc.exe' -i foo.inp -o bar.csv --size 300 --workers 8
```

- `--verify` 会额外运行一次加密网格，默认每轴加倍。如果验证网格超过约 2 亿点，程序会要求显式提供 `--verify-size`。

## 测试与开发

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m ruff check .
& '.\.venv\Scripts\python.exe' -m ruff format --check .
```

当前测试套件包含 240 个用例。

## 许可证

本项目采用 GNU General Public License v3.0 or later（GPL-3.0-or-later）。完整文本见 [COPYING](COPYING)。

部分几何和解析模块移植自 `csg_draw`，同样遵循 GPL-3.0-or-later。

## 第三方依赖与署名

本项目不打包 numpy、scipy 等第三方包；它们由 `pip` 在用户环境中单独安装。以下为直接依赖的署名和许可证信息：

| 包 | 用途 | 版本 | 许可证 | 版权 |
|---|---|---|---|---|
| numpy | 运行时 | 2.4.4 | BSD-3-Clause | Copyright (c) 2005-2025, NumPy Developers |
| scipy | 运行时 | >=1.16（验证 1.18.1） | BSD-3-Clause | Copyright (c) 2001-2002 Enthought, Inc.; 2003, SciPy Developers |
| pytest | 测试 | 9.0.3 | MIT | Copyright (c) 2004 Holger Krekel and others |
| ruff | 代码检查/格式化 | 0.15.14 | MIT | Copyright (c) 2022 Charles Marsh |
| hatchling | 构建后端 | 构建时由 pip 解析 | MIT | Copyright (c) 2017-present Ofek Lev |

上游链接：

- numpy：<https://numpy.org/>，许可证：<https://github.com/numpy/numpy/blob/main/LICENSE.txt>
- scipy：<https://scipy.org/>，许可证：<https://github.com/scipy/scipy/blob/main/LICENSE.txt>
- pytest：<https://pytest.org/>，许可证：<https://github.com/pytest-dev/pytest/blob/main/LICENSE>
- ruff：<https://docs.astral.sh/ruff/>，许可证：<https://github.com/astral-sh/ruff/blob/main/LICENSE>
- hatchling：<https://hatch.pypa.io/>，许可证：<https://github.com/pypa/hatch/blob/master/LICENSE.txt>

说明：

- numpy 主许可证为 BSD-3-Clause；其发行包中还包含其他组件的额外许可证（0BSD、MIT、Zlib、CC0-1.0 等），完整信息见 numpy 上游 `LICENSE.txt`。
- 传递依赖（如 `packaging`、`pluggy`、`iniconfig`、`colorama`、`Pygments`、`pip`）由测试、构建或虚拟环境引入，未在本 README 中逐条列出完整许可证。
- 如果将来发布包含第三方包的独立环境、二进制或离线安装包，应同时附带对应依赖的完整许可证文本。
