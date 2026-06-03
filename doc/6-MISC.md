# 6. MISC

本组包含以下指令：

- `tfill`
- `tbcast`
- `tbcastRow`
- `tbcastCol`
- `tshift`
- `ttrans`
- `ttransRevRow`
- `ttransRevCol`
- `tsetmask`
- `tsetmaskLowTri`
- `tsetmaskUpTri`

## 1. 数据搬运与形状变换

### `tfill`

- 用一个 32 bit 标量值填满整个输出 tile
- 不做额外类型解释，最终按 `trd.dtype` 写入寄存器

### `tbcast`

- 取 `trs1[row][col]`
- 广播到输出 tile 的全部位置

### `tbcastRow`

- 对每一行 `i`，取 `trs1[i][col]`
- 广播到该行全部列

### `tbcastCol`

- 对每一列 `j`，取 `trs1[row][j]`
- 广播到该列全部行

### `tshift`

- 执行平移
- `row_offset > 0` 表示向下移
- `row_offset < 0` 表示向上移
- `col_offset > 0` 表示向右移
- `col_offset < 0` 表示向左移
- 越界源元素丢弃，新空位补 `0`

### `ttrans`

- 普通转置
- `result[i][j] = src[j][i]`

### `ttransRevRow`

- 上下翻转
- `result[i][j] = src[31 - i][j]`

### `ttransRevCol`

- 左右翻转
- `result[i][j] = src[i][31 - j]`

## 2. 掩码生成

以下三条指令输出的都是布尔 tile：

- 真写 `1`
- 假写 `0`

输出目标 `trd_m8` 只能是：

- 8 bit 宽普通寄存器
- `Mask`

### `tsetmask`

- 先根据 `(top, bottom, left, right)` 定义一个中心矩形有效区
- 有效区内为真，外部为假
- `invert = true` 时反转

矩形有效区判定：

```text
top <= i < 32 - bottom
left <= j < 32 - right
```

### `tsetmaskLowTri`

- 先限定在上面的中心矩形内
- 再取该矩形内部的下三角区域
- `diagonal = true` 时包含对角线
- `diagonal = false` 时只保留严格下三角
- `invert = true` 时整体反转

三角判断基于裁剪后局部坐标：

- `row_idx = i - top`
- `col_idx = j - left`

### `tsetmaskUpTri`

- 与 `tsetmaskLowTri` 类似
- 但选择的是上三角区域
- `diagonal = true` 时包含对角线
- `diagonal = false` 时只保留严格上三角

## 3. 参数合法性

三类掩码生成指令都要求：

- `top + bottom < 32`
- `left + right < 32`

否则会触发断言。
