# 2. GEMM 指令

本组包含以下指令：

- `tgemm`
- `taccget`

## 1. 逻辑累加寄存器 `TACC`

GEMM 维护了 4 个架构可见的逻辑累加寄存器 `TACC[0..3]`，每个都是一个 `32 x 32` 高精度累加器。

- `tgemm` 负责把乘加结果累加进指定的 `TACC[idx]`
- `taccget` 负责把指定的 `TACC[idx]` 与一个输入 tile 相加后，按目标类型写回寄存器

## 2. `tgemm`

```cpp
void tgemm(uint8_t tacc_idx, const TOPRand &trs1z, const TOPRand &trs2z, bool trans1, bool trans2, bool accum);
```

语义：

- 先根据 `trans1/trans2` 决定是否转置两个输入矩阵
- 然后执行 `32 x 32 x 32` 的矩阵乘
- 结果逐元素累加到 `tacc_idx` 选择的逻辑累加寄存器

等价数学形式：

```text
TACC[tacc_idx] = (accum ? TACC[tacc_idx] : 0) + op(A) x op(B)
```

其中：

- `op(A)` 表示 `trs1z` 原矩阵或其转置
- `op(B)` 表示 `trs2z` 原矩阵或其转置

类型约束：

- `trs1z.dtype == trs2z.dtype`
- 当前实现支持 `SINT8 / UINT8 / F8E4M3 / F8E5M2 / SINT16 / UINT16 / F16`
- `0` 寄存器可作为输入，此时对应矩阵视为全 0

`accum` 含义：

- `false`：先把 `TACC` 清零，再做本次 GEMM
- `true`：保留上次 `TACC` 内容，继续累加

`tacc_idx` 含义：

- 选择 4 个架构可见逻辑累加寄存器中的一个
- 当前合法取值为 `0 / 1 / 2 / 3`

## 3. `taccget`

```cpp
void taccget(uint8_t tacc_idx, const TOPRand &trdt, const TOPRand &trs1zt, TRM rm, bool sat);
```

语义：

- 读取 `tacc_idx` 选择的 `TACC[tacc_idx]`
- 逐元素执行 `TACC[tacc_idx][i][j] + trs1zt[i][j]`
- 按 `rm` 舍入
- 按 `sat` 控制是否饱和
- 最终转换成 `trdt.dtype` 后写入 `trdt`

支持的输出类型：

- `SINT32`
- `UINT32`
- `F32`
- `F16`
- `SINT16`
- `UINT16`

当前实现要求：

- `trdt.dtype == trs1zt.dtype`
- `trdt` 可为普通寄存器或 `Temp`
- `trs1zt` 可为普通寄存器、`0` 寄存器或 `Temp`

## 4. 舍入与饱和

- `rm` 支持 `RNE/RTZ/RDN/RUP`
- `sat = true` 时，转换到目标类型会启用饱和行为
- `sat = false` 时，采用默认转换行为

## 5. 使用模式

典型流程：

1. 用一次或多次 `tgemm(..., accum, tacc_idx)` 把结果累加到指定的 `TACC[tacc_idx]`
2. 再用 `taccget(..., tacc_idx)` 把该 `TACC[tacc_idx]` 取出并落到可见寄存器

这让 4 个逻辑 `TACC` 都可以独立承担 GEMM 的中间高精度累加角色。
