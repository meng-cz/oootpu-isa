#pragma once

#include <cstdint>
#include <array>

namespace oootpu {

constexpr uint32_t TILE_SIZE = 32;

using TRegIdx = uint16_t;

/**
 * 获取Tensor逻辑寄存器的数量
 */
TRegIdx getTRegNum();

// 支持数据类型定义

enum class TDType : uint8_t {
    SINT8 = 0b000000,
    UINT8 = 0b000001,
    F8E4M3 = 0b000010,
    F8E5M2 = 0b000011,
    SINT16 = 0b010000,
    UINT16 = 0b010001,
    F16 = 0b010010,
    SINT32 = 0b100000,
    UINT32 = 0b100001,
    F32 = 0b100010,
};

inline constexpr uint32_t dtypeWidth(TDType dtype) {
    switch (dtype) {
        case TDType::SINT8:
        case TDType::UINT8:
        case TDType::F8E4M3:
        case TDType::F8E5M2:
            return 8;
        case TDType::SINT16:
        case TDType::UINT16:
        case TDType::F16:
            return 16;
        case TDType::SINT32:
        case TDType::UINT32:
        case TDType::F32:
            return 32;
    }
    return 0;
}

inline constexpr bool dtypeIsInt(TDType dtype) {
    switch (dtype) {
        case TDType::SINT8:
        case TDType::UINT8:
        case TDType::SINT16:
        case TDType::UINT16:
        case TDType::SINT32:
        case TDType::UINT32:
            return true;
        default:
            return false;
    }
}

inline constexpr bool dtypeIsSignedInt(TDType dtype) {
    switch (dtype) {
        case TDType::SINT8:
        case TDType::SINT16:
        case TDType::SINT32:
            return true;
        default:
            return false;
    }
}

// Tensor 操作数定义

enum class TOPRandType : uint8_t {
    NORM = 0, // 普通寄存器
    TEMP = 1, // 临时寄存器（指令内部使用，无法直接访问）
    ZERO = 2, // 0寄存器（读取恒为0，无法写入）
    M1 = 3, // Mask寄存器1
    M2 = 4, // Mask寄存器2
    M3 = 5, // Mask寄存器3
};

struct TOPRand {
    TRegIdx index;
    TDType dtype;
    TOPRandType type;

    constexpr TOPRand(TRegIdx idx) : index(idx), dtype(TDType::SINT8), type(TOPRandType::NORM) {}
    constexpr TOPRand(TRegIdx idx, TDType dt) : index(idx), dtype(dt), type(TOPRandType::NORM) {}
    constexpr TOPRand(TRegIdx idx, TDType dt, TOPRandType t) : index(idx), dtype(dt), type(t) {}
};

inline constexpr bool isMaskType(TOPRandType type) {
    return type >= TOPRandType::M1 && type <= TOPRandType::M3;
}

inline constexpr bool isMaskOperand(const TOPRand &t) {
    return isMaskType(t.type);
}

constexpr TOPRand TTEMP = {0, TDType::SINT8, TOPRandType::TEMP};
constexpr TOPRand TZERO = {0, TDType::SINT8, TOPRandType::ZERO};
constexpr TOPRand TMASK1 = {0, TDType::SINT8, TOPRandType::M1};
constexpr TOPRand TMASK2 = {0, TDType::SINT8, TOPRandType::M2};
constexpr TOPRand TMASK3 = {0, TDType::SINT8, TOPRandType::M3};

constexpr TOPRand S8(const TOPRand &op) { return TOPRand(op.index, TDType::SINT8, op.type); }
constexpr TOPRand U8(const TOPRand &op) { return TOPRand(op.index, TDType::UINT8, op.type); }
constexpr TOPRand F8E4M3(const TOPRand &op) { return TOPRand(op.index, TDType::F8E4M3, op.type); }
constexpr TOPRand F8E5M2(const TOPRand &op) { return TOPRand(op.index, TDType::F8E5M2, op.type); }
constexpr TOPRand S16(const TOPRand &op) { return TOPRand(op.index, TDType::SINT16, op.type); }
constexpr TOPRand U16(const TOPRand &op) { return TOPRand(op.index, TDType::UINT16, op.type); }
constexpr TOPRand F16(const TOPRand &op) { return TOPRand(op.index, TDType::F16, op.type); }
constexpr TOPRand S32(const TOPRand &op) { return TOPRand(op.index, TDType::SINT32, op.type); }
constexpr TOPRand U32(const TOPRand &op) { return TOPRand(op.index, TDType::UINT32, op.type); }
constexpr TOPRand F32(const TOPRand &op) { return TOPRand(op.index, TDType::F32, op.type); }

constexpr TOPRand S8(TRegIdx idx) { return TOPRand(idx, TDType::SINT8, TOPRandType::NORM); }
constexpr TOPRand U8(TRegIdx idx) { return TOPRand(idx, TDType::UINT8, TOPRandType::NORM); }
constexpr TOPRand F8E4M3(TRegIdx idx) { return TOPRand(idx, TDType::F8E4M3, TOPRandType::NORM); }
constexpr TOPRand F8E5M2(TRegIdx idx) { return TOPRand(idx, TDType::F8E5M2, TOPRandType::NORM); }
constexpr TOPRand S16(TRegIdx idx) { return TOPRand(idx, TDType::SINT16, TOPRandType::NORM); }
constexpr TOPRand U16(TRegIdx idx) { return TOPRand(idx, TDType::UINT16, TOPRandType::NORM); }
constexpr TOPRand F16(TRegIdx idx) { return TOPRand(idx, TDType::F16, TOPRandType::NORM); }
constexpr TOPRand S32(TRegIdx idx) { return TOPRand(idx, TDType::SINT32, TOPRandType::NORM); }
constexpr TOPRand U32(TRegIdx idx) { return TOPRand(idx, TDType::UINT32, TOPRandType::NORM); }
constexpr TOPRand F32(TRegIdx idx) { return TOPRand(idx, TDType::F32, TOPRandType::NORM); }

constexpr TOPRand operator+(const TOPRand &t, uint16_t offset) {
    return TOPRand(t.index + offset, t.dtype, t.type);
}

inline constexpr bool checkRegNorm(const TOPRand &t) {
    return t.type == TOPRandType::NORM;
}
inline constexpr bool checkRegZ(const TOPRand &t) {
    return t.type == TOPRandType::ZERO || t.type == TOPRandType::NORM;
}
inline constexpr bool checkRegZI(const TOPRand &t) {
    return checkRegZ(t) && dtypeIsInt(t.dtype);
}
inline constexpr bool checkRegT(const TOPRand &t) {
    return t.type == TOPRandType::TEMP || t.type == TOPRandType::NORM;
}
inline constexpr bool checkRegZT(const TOPRand &t) {
    return t.type == TOPRandType::TEMP || t.type == TOPRandType::NORM || t.type == TOPRandType::ZERO;
}
inline constexpr bool checkRegTM8(const TOPRand &t) {
    return (t.type == TOPRandType::TEMP || t.type == TOPRandType::NORM || (t.type >= TOPRandType::M1 && t.type <= TOPRandType::M3))
        && dtypeWidth(t.dtype) == 8;
}
inline constexpr bool checkRegZTM8(const TOPRand &t) {
    return (t.type == TOPRandType::TEMP || t.type == TOPRandType::NORM || t.type == TOPRandType::ZERO || (t.type >= TOPRandType::M1 && t.type <= TOPRandType::M3))
        && dtypeWidth(t.dtype) == 8;
}
inline constexpr bool checkRegZM8(const TOPRand &t) {
    return (t.type == TOPRandType::NORM || t.type == TOPRandType::ZERO || (t.type >= TOPRandType::M1 && t.type <= TOPRandType::M3))
        && dtypeWidth(t.dtype) == 8;
}
inline constexpr bool checkRegM8(const TOPRand &t) {
    return (t.type == TOPRandType::NORM || (t.type >= TOPRandType::M1 && t.type <= TOPRandType::M3))
        && dtypeWidth(t.dtype) == 8;
}

// 1. 访存指令

using TMemAddr = int64_t;

/**
 * 从连续内存加载一个tile到Tensor寄存器
 * @param trd 目标Tensor寄存器（仅普通寄存器）
 * @param baseaddr 内存基地址（对齐到tile行大小）
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trd）
 */
void tload(const TOPRand &trd, TMemAddr baseaddr, uint8_t mask = 0);

/**
 * 向连续内存存储一个tile寄存器的内容
 * @param trs1 源Tensor寄存器（仅普通寄存器和0寄存器）
 * @param baseaddr 内存基地址（对齐到tile行大小）
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 */
void tstore(const TOPRand &trs1z, TMemAddr baseaddr, uint8_t mask = 0);

/**
 * 从内存以行主布局加载一个tile到Tensor寄存器
 * @param trd 目标Tensor寄存器（仅普通寄存器）
 * @param baseaddr 内存基地址（对齐到tile行大小）
 * @param rowstride 行间距（以元素为单位）
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trd）
 */
void tloadRow(const TOPRand &trd, TMemAddr baseaddr, uint32_t rowstride, uint8_t mask = 0);

/**
 * 向内存以行主布局存储一个tile寄存器的内容
 * @param trs1 源Tensor寄存器（仅普通寄存器和0寄存器）
 * @param baseaddr 内存基地址（对齐到tile行大小）
 * @param rowstride 行间距（以元素为单位）
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 */
void tstoreRow(const TOPRand &trs1z, TMemAddr baseaddr, uint32_t rowstride, uint8_t mask = 0);

/**
 * 从内存按行选择偏移加载一个tile到Tensor寄存器，行元素偏移由trs2的第rowsel行指定
 * @param trd 目标Tensor寄存器（仅普通寄存器）
 * @param trs2 行元素偏移寄存器（仅整数类型普通寄存器）
 * @param baseaddr 内存基地址（对齐到tile行大小）
 * @param rowsel 行选择器，指定trs2中哪一行的元素偏移被使用
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs2）
 */
void tgatherRow(const TOPRand &trd, const TOPRand &trs2_i, TMemAddr baseaddr, uint8_t rowsel, uint8_t mask = 0);

/**
 * 向内存按行选择偏移存储一个tile寄存器的内容，行元素偏移由trs2的第rowsel行指定
 * @param trs1 源Tensor寄存器（仅普通寄存器和0寄存器）
 * @param trs2 行元素偏移寄存器（仅整数类型普通寄存器）
 * @param baseaddr 内存基地址（对齐到tile行大小）
 * @param rowsel 行选择器，指定trs2中哪一行的元素偏移被使用
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs2）
 */
void tscatterRow(const TOPRand &trs1z, const TOPRand &trs2_i, TMemAddr baseaddr, uint8_t rowsel, uint8_t mask = 0);


enum class Im2colLayout : uint8_t {
    K1S1P0 = 0, // kernel=1 stride=1 padding=0
    K3S1P1 = 1, // kernel=3 stride=1 padding=1
    K3S2P1 = 2, // kernel=3 stride=2 padding=1
    K5S1P2 = 3, // kernel=5 stride=1 padding=2
    K7S2P3 = 4, // kernel=7 stride=2 padding=3
};

/**
 * 配置一个卷积或矩阵乘法指令的内存访问参数，生成一个描述符寄存器供后续指令使用
 * @note im2col逻辑输出布局：N*OH*OW 行，KH*KW*C 列
 * @param layout 卷积im2col的布局类型
 * @param baseaddr 内存基地址（对齐到tile行大小）
 * @param N 批大小
 * @param H 输入特征图高度
 * @param W 输入特征图宽度
 * @param C 输入特征图通道数
 */
void tim2cfgNHWC(Im2colLayout layout, TMemAddr baseaddr, uint16_t N, uint16_t H, uint16_t W, uint16_t C);

/**
 * 使用tim2cfg指令配置的描述符将内存中的数据根据im2col算法加载到Tensor寄存器
 * @note im2col逻辑输出布局：N*OH*OW 行，KH*KW*C 列
 * @param trd 目标Tensor寄存器（仅普通寄存器）
 * @param subtile_row 加载的COL矩阵中的行子tile索引
 * @param subtile_col 加载的COL矩阵中的列子tile索引
 */
void tim2col(const TOPRand &trd, uint16_t subtile_row, uint16_t subtile_col);

// 2. GEMM 指令

enum class TRM : uint8_t {
    RNE = 0b00, // Round to Nearest, ties to Even
    RTZ = 0b01, // Round towards Zero
    RDN = 0b10, // Round Down (towards -inf)
    RUP = 0b11, // Round Up (towards +inf)
};

/**
 * 执行矩阵乘法，trs1和trs2分别作为输入矩阵，结果累加到指定逻辑TACC寄存器
 * @note 仅支持F8,I8,FP16数据类型的精确累加
 * @param tacc_idx 逻辑累加寄存器编号，取值范围为0~3
 * @param trs1z 输入矩阵1（仅普通寄存器和0寄存器）
 * @param trs2z 输入矩阵2（仅普通寄存器和0寄存器）
 * @param trans1 输入矩阵1是否转置
 * @param trans2 输入矩阵2是否转置
 * @param accum 是否跨指令累加到选中的TACC逻辑寄存器（否则覆盖）
 */
void tgemm(uint8_t tacc_idx, const TOPRand &trs1z, const TOPRand &trs2z, bool trans1, bool trans2, bool accum);

/**
 * 将指定逻辑TACC寄存器的内容与输入Tensor寄存器累加并输出到一个Tensor寄存器
 * @note 仅支持F32, I32数据类型
 * @param tacc_idx 逻辑累加寄存器编号，取值范围为0~3
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（普通寄存器、0寄存器和Temp寄存器）
 * @param rm 结果舍入模式
 * @param sat 是否对结果进行饱和截断
 */
void taccget(uint8_t tacc_idx, const TOPRand &trdt, const TOPRand &trs1zt, TRM rm, bool sat);


// 3. 逐元素算术指令

/**
 * 将两个Tensor寄存器的内容逐元素相加并输出到一个Tensor寄存器
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存器2（仅普通寄存器和0寄存器）
 */
void tadd(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容逐元素相减并输出到一个Tensor寄存器
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存器2（仅普通寄存器和0寄存器）
 */
void tsub(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容逐元素相乘并输出到一个Tensor寄存器
 * @note 结果类型可以是输入类型的两倍宽度以避免溢出，例如F8乘法结果可以是F16，I16乘法结果可以是I32
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存器2（仅普通寄存器和0寄存器）
 */
void tmul(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容逐元素取最大值并输出到一个Tensor寄存器
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存员2（仅普通寄存器和0寄存 器）
 */
void tmax(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容逐元素取最小值并输出到一个Tensor寄存器
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存员2（仅普通寄存器和0寄存 器）
 */
void tmin(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容逐元素进行按位与运算并输出到一个Tensor寄存器
 * @note 仅使用类型宽度，无视数据类型符号和格式
 * @param trdt 输出Tensor寄存器（仅整数类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（整数类型的普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存员2（仅整数类型的普通寄存器和0寄存器）
 */
void tand(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i);

/**
 * 将两个Tensor寄存器的内容逐元素进行按位或运算并输出到一个Tensor寄存器
 * @note 仅使用类型宽度，无视数据类型符号和格式
 * @param trdt 输出Tensor寄存器（仅整数类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（整数类型的普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存员2（仅整数类型的普通寄存器和0寄存器）
 */
void tor(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i);

/**
 * 将两个Tensor寄存器的内容逐元素进行按位异或运算并输出到一个Tensor寄存器
 * @note 仅使用类型宽度，无视数据类型符号和格式
 * @param trdt 输出Tensor寄存器（仅整数类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（整数类型的普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存员2（仅整数类型的普通寄存器和0寄存器）
 */
void txor(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i);

/**
 * 将一个Tensor寄存器的内容逐元素逻辑左移并输出到一个Tensor寄存器
 * @note 仅使用类型宽度，无视数据类型符号和格式
 * @param trdt 输出Tensor寄存器（仅整数类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（整数类型的普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存员2（仅整数类型的普通寄存器和0寄存器）
 */
void tsll(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i);

/**
 * 将一个Tensor寄存器的内容逐元素逻辑右移并输出到一个Tensor寄存器
 * @note 仅使用类型宽度，无视数据类型符号和格式
 * @param trdt 输出Tensor寄存器（仅整数类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（整数类型的普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存员2（仅整数类型的普通寄存器和0寄存器）
 */
void tsrl(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i);

/**
 * 将一个Tensor寄存器的内容逐元素算术右移并输出到一个Tensor寄存器
 * @note 仅使用类型宽度，无视数据类型符号和格式
 * @param trdt 输出Tensor寄存器（仅整数类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（整数类型的普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存员2（仅整数类型的普通寄存器和0寄存器）
 */
void tsra(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i);

/**
 * 将两个Tensor寄存器的内容逐元素根据掩码选择并输出到一个Tensor寄存器，掩码有效代表选择trs1zt的元素，无效代表选择trs2z的元素
 * @note 仅使用类型宽度，无视数据类型符号和格式
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器1（普通寄存器、0寄存器和Temp寄存器）
 * @param trs2z 输入Tensor寄存员2（仅普通寄存器和0寄存器）
 */
void tmerge(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将一个Tensor寄存器的内容逐元素进行按位取反并输出到一个Tensor寄存器
 * @note 仅使用类型宽度，无视数据类型符号和格式
 * @param trdt 输出Tensor寄存器（仅整数类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（整数类型的普通寄存器、0寄存器和Temp寄存器）
 */
void tnot(const TOPRand &trdt_i, const TOPRand &trs1zt_i);

/**
 * 将一个Tensor寄存器的内容逐元素取绝对值并输出到一个Tensor寄存器
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（普通寄存器、0寄存器和Temp寄存器）
 */
void tabs(const TOPRand &trdt, const TOPRand &trs1zt);

/**
 * 将一个Tensor寄存器的内容逐元素取相反数并输出到一个Tensor寄存器
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（普通寄存器、0寄存器和Temp寄存器）
 */
void tneg(const TOPRand &trdt, const TOPRand &trs1zt);

/**
 * 将一个Tensor寄存器的内容逐元素开平方并输出到一个Tensor寄存器
 * @note 非精确指令
 * @param trdt 输出Tensor寄存器（仅浮点类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（浮点类型的普通寄存器、0寄存器和Temp寄存器）
 */
void tsqrt(const TOPRand &trdt_f, const TOPRand &trs1zt_f);

/**
 * 将一个Tensor寄存器的内容逐元素取平方根倒数并输出到一个Tensor寄存器
 * @note 非精确指令
 * @param trdt 输出Tensor寄存器（仅浮点类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（浮点类型的普通寄存器、0寄存器和Temp寄存器）
 */
void trsqrt(const TOPRand &trdt_f, const TOPRand &trs1zt_f);

/**
 * 将一个Tensor寄存器的内容逐元素计算自然指数函数并输出到一个Tensor寄存器
 * @note 非精确指令
 * @param trdt 输出Tensor寄存器（仅浮点类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（浮点类型的普通寄存器、0寄存器和Temp寄存器）
 */
void texp(const TOPRand &trdt_f, const TOPRand &trs1zt_f);

/**
 * 将一个Tensor寄存器的内容逐元素取倒数并输出到一个Tensor寄存器
 * @note 非精确指令
 * @param trdt 输出Tensor寄存器（仅浮点类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（浮点类型的普通寄存器、0寄存器和Temp寄存器）
 */
void trcp(const TOPRand &trdt_f, const TOPRand &trs1zt_f);

/**
 * 将一个Tensor寄存器的内容逐元素执行ReLU激活函数并输出到一个Tensor寄存器
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（普通寄存器、0寄存器和Temp寄存器）
 */
void trelu(const TOPRand &trdt, const TOPRand &trs1zt);

/**
 * 将一个Tensor寄存器的内容逐元素执行ReLU6激活函数并输出到一个Tensor寄存器
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（普通寄存器、0寄存器和Temp寄存器）
 */
void trelu6(const TOPRand &trdt, const TOPRand &trs1zt);

/**
 * 将一个Tensor寄存器的内容逐元素执行Sigmoid激活函数并输出到一个Tensor寄存器
 * @note 非精确指令
 * @param trdt 输出Tensor寄存器（仅浮点类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（浮点类型的普通寄存器、0寄存器和Temp寄存器）
 */
void tsigmoid(const TOPRand &trdt_f, const TOPRand &trs1zt_f);

/**
 * 将一个Tensor寄存器的内容逐元素执行Tanh激活函数并输出到一个Tensor寄存器
 * @note 非精确指令
 * @param trdt 输出Tensor寄存器（仅浮点类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（浮点类型的普通寄存器、0寄存器和Temp寄存器）
 */
void ttanh(const TOPRand &trdt_f, const TOPRand &trs1zt_f);

/**
 * 将一个Tensor寄存器的内容逐元素执行GELU激活函数并输出到一个Tensor寄存器
 * @note 非精确指令
 * @param trdt 输出Tensor寄存器（仅浮点类型的普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（浮点类型的普通寄存器、0寄存器和Temp寄存器）
 */
void tgelu(const TOPRand &trdt_f, const TOPRand &trs1zt_f);


enum class TOV : uint8_t {
    DEF = 0b00, // 默认溢出行为（整数溢出回绕，浮点设置为inf）
    SAT = 0b01, // 饱和截断到目标类型范围
    CLIP = 0b10, // 截断到给定范围
};

/**
 * 将一个Tensor寄存器的内容逐元素转换数据类型并输出到一个Tensor寄存器
 * @param trdt 输出Tensor寄存器（仅普通寄存器和Temp寄存器）
 * @param trs1zt 输入Tensor寄存器（普通寄存器、0寄存器和Temp寄存器）
 * @param rm 结果舍入模式
 * @param ov 溢出行为
 * @param max 当ov为TOV::CLIP时的最大值
 * @param min 当ov为TOV::CLIP时的最小值
 */
void tcvt(const TOPRand &trdt, const TOPRand &trs1t, TRM rm, TOV ov, uint32_t max = 0, uint32_t min = 0);


// 4. Mask 指令

/**
 * 将一个Tensor寄存器的内容以布尔量视图移动到另一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器、Temp寄存器和Mask寄存器）
 */
void tmmv(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8);

/**
 * 将一个Tensor寄存器的内容以布尔量视图进行取反并输出到另一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器、Temp寄存器和Mask寄存器）
 */
void tmnot(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8);

/**
 * 将两个Tensor寄存器的内容以布尔量视图进行与运算并输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器、Temp寄存器和Mask寄存器）
 * @param trs2_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器和Mask寄存器）
 */
void tmand(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8);

/**
 * 将两个Tensor寄存器的内容以布尔量视图进行与非运算并输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器、Temp寄存器和Mask寄存器）
 * @param trs2_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器和Mask寄存器）
 */
void tmnand(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8);

/**
 * 将两个Tensor寄存器的内容以布尔量视图进行或运算并输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器、Temp寄存器和Mask寄存器）
 * @param trs2_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器和Mask寄存
 */
void tmor(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8);

/**
 * 将两个Tensor寄存器的内容以布尔量视图进行或非运算并输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器、Temp寄存器和Mask寄存器）
 * @param trs2_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器和Mask寄存
 */
void tmnor(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8);

/**
 * 将两个Tensor寄存器的内容以布尔量视图进行异或运算并输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器、Temp寄存器和Mask寄存器）
 * @param trs2_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器和Mask寄存
 */
void tmxor(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8);

/**
 * 将两个Tensor寄存器的内容以布尔量视图进行同或运算并输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器、Temp寄存器和Mask寄存器）
 * @param trs2_m8 输入Tensor寄存器（仅8位宽整数类型的普通寄存器、0寄存器和Mask寄存
 */
void tmnxor(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8);

/**
 * 将两个Tensor寄存器的内容进行相等比较并以布尔视图输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t 输入Tensor寄存器1（仅普通寄存器、0寄存器和Temp寄存器）
 * @param trs2 输入Tensor寄存器2（仅普通寄存器和0寄存器）
 */
void tmeq(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容进行不等比较并以布尔视图输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t 输入Tensor寄存器1（仅普通寄存器、0寄存器和Temp寄存器）
 * @param trs2 输入Tensor寄存器2（仅普通寄存器和0寄存器）
 */
void tmne(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容进行小于比较并以布尔视图输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t 输入Tensor寄存器1（仅普通寄存器、0寄存器和Temp寄存器）
 * @param trs2 输入Tensor寄存器2（仅普通寄存器和0寄存器）
 */
void tmlt(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容进行大于等于比较并以布尔视图输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t 输入Tensor寄存器1（仅普通寄存器、0寄存器和Temp寄存器）
 * @param trs2 输入Tensor寄存器2（仅普通寄存器和0寄存器）
 */
void tmge(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容进行小于等于比较并以布尔视图输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t 输入Tensor寄存器1（仅普通寄存器、0寄存器和Temp寄存器）
 * @param trs2 输入Tensor寄存器2（仅普通寄存器和0寄存器）
 */
void tmle(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z);

/**
 * 将两个Tensor寄存器的内容进行大于比较并以布尔视图输出到一个Tensor寄存器
 * @param trdt_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器）
 * @param trs1t 输入Tensor寄存器1（仅普通寄存器、0寄存器和Temp寄存器）
 * @param trs2 输入Tensor寄存器2（仅普通寄存器和0寄存器）
 */
void tmgt(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z);


// 5. 归约指令

/**
 * 将一个Tensor寄存器的内容进行最小值归约操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param row 归约结果写入位置的行索引
 * @param col 归约结果写入位置的列索引
 */
void tredmin(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t row = 0, uint8_t col = 0);

/**
 * 将一个Tensor寄存器的内容逐行进行最小值归约操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param col 归约结果写入位置的列索引
 */
void tredminRow(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t col = 0);

/**
 * 将一个Tensor寄存器的内容逐列进行最小值归约操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param row 归约结果写入位置的行索引
 */
void tredminCol(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t row = 0);


/**
 * 将一个Tensor寄存器的内容进行最大值归约操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param row 归约结果写入位置的行索引
 * @param col 归约结果写入位置的列索引
 */
void tredmax(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t row = 0, uint8_t col = 0);

/**
 * 将一个Tensor寄存器的内容逐行进行最大值归约操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param col 归约结果写入位置的列索引
 */
void tredmaxRow(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t col = 0);

/**
 * 将一个Tensor寄存器的内容逐列进行最大值归约操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param row 归约结果写入位置的行索引
 */
void tredmaxCol(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t row = 0);

/**
 * 将一个Tensor寄存器的内容进行求和归约操作并输出到一个Tensor寄存器
 * @note 结果类型只能是F32或I32(UI32)
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param row 归约结果写入位置的行索引
 * @param col 归约结果写入位置的列索引
 */
void tredsum(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t row = 0, uint8_t col = 0);

/**
 * 将一个Tensor寄存器的内容逐行进行求和归约操作并输出到一个Tensor寄存器
 * @note 结果类型只能是F32或I32(UI32)
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param col 归约结果写入位置的列索引
 */
void tredsumRow(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t col = 0);

/**
 * 将一个Tensor寄存器的内容逐列进行求和归约操作并输出到一个Tensor寄存器
 * @note 结果类型只能是F32或I32(UI32)
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param row 归约结果写入位置的行索引
 */
void tredsumCol(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t row = 0);

/**
 * 将一个Tensor寄存器的内容进行平均值归约操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param row 归约结果写入位置的行索引
 * @param col 归约结果写入位置的列索引
 */
void tredavg(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t row = 0, uint8_t col = 0);

/**
 * 将一个Tensor寄存器的内容逐行进行平均值归约操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param col 归约结果写入位置的列索引
 */
void tredavgRow(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t col = 0);

/**
 * 将一个Tensor寄存器的内容逐列进行平均值归约操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param broadcast 是否将归约结果广播到整个输出Tensor寄存器
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param row 归约结果写入位置的行索引
 */
void tredavgCol(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask = 0, uint8_t row = 0);

/**
 * 将一个Tensor寄存器的内容进行最大值池化操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param subtile_row 结果写入位置的子tile行索引，范围[0, window-1]
 * @param subtile_col 结果写入位置的子tile列索引，范围[0, window-1]
 */
void tpol2max(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, uint8_t mask = 0, uint8_t subtile_row = 0, uint8_t subtile_col = 0);

/**
 * 将一个Tensor寄存器的内容进行平均值池化操作并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param trs2z 输入Tensor寄存器（仅普通寄存器和0寄存器），作为非broadcast下无关元素的输出初始值
 * @param mask 使用的掩码逻辑寄存器编号，0表示不使用掩码，1/2/3表示TMASK1/2/3（作用于trs1）
 * @param subtile_row 结果写入位置的子tile行索引，范围[0, window-1]
 * @param subtile_col 结果写入位置的子tile列索引，范围[0, window-1]
 */
void tpol2avg(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, uint8_t mask = 0, uint8_t subtile_row = 0, uint8_t subtile_col = 0);

// 6. MISC

/**
 * 将一个Tensor寄存器的内容以标量值填充并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param value 用于填充的标量值
 */
void tfill(const TOPRand &trd, uint32_t value);

/**
 * 将一个Tensor寄存器的指定位置的元素广播到另一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param row 要广播的元素的行索引
 * @param col 要广播的元素的列索引
 */
void tbcast(const TOPRand &trd, const TOPRand &trs1, uint8_t row, uint8_t col);

/**
 * 将一个Tensor寄存器的指定列的元素逐行广播到另一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param col 要广播的列索引
 */
void tbcastRow(const TOPRand &trd, const TOPRand &trs1, uint8_t col);

/**
 * 将一个Tensor寄存器的指定行的元素逐列广播到另一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param row 要广播的行索引
 */
void tbcastCol(const TOPRand &trd, const TOPRand &trs1, uint8_t row);

/**
 * 将一个Tensor寄存器的内容进行行列平移并输出到一个Tensor寄存器，平移过程中超出边界的元素将被丢弃，空缺位置将被填充0
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 * @param row_offset 行平移的偏移量，正值表示向下平移，负值表示向上平移
 * @param col_offset 列平移的偏移量，正值表示向右平移，负值表示向左平移
 */
void tshift(const TOPRand &trd, const TOPRand &trs1, int8_t row_offset, int8_t col_offset);

/**
 * 将一个Tensor寄存器的内容进行转置并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 */
void ttrans(const TOPRand &trd, const TOPRand &trs1);

/**
 * 将一个Tensor寄存器的内容进行行翻转（上下翻转）并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 */
void ttransRevRow(const TOPRand &trd, const TOPRand &trs1);

/**
 * 将一个Tensor寄存器的内容进行列翻转（左右翻转）并输出到一个Tensor寄存器
 * @param trd 输出Tensor寄存器（仅普通寄存器）
 * @param trs1 输入Tensor寄存器（仅普通寄存器）
 */
void ttransRevCol(const TOPRand &trd, const TOPRand &trs1);

/**
 * 将一个Tensor寄存器的内容根据给定的边界（四个方向的空白元素数）进行掩码设置并输出到一个Tensor寄存器
 * @param trd_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器和Mask寄存器）
 * @param top 上边界的空白元素数
 * @param bottom 下边界的空白元素数
 * @param left 左边界的空白元素数
 * @param right 右边界的空白元素数
 * @param invert 是否反转掩码，即中央子矩形区域为无效，边界区域为有效
 */
void tsetmask(const TOPRand &trd_m8, uint8_t top, uint8_t bottom, uint8_t left, uint8_t right, bool invert = false);


/**
 * 将一个Tensor寄存器的内容根据给定的边界（四个方向的空白元素数）进行下三角掩码设置并输出到一个Tensor寄存器
 * @param trd_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器和Mask寄存器）
 * @param top 上边界的空白元素数
 * @param bottom 下边界的空白元素数
 * @param left 左边界的空白元素数
 * @param right 右边界的空白元素数
 * @param diagonal 是否包含对角线元素在有效区域内
 * @param invert 是否反转掩码，即选定三角区域为无效，其他区域为有效
 */
void tsetmaskLowTri(const TOPRand &trd_m8, uint8_t top, uint8_t bottom, uint8_t left, uint8_t right, bool diagonal, bool invert = false);

/**
 * 将一个Tensor寄存器的内容根据给定的边界（四个方向的空白元素数）进行上三角掩码设置并输出到一个Tensor寄存器
 * @param trd_m8 输出Tensor寄存器（仅8位宽整数类型的普通寄存器和Mask寄存器）
 * @param top 上边界的空白元素数
 * @param bottom 下边界的空白元素数
 * @param left 左边界的空白元素数
 * @param right 右边界的空白元素数
 * @param diagonal 是否包含对角线元素在有效区域内
 * @param invert 是否反转掩码，即选定三角区域为无效，其他区域为有效
 */
void tsetmaskUpTri(const TOPRand &trd_m8, uint8_t top, uint8_t bottom, uint8_t left, uint8_t right, bool diagonal, bool invert = false);




} // namespace oootpu
