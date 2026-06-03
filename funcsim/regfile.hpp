#pragma once

#include "funcsim.hpp"

#include <array>
#include <cassert>

using std::array;

namespace oootpu {

using RawData = uint32_t;
using RawTile = array<array<RawData, TILE_SIZE>, TILE_SIZE>;
using TileBool = array<array<bool, TILE_SIZE>, TILE_SIZE>;

#define LOOPI for (uint32_t i = 0; i < TILE_SIZE; ++i)
#define LOOPJ for (uint32_t j = 0; j < TILE_SIZE; ++j)
#define LOOPIJ for (uint32_t i = 0; i < TILE_SIZE; ++i) for (uint32_t j = 0; j < TILE_SIZE; ++j)

RawTile getRawTile(const TOPRand &t);

void setRawTile(const TOPRand &t, const RawTile &tile);

TileBool &maskTile(uint8_t mask);

inline uint8_t normalizeMask(uint8_t mask) {
    assert(mask <= 3 && "mask逻辑寄存器编号必须在0到3之间");
    return mask;
}

inline bool hasMask(uint8_t mask) {
    return normalizeMask(mask) != 0;
}


void memread(TMemAddr addr, void *data, uint32_t size);

void memwrite(TMemAddr addr, const void *data, uint32_t size);

#define CHECK_TREG_NORM(topr) assert(checkRegNorm(topr) && "仅普通寄存器")
#define CHECK_TREG_Z(topr) assert(checkRegZ(topr) && "允许普通寄存器和0寄存器")
#define CHECK_TREG_ZI(topr) assert(checkRegZI(topr) && "允许整数格式的普通寄存器和0寄存器")
#define CHECK_TREG_T(topr) assert(checkRegT(topr) && "允许普通寄存器和Temp寄存器")
#define CHECK_TREG_ZT(topr) assert(checkRegZT(topr) && "允许普通寄存器和0寄存器和Temp寄存器")

#define CHECK_TREG_TM8(topr) assert(checkRegTM8(topr) && "允许8位宽整数类型的普通寄存器、Temp寄存器和Mask寄存器")
#define CHECK_TREG_ZTM8(topr) assert(checkRegZTM8(topr) && "允许8位宽整数类型的普通寄存器、0寄存器、Temp寄存器和Mask寄存器")
#define CHECK_TREG_ZM8(topr) assert(checkRegZM8(topr) && "允许8位宽整数类型的普通寄存器、0寄存器和Mask寄存器")
#define CHECK_TREG_M8(topr) assert(checkRegM8(topr) && "允许8位宽整数类型的普通寄存器和Mask寄存器")

#define CHECK_ADDR_ALIGN(addr, dwidth) assert(((addr) % (dwidth * TILE_SIZE)) == 0 && "地址未对齐")

} // namespace oootpu
