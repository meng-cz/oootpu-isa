
#include "regfile.hpp"

#include <cassert>
#include <cstddef>

namespace oootpu {

TRegIdx getTRegNum() {
    return TREG_NUM;
}

using Tile8 = array<array<uint8_t, TILE_SIZE>, TILE_SIZE>;

static thread_local array<Tile8, TREG_NUM> TReg;
static thread_local RawTile TTemp;
static thread_local array<TileBool, 3> TMask;

TileBool &maskTile(uint8_t mask) {
    assert(mask >= 1 && mask <= 3 && "mask逻辑寄存器编号必须在1到3之间");
    return TMask[static_cast<size_t>(mask - 1)];
}

RawTile getRawTile(const TOPRand &t) {
    RawTile ret;
    if (t.type == TOPRandType::ZERO) {
        LOOPIJ {
            ret[i][j] = 0;
        }
    } else if (isMaskOperand(t)) {
        auto &mask = maskTile(static_cast<uint8_t>(t.type) - static_cast<uint8_t>(TOPRandType::M1) + 1);
        LOOPIJ {
            ret[i][j] = mask[i][j] ? 1u : 0u;
        }
    } else if (t.type == TOPRandType::TEMP) {
        ret = TTemp;
    } else {
        assert(t.index < TREG_NUM && "寄存器索引越界");
        uint32_t width = dtypeWidth(t.dtype) / 8;
        assert((width == 1 || width == 2 || width == 4) && "不支持的元素宽度");
        if (width == 1) {
            LOOPIJ {
                ret[i][j] = static_cast<RawData>(TReg[t.index][i][j]);
            }
        } else if (width == 2) {
            LOOPIJ {
                uint16_t val = static_cast<uint16_t>(
                    static_cast<uint16_t>(TReg[t.index][i][j]) |
                    static_cast<uint16_t>(static_cast<uint16_t>(TReg[t.index + 1][i][j]) << 8)
                );
                ret[i][j] = static_cast<RawData>(val);
            }
        } else if (width == 4) {
            LOOPIJ {
                uint32_t val = static_cast<uint32_t>(TReg[t.index][i][j]) |
                               (static_cast<uint32_t>(TReg[t.index + 1][i][j]) << 8) |
                               (static_cast<uint32_t>(TReg[t.index + 2][i][j]) << 16) |
                               (static_cast<uint32_t>(TReg[t.index + 3][i][j]) << 24);
                ret[i][j] = static_cast<RawData>(val);
            }
        }
    }
    return ret;
}

void setRawTile(const TOPRand &t, const RawTile &tile) {
    if (t.type == TOPRandType::TEMP) {
        LOOPIJ {
            TTemp[i][j] = tile[i][j];
        }
    } else if (isMaskOperand(t)) {
        auto &mask = maskTile(static_cast<uint8_t>(t.type) - static_cast<uint8_t>(TOPRandType::M1) + 1);
        LOOPIJ {
            mask[i][j] = tile[i][j] != 0;
        }
    } else if (t.type == TOPRandType::ZERO) {
        // 不需要实际存储
    } else {
        assert(t.index < TREG_NUM && "寄存器索引越界");
        uint32_t width = dtypeWidth(t.dtype) / 8;
        assert((width == 1 || width == 2 || width == 4) && "不支持的元素宽度");
        if (width == 1) {
            LOOPIJ {
                TReg[t.index][i][j] = static_cast<uint8_t>(tile[i][j]);
            }
        } else if (width == 2) {
            LOOPIJ {
                uint16_t val = static_cast<uint16_t>(tile[i][j]);
                TReg[t.index][i][j] = static_cast<uint8_t>(val & 0xFFu);
                TReg[t.index + 1][i][j] = static_cast<uint8_t>((val >> 8) & 0xFFu);
            }
        } else if (width == 4) {
            LOOPIJ {
                uint32_t val = static_cast<uint32_t>(tile[i][j]);
                TReg[t.index][i][j] = static_cast<uint8_t>(val & 0xFFu);
                TReg[t.index + 1][i][j] = static_cast<uint8_t>((val >> 8) & 0xFFu);
                TReg[t.index + 2][i][j] = static_cast<uint8_t>((val >> 16) & 0xFFu);
                TReg[t.index + 3][i][j] = static_cast<uint8_t>((val >> 24) & 0xFFu);
            }
        }
    }
}

TMemProxy *memProxy = nullptr;

void setMemProxy(TMemProxy *proxy) {
    memProxy = proxy;
}

void memread(TMemAddr addr, void *data, uint32_t size) {
    assert(memProxy != nullptr && "内存代理未设置");
    memProxy->read(addr, data, size);
}

void memwrite(TMemAddr addr, const void *data, uint32_t size) {
    assert(memProxy != nullptr && "内存代理未设置");
    memProxy->write(addr, data, size);
}

}
