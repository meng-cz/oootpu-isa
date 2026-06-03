
#include "regfile.hpp"

#include <vector>

using std::vector;

namespace oootpu {

inline array<TMemAddr, TILE_SIZE> genRowOffsetVec(const TOPRand &trs2_i, uint8_t rowsel) {
    uint32_t idx_width = dtypeWidth(trs2_i.dtype) / 8;
    bool idx_signed = dtypeIsSignedInt(trs2_i.dtype);
    array<TMemAddr, TILE_SIZE> ele_offsets;
    RawTile idx_tile = getRawTile(trs2_i);
    if (idx_width == 1 && idx_signed) {
        for (TRegIdx j = 0; j < TILE_SIZE; ++j) {
            ele_offsets[j] = static_cast<TMemAddr>(static_cast<int8_t>(idx_tile[rowsel][j] & 0xFF));
        }
    } else if (idx_width == 1 && !idx_signed) {
        for (TRegIdx j = 0; j < TILE_SIZE; ++j) {
            ele_offsets[j] = static_cast<TMemAddr>(idx_tile[rowsel][j] & 0xFF);
        }
    } else if (idx_width == 2 && idx_signed) {
        for (TRegIdx j = 0; j < TILE_SIZE; ++j) {
            ele_offsets[j] = static_cast<TMemAddr>(static_cast<int16_t>(idx_tile[rowsel][j] & 0xFFFF));
        }
    } else if (idx_width == 2 && !idx_signed) {
        for (TRegIdx j = 0; j < TILE_SIZE; ++j) {
            ele_offsets[j] = static_cast<TMemAddr>(idx_tile[rowsel][j] & 0xFFFF);
        }
    } else if (idx_width == 4 && idx_signed) {
        for (TRegIdx j = 0; j < TILE_SIZE; ++j) {
            ele_offsets[j] = static_cast<TMemAddr>(static_cast<int32_t>(idx_tile[rowsel][j]));
        }
    } else if (idx_width == 4 && !idx_signed) {
        for (TRegIdx j = 0; j < TILE_SIZE; ++j) {
            ele_offsets[j] = static_cast<TMemAddr>(idx_tile[rowsel][j]);
        }
    }
    return ele_offsets;
}

void tgatherRow(const TOPRand &trd, const TOPRand &trs2_i, TMemAddr baseaddr, uint8_t rowsel, uint8_t mask) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_ZI(trs2_i);
    uint32_t width = dtypeWidth(trd.dtype) / 8;
    uint32_t idx_width = dtypeWidth(trs2_i.dtype) / 8;

    assert((width == 1 || width == 2 || width == 4) && "不支持的元素宽度");
    assert((idx_width == 1 || idx_width == 2 || idx_width == 4) && "不支持的索引元素宽度");
    assert(rowsel < TILE_SIZE && "行选择器越界");
    CHECK_ADDR_ALIGN(baseaddr, width);

    array<TMemAddr, TILE_SIZE> ele_offsets = genRowOffsetVec(trs2_i, rowsel);

    RawTile tile;
    if (!hasMask(mask)) {
        vector<uint8_t> buffer;
        buffer.resize(TILE_SIZE * TILE_SIZE * width);
        LOOPI {
            TMemAddr row_base = baseaddr + ele_offsets[i] * width;
            memread(row_base, buffer.data() + i * TILE_SIZE * width, TILE_SIZE * width);
        }
        uint32_t cur = 0;
        if (width == 1) {
            LOOPIJ {
                tile[i][j] = buffer[cur];
                cur += width;
            }
        } else if (width == 2) {
            LOOPIJ {
                const uint16_t val = static_cast<uint16_t>(
                    static_cast<uint16_t>(buffer[cur]) |
                    static_cast<uint16_t>(static_cast<uint16_t>(buffer[cur + 1]) << 8)
                );
                tile[i][j] = val;
                cur += width;
            }
        } else if (width == 4) {
            LOOPIJ {
                uint32_t val = static_cast<uint32_t>(buffer[cur]) |
                               (static_cast<uint32_t>(buffer[cur + 1]) << 8) |
                               (static_cast<uint32_t>(buffer[cur + 2]) << 16) |
                               (static_cast<uint32_t>(buffer[cur + 3]) << 24);
                tile[i][j] = val;
                cur += width;
            }
        }
    } else {
        vector<uint8_t> line_buffer;
        line_buffer.resize(TILE_SIZE * width);
        auto &TMask = maskTile(mask);
        LOOPI {
            if (!TMask[rowsel][i]) {
                LOOPJ {
                    tile[i][j] = 0;
                }
            } else {
                TMemAddr row_base = baseaddr + ele_offsets[i] * width;
                memread(row_base, line_buffer.data(), static_cast<uint32_t>(line_buffer.size()));
                uint32_t cur = 0;
                if (width == 1) {
                    LOOPJ {
                        tile[i][j] = line_buffer[cur];
                        cur += width;
                    }
                } else if (width == 2) {
                    LOOPJ {
                        const uint16_t val = static_cast<uint16_t>(
                            static_cast<uint16_t>(line_buffer[cur]) |
                            static_cast<uint16_t>(static_cast<uint16_t>(line_buffer[cur + 1]) << 8)
                        );
                        tile[i][j] = val;
                        cur += width;
                    }
                } else if (width == 4) {
                    LOOPJ {
                        uint32_t val = static_cast<uint32_t>(line_buffer[cur]) |
                                       (static_cast<uint32_t>(line_buffer[cur + 1]) << 8) |
                                       (static_cast<uint32_t>(line_buffer[cur + 2]) << 16) |
                                       (static_cast<uint32_t>(line_buffer[cur + 3]) << 24);
                        tile[i][j] = val;
                        cur += width;
                    }
                }
            }

        }
    }
    setRawTile(trd, tile);
}

void tscatterRow(const TOPRand &trs1z, const TOPRand &trs2_i, TMemAddr baseaddr, uint8_t rowsel, uint8_t mask) {
    CHECK_TREG_Z(trs1z);
    CHECK_TREG_ZI(trs2_i);
    uint32_t width = dtypeWidth(trs1z.dtype) / 8;
    uint32_t idx_width = dtypeWidth(trs2_i.dtype) / 8;

    assert((width == 1 || width == 2 || width == 4) && "不支持的元素宽度");
    assert((idx_width == 1 || idx_width == 2 || idx_width == 4) && "不支持的索引元素宽度");
    assert(rowsel < TILE_SIZE && "行选择器越界");
    CHECK_ADDR_ALIGN(baseaddr, width);

    array<TMemAddr, TILE_SIZE> ele_offsets = genRowOffsetVec(trs2_i, rowsel);
    RawTile tile = getRawTile(trs1z);
    vector<uint8_t> buffer;
    buffer.resize(TILE_SIZE * TILE_SIZE * width);
    uint32_t cur = 0;
    if (width == 1) {
        LOOPIJ {
            buffer[cur] = static_cast<uint8_t>(tile[i][j] & 0xFF);
            cur += width;
        }
    } else if (width == 2) {
        LOOPIJ {
            buffer[cur] = static_cast<uint8_t>(tile[i][j] & 0xFF);
            buffer[cur + 1] = static_cast<uint8_t>((tile[i][j] >> 8) & 0xFF);
            cur += width;
        }
    } else if (width == 4) {
        LOOPIJ {
            buffer[cur] = static_cast<uint8_t>(tile[i][j] & 0xFF);
            buffer[cur + 1] = static_cast<uint8_t>((tile[i][j] >> 8) & 0xFF);
            buffer[cur + 2] = static_cast<uint8_t>((tile[i][j] >> 16) & 0xFF);
            buffer[cur + 3] = static_cast<uint8_t>((tile[i][j] >> 24) & 0xFF);
            cur += width;
        }
    }
    if (!hasMask(mask)) {
        LOOPI {
            TMemAddr row_base = baseaddr + ele_offsets[i] * width;
            memwrite(row_base, buffer.data() + i * TILE_SIZE * width, TILE_SIZE * width);
        }
    } else {
        auto &TMask = maskTile(mask);
        LOOPI {
            if (!TMask[rowsel][i]) {
                continue;
            }
            TMemAddr row_base = baseaddr + ele_offsets[i] * width;
            memwrite(row_base, buffer.data() + i * TILE_SIZE * width, TILE_SIZE * width);
        }
    }
}


} // namespace oootpu
