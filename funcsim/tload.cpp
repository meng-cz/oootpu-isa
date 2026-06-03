
#include "regfile.hpp"

#include <vector>

using std::vector;

namespace oootpu {

inline RawTile tload_impl_nomask(const TOPRand &trd, TMemAddr baseaddr, uint32_t rowstride) {
    uint32_t width = dtypeWidth(trd.dtype) / 8;
    vector<uint8_t> buffer;
    buffer.resize(TILE_SIZE * TILE_SIZE * width);
    if (rowstride == TILE_SIZE) {
        memread(baseaddr, buffer.data(), static_cast<uint32_t>(buffer.size()));
    } else {
        for (TRegIdx i = 0; i < TILE_SIZE; ++i) {
            memread(
                baseaddr + static_cast<TMemAddr>(i) * rowstride * width,
                buffer.data() + i * TILE_SIZE * width,
                TILE_SIZE * width
            );
        }
    }
    
    RawTile tile;
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
    return tile;
}

inline RawTile tload_impl_mask(const TOPRand &trd, TMemAddr baseaddr, uint32_t rowstride, uint8_t mask) {
    uint32_t width = dtypeWidth(trd.dtype) / 8;
    auto &TMask = maskTile(mask);
    RawTile tile;
    TMemAddr row_addr_step = static_cast<TMemAddr>(rowstride) * width;
    LOOPIJ {
        tile[i][j] = 0;
        if (TMask[i][j]) {
            memread(baseaddr + i * row_addr_step + j * width, &tile[i][j], width);
        }
    }
    return tile;
}

void tload(const TOPRand &trd, TMemAddr baseaddr, uint8_t mask) {
    return tloadRow(trd, baseaddr, TILE_SIZE, mask);
}

void tloadRow(const TOPRand &trd, TMemAddr baseaddr, uint32_t rowstride, uint8_t mask) {
    CHECK_TREG_NORM(trd);

    uint32_t width = dtypeWidth(trd.dtype) / 8;
    assert((width == 1 || width == 2 || width == 4) && "不支持的元素宽度");
    CHECK_ADDR_ALIGN(baseaddr, width);

    RawTile tile = hasMask(mask) ? tload_impl_mask(trd, baseaddr, rowstride, mask) : tload_impl_nomask(trd, baseaddr, rowstride);
    setRawTile(trd, tile);
}


} // namespace oootpu
