
#include "regfile.hpp"

#include <vector>

using std::vector;

namespace oootpu {

inline void tstore_impl_nomask(const TOPRand &trd, TMemAddr baseaddr, uint32_t rowstride, const RawTile &tile) {
    uint32_t width = dtypeWidth(trd.dtype) / 8;
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

    if (rowstride == TILE_SIZE) {
        memwrite(baseaddr, buffer.data(), static_cast<uint32_t>(buffer.size()));
    } else {
        for (TRegIdx i = 0; i < TILE_SIZE; ++i) {
            memwrite(
                baseaddr + static_cast<TMemAddr>(i) * rowstride * width,
                buffer.data() + i * TILE_SIZE * width,
                TILE_SIZE * width
            );
        }
    }
}

inline void tstore_impl_mask(const TOPRand &trd, TMemAddr baseaddr, uint32_t rowstride, const RawTile &tile, uint8_t mask) {
    uint32_t width = dtypeWidth(trd.dtype) / 8;
    auto &TMask = maskTile(mask);
    TMemAddr row_addr_step = static_cast<TMemAddr>(rowstride) * width;
    LOOPIJ {
        if (TMask[i][j]) {
            memwrite(baseaddr + i * row_addr_step + j * width, &tile[i][j], width);
        }
    }
}

void tstore(const TOPRand &trs1z, TMemAddr baseaddr, uint8_t mask) {
    tstoreRow(trs1z, baseaddr, TILE_SIZE, mask);
}

void tstoreRow(const TOPRand &trs1z, TMemAddr baseaddr, uint32_t rowstride, uint8_t mask) {
    CHECK_TREG_Z(trs1z);

    uint32_t width = dtypeWidth(trs1z.dtype) / 8;
    assert((width == 1 || width == 2 || width == 4) && "不支持的元素宽度");
    CHECK_ADDR_ALIGN(baseaddr, width);

    RawTile tile = getRawTile(trs1z);
    if (hasMask(mask)) {
        tstore_impl_mask(trs1z, baseaddr, rowstride, tile, mask);
    } else {
        tstore_impl_nomask(trs1z, baseaddr, rowstride, tile);
    }
}

}
