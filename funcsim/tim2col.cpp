
#include "regfile.hpp"

namespace oootpu {

struct Im2colConfig {
    TMemAddr baseaddr;
    Im2colLayout layout; 
    uint16_t N;
    uint16_t H;
    uint16_t W;
    uint16_t C;
    uint16_t OH;
    uint16_t OW;
};

inline std::pair<uint16_t, uint16_t> getOutputDim(uint16_t H, uint16_t W, Im2colLayout layout) {
    switch (layout) {
        case Im2colLayout::K1S1P0:
            return {H, W};
        case Im2colLayout::K3S1P1:
            return {H, W};
        case Im2colLayout::K3S2P1:
            return {(H + 1) / 2, (W + 1) / 2};
        case Im2colLayout::K5S1P2:
            return {H, W};
        case Im2colLayout::K7S2P3:
            return {(H + 1) / 2, (W + 1) / 2};
    }
    return {0, 0};
}

static thread_local Im2colConfig im2colConfig;

void tim2cfgNHWC(Im2colLayout layout, TMemAddr baseaddr, uint16_t N, uint16_t H, uint16_t W, uint16_t C) {
    
    im2colConfig.layout = layout;
    im2colConfig.baseaddr = baseaddr;
    im2colConfig.N = N;
    im2colConfig.H = H;
    im2colConfig.W = W;
    im2colConfig.C = C;
    auto [OH, OW] = getOutputDim(H, W, layout);
    im2colConfig.OH = OH;
    im2colConfig.OW = OW;
}

inline RawTile im2col_impl_NHWC(uint32_t dwidth, uint16_t subtile_row, uint16_t subtile_col) {
    uint16_t kernel = 0;
    uint16_t stride = 0;
    uint16_t pad = 0;
    switch (im2colConfig.layout) {
        case Im2colLayout::K1S1P0:
            kernel = 1;
            stride = 1;
            pad = 0;
            break;
        case Im2colLayout::K3S1P1:
            kernel = 3;
            stride = 1;
            pad = 1;
            break;
        case Im2colLayout::K3S2P1:
            kernel = 3;
            stride = 2;
            pad = 1;
            break;
        case Im2colLayout::K5S1P2:
            kernel = 5;
            stride = 1;
            pad = 2;
            break;
        case Im2colLayout::K7S2P3:
            kernel = 7;
            stride = 2;
            pad = 3;
            break;
    }

    RawTile tile {};
    const uint32_t rows_per_batch = static_cast<uint32_t>(im2colConfig.OH) * im2colConfig.OW;
    const uint32_t cols_per_kernel = static_cast<uint32_t>(kernel) * kernel * im2colConfig.C;
    const uint32_t row_base = static_cast<uint32_t>(subtile_row) * TILE_SIZE;
    const uint32_t col_base = static_cast<uint32_t>(subtile_col) * TILE_SIZE;
    const uint32_t plane_stride = static_cast<uint32_t>(im2colConfig.H) * im2colConfig.W;
    const uint32_t batch_stride = plane_stride * im2colConfig.C;

    LOOPI {
        const uint32_t out_row = row_base + i;
        if (out_row >= static_cast<uint32_t>(im2colConfig.N) * rows_per_batch) {
            continue;
        }

        const uint32_t n = out_row / rows_per_batch;
        const uint32_t batch_row = out_row % rows_per_batch;
        const uint32_t oh = batch_row / im2colConfig.OW;
        const uint32_t ow = batch_row % im2colConfig.OW;
        const int32_t ih_base = static_cast<int32_t>(oh * stride) - pad;
        const int32_t iw_base = static_cast<int32_t>(ow * stride) - pad;
        const TMemAddr batch_base = im2colConfig.baseaddr + static_cast<TMemAddr>(n) * batch_stride * dwidth;

        LOOPJ {
            const uint32_t out_col = col_base + j;
            if (out_col >= cols_per_kernel) {
                continue;
            }

            const uint32_t c = out_col % im2colConfig.C;
            const uint32_t kernel_idx = out_col / im2colConfig.C;
            const uint32_t kh = kernel_idx / kernel;
            const uint32_t kw = kernel_idx % kernel;
            const int32_t ih = ih_base + static_cast<int32_t>(kh);
            const int32_t iw = iw_base + static_cast<int32_t>(kw);
            if (ih < 0 || iw < 0 || ih >= im2colConfig.H || iw >= im2colConfig.W) {
                continue;
            }

            const uint32_t input_idx =
                (static_cast<uint32_t>(ih) * im2colConfig.W + static_cast<uint32_t>(iw)) * im2colConfig.C + c;
            memread(batch_base + static_cast<TMemAddr>(input_idx) * dwidth, &tile[i][j], dwidth);
        }
    }

    return tile;
}

void tim2col(const TOPRand &trd, uint16_t subtile_row, uint16_t subtile_col) {
    CHECK_TREG_NORM(trd);

    uint32_t width = dtypeWidth(trd.dtype) / 8;
    assert((width == 1 || width == 2 || width == 4) && "不支持的元素宽度");
    CHECK_ADDR_ALIGN(im2colConfig.baseaddr, width);
    
    RawTile tile = im2col_impl_NHWC(width, subtile_row, subtile_col);
    setRawTile(trd, tile);
}




};
