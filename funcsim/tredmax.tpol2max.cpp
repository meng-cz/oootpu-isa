
#include "regfile.hpp"

#include "res/arithm.hpp"
#include "res/exactfp.hpp"

namespace oootpu {


inline RawData redmaxImpl(const vector<RawData> &data, TDType dtype) {
    RawData max_val = data[0];
    if (dtype == TDType::UINT8) {
        for (const auto &val : data) {
            max_val = std::max(max_val & 0xFF, val & 0xFF);
        }
    } else if (dtype == TDType::UINT16) {
        for (const auto &val : data) {
            max_val = std::max(max_val & 0xFFFF, val & 0xFFFF);
        }
    } else if (dtype == TDType::UINT32) {
        for (const auto &val : data) {
            max_val = std::max(max_val, val);
        }
    } else if (dtype == TDType::SINT8) {
        for (const auto &val : data) {
            int8_t a = static_cast<int8_t>(max_val & 0xFF);
            int8_t b = static_cast<int8_t>(val & 0xFF);
            max_val = static_cast<uint32_t>(std::max(a, b) & 0xFF);
        }
    } else if (dtype == TDType::SINT16) {
        for (const auto &val : data) {
            int16_t a = static_cast<int16_t>(max_val & 0xFFFF);
            int16_t b = static_cast<int16_t>(val & 0xFFFF);
            max_val = static_cast<uint32_t>(std::max(a, b) & 0xFFFF);
        }
    } else if (dtype == TDType::SINT32) {
        for (const auto &val : data) {
            int32_t a = static_cast<int32_t>(max_val);
            int32_t b = static_cast<int32_t>(val);
            max_val = static_cast<uint32_t>(std::max(a, b));
        }
    } else if (dtype == TDType::F8E4M3) {
        for (const auto &val : data) {
            max_val = tpuarithm::flt<4, 3>(max_val, val) ? val : max_val;
        }
    } else if (dtype == TDType::F8E5M2) {
        for (const auto &val : data) {
            max_val = tpuarithm::flt<5, 2>(max_val, val) ? val : max_val;
        }
    } else if (dtype == TDType::F16) {
        for (const auto &val : data) {
            max_val = tpuarithm::flt<5, 10>(max_val, val) ? val : max_val;
        }
    } else if (dtype == TDType::F32) {
        for (const auto &val : data) {
            max_val = tpuarithm::flt<8, 23>(max_val, val) ? val : max_val;
        }
    } else {
        assert(false && "不支持的reduction数据类型");
    }
    return max_val;
}

void tredmax(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t row, uint8_t col) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    CHECK_TREG_Z(trs2z);

    assert(trd.dtype == trs1.dtype && trd.dtype == trs2z.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    vector<RawData> data;
    data.reserve(TILE_SIZE * TILE_SIZE);

    RawTile tile1 = getRawTile(trs1);
    RawTile tile2 = getRawTile(trs2z);

    if (!hasMask(mask)) {
        LOOPIJ {
            data.push_back(tile1[i][j]);
        }
    } else {
        auto &TMask = maskTile(mask);
        LOOPIJ {
            if (TMask[i][j]) {
                data.push_back(tile1[i][j]);
            }
        }
    }

    RawData max_val = redmaxImpl(data, trd.dtype);

    if (broadcast) {
        LOOPIJ {
            tile2[i][j] = max_val;
        }
    } else {
        tile2[row][col] = max_val;
    }

    setRawTile(trd, tile2);
}

void tredmaxRow(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t col) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    CHECK_TREG_Z(trs2z);

    assert(trd.dtype == trs1.dtype && trd.dtype == trs2z.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    vector<RawData> data;
    data.reserve(TILE_SIZE);

    RawTile tile1 = getRawTile(trs1);
    RawTile tile2 = getRawTile(trs2z);

    LOOPI {
        data.clear();
        if (!hasMask(mask)) {
            LOOPJ {
                data.push_back(tile1[i][j]);
            }
        } else {
            auto &TMask = maskTile(mask);
            LOOPJ {
                if (TMask[i][j]) {
                    data.push_back(tile1[i][j]);
                }
            }
        }

        RawData max_val = redmaxImpl(data, trd.dtype);

        if (broadcast) {
            LOOPJ {
                tile2[i][j] = max_val;
            }
        } else {
            tile2[i][col] = max_val;
        }
    }

    setRawTile(trd, tile2);
}

void tredmaxCol(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t row) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    CHECK_TREG_Z(trs2z);

    assert(trd.dtype == trs1.dtype && trd.dtype == trs2z.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    vector<RawData> data;
    data.reserve(TILE_SIZE);

    RawTile tile1 = getRawTile(trs1);
    RawTile tile2 = getRawTile(trs2z);

    LOOPJ {
        data.clear();
        if (!hasMask(mask)) {
            LOOPI {
                data.push_back(tile1[i][j]);
            }
        } else {
            auto &TMask = maskTile(mask);
            LOOPI {
                if (TMask[i][j]) {
                    data.push_back(tile1[i][j]);
                }
            }
        }

        RawData max_val = redmaxImpl(data, trd.dtype);

        if (broadcast) {
            LOOPI {
                tile2[i][j] = max_val;
            }
        } else {
            tile2[row][j] = max_val;
        }
    }

    setRawTile(trd, tile2);
}

void tpol2max(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, uint8_t mask, uint8_t subtile_row, uint8_t subtile_col) {

    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    CHECK_TREG_Z(trs2z);

    assert(trd.dtype == trs1.dtype && trd.dtype == trs2z.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    vector<RawData> data;
    data.reserve(4);

    RawTile tile1 = getRawTile(trs1);
    RawTile tile2 = getRawTile(trs2z);

    constexpr uint8_t window = 2;
    constexpr uint8_t subtile_size = TILE_SIZE / window;

    assert(subtile_row < window && subtile_col < window && "子tile索引超出范围");

    for (uint32_t i = 0; i < subtile_size; i++) {
        for (uint32_t j = 0; j < subtile_size; j++) {
            data.clear();
            
            uint32_t base_row = i * window;
            uint32_t base_col = j * window;

            if (!hasMask(mask)) {
                for (uint32_t r = 0; r < window; r++) {
                    for (uint32_t c = 0; c < window; c++) {
                        data.push_back(tile1[base_row + r][base_col + c]);
                    }
                }
            } else {
                auto &TMask = maskTile(mask);
                for (uint32_t r = 0; r < window; r++) {
                    for (uint32_t c = 0; c < window; c++) {
                        if (TMask[base_row + r][base_col + c]) {
                            data.push_back(tile1[base_row + r][base_col + c]);
                        }
                    }
                }
            }

            RawData max_val = redmaxImpl(data, trd.dtype);

            tile2[subtile_row * subtile_size + i][subtile_col * subtile_size + j] = max_val;
        }
    }

    setRawTile(trd, tile2);
}



}
