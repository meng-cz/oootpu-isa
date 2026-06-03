
#include "regfile.hpp"

#include "res/arithm.hpp"
#include "res/exactfp.hpp"

namespace oootpu {

inline RawData redminImpl(const vector<RawData> &data, TDType dtype) {
    RawData min_val = data[0];
    if (dtype == TDType::UINT8) {
        for (const auto &val : data) {
            min_val = std::min(min_val & 0xFF, val & 0xFF);
        }
    } else if (dtype == TDType::UINT16) {
        for (const auto &val : data) {
            min_val = std::min(min_val & 0xFFFF, val & 0xFFFF);
        }
    } else if (dtype == TDType::UINT32) {
        for (const auto &val : data) {
            min_val = std::min(min_val, val);
        }
    } else if (dtype == TDType::SINT8) {
        for (const auto &val : data) {
            int8_t a = static_cast<int8_t>(min_val & 0xFF);
            int8_t b = static_cast<int8_t>(val & 0xFF);
            min_val = static_cast<uint32_t>(std::min(a, b) & 0xFF);
        }
    } else if (dtype == TDType::SINT16) {
        for (const auto &val : data) {
            int16_t a = static_cast<int16_t>(min_val & 0xFFFF);
            int16_t b = static_cast<int16_t>(val & 0xFFFF);
            min_val = static_cast<uint32_t>(std::min(a, b) & 0xFFFF);
        }
    } else if (dtype == TDType::SINT32) {
        for (const auto &val : data) {
            int32_t a = static_cast<int32_t>(min_val);
            int32_t b = static_cast<int32_t>(val);
            min_val = static_cast<uint32_t>(std::min(a, b));
        }
    } else if (dtype == TDType::F8E4M3) {
        for (const auto &val : data) {
            min_val = tpuarithm::fgt<4, 3>(min_val, val) ? val : min_val;
        }
    } else if (dtype == TDType::F8E5M2) {
        for (const auto &val : data) {
            min_val = tpuarithm::fgt<5, 2>(min_val, val) ? val : min_val;
        }
    } else if (dtype == TDType::F16) {
        for (const auto &val : data) {
            min_val = tpuarithm::fgt<5, 10>(min_val, val) ? val : min_val;
        }
    } else if (dtype == TDType::F32) {
        for (const auto &val : data) {
            min_val = tpuarithm::fgt<8, 23>(min_val, val) ? val : min_val;
        }
    } else {
        assert(false && "不支持的reduction数据类型");
    }
    return min_val;
}

void tredmin(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t row, uint8_t col) {

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

    RawData min_val = redminImpl(data, trd.dtype);
    
    if (broadcast) {
        LOOPIJ {
            tile2[i][j] = min_val;
        }
    } else {
        tile2[row][col] = min_val;
    }

    setRawTile(trd, tile2);
}

void tredminRow(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t col) {
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

        RawData min_val = redminImpl(data, trd.dtype);

        if (broadcast) {
            LOOPJ {
                tile2[i][j] = min_val;
            }
        } else {
            tile2[i][col] = min_val;
        }
    }

    setRawTile(trd, tile2);
}

void tredminCol(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t row) {
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

        RawData min_val = redminImpl(data, trd.dtype);

        if (broadcast) {
            LOOPI {
                tile2[i][j] = min_val;
            }
        } else {
            tile2[row][j] = min_val;
        }
    }

    setRawTile(trd, tile2);
}

}
