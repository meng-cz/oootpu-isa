
#include "regfile.hpp"

#include "res/arithm.hpp"
#include "res/exactfp.hpp"

namespace oootpu {

template <typename T>
inline T clipFromI64(int64_t val) {
    if (val > static_cast<int64_t>(std::numeric_limits<T>::max())) {
        return std::numeric_limits<T>::max();
    }
    if (val < static_cast<int64_t>(std::numeric_limits<T>::min())) {
        return std::numeric_limits<T>::min();
    }
    return static_cast<T>(val);
}

inline RawData redavgImpl(const vector<RawData> &data, TDType dtype) {
    if (data.empty()) {
        return 0;
    }
    ExactFloat sum = ExactFloat::ZERO();
    if (dtype == TDType::UINT8) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<uint8_t>(val & 0xFF));
    } else if (dtype == TDType::UINT16) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<uint16_t>(val & 0xFFFF));
    } else if (dtype == TDType::UINT32) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<uint32_t>(val));
    } else if (dtype == TDType::SINT8) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<int8_t>(val & 0xFF));
    } else if (dtype == TDType::SINT16) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<int16_t>(val & 0xFFFF));
    } else if (dtype == TDType::SINT32) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<int32_t>(val));
    } else if (dtype == TDType::F8E4M3) {
        for (const auto &val : data) sum = sum + floatToEF<4, 3>(val & 0xFF);
    } else if (dtype == TDType::F8E5M2) {
        for (const auto &val : data) sum = sum + floatToEF<5, 2>(val & 0xFF);
    } else if (dtype == TDType::F16) {
        for (const auto &val : data) sum = sum + floatToEF<5, 10>(val & 0xFFFF);
    } else if (dtype == TDType::F32) {
        for (const auto &val : data) sum = sum + floatToEF<8, 23>(val);
    } else {
        assert(false && "不支持的reduction数据类型");
    }
    uint32_t count = static_cast<uint32_t>(data.size());

    if (dtypeIsInt(dtype)) {
        int64_t sum_int = intFromEF<int64_t>(sum, RoundingMode::RNE, true);
        int64_t avg_int = sum_int / count;
        if (dtype == TDType::SINT32) {
            return static_cast<uint32_t>(clipFromI64<int32_t>(avg_int));
        } else if (dtype == TDType::UINT32) {
            return static_cast<uint32_t>(clipFromI64<uint32_t>(avg_int));
        } else if (dtype == TDType::SINT16) {
            return static_cast<uint32_t>(clipFromI64<int16_t>(avg_int));
        } else if (dtype == TDType::UINT16) {
            return static_cast<uint32_t>(clipFromI64<uint16_t>(avg_int));
        } else if (dtype == TDType::SINT8) {
            return static_cast<uint32_t>(clipFromI64<int8_t>(avg_int));
        } else if (dtype == TDType::UINT8) {
            return static_cast<uint32_t>(clipFromI64<uint8_t>(avg_int));
        }
    } else if (dtype == TDType::F32) {
        return divUIntAndRound<8, 23>(sum, count, RoundingMode::RNE);
    } else if (dtype == TDType::F16) {
        return divUIntAndRound<5, 10>(sum, count, RoundingMode::RNE);
    } else if (dtype == TDType::F8E4M3) {
        return divUIntAndRound<4, 3>(sum, count, RoundingMode::RNE);
    } else if (dtype == TDType::F8E5M2) {
        return divUIntAndRound<5, 2>(sum, count, RoundingMode::RNE);
    }
    assert(false && "不支持的tredavg输出数据类型");
    return 0;
}

void tredavg(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t row, uint8_t col) {
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

    RawData sum_val = redavgImpl(data, trs1.dtype);

    if (broadcast) {
        LOOPIJ {
            tile2[i][j] = sum_val;
        }
    } else {
        tile2[row][col] = sum_val;
    }

    setRawTile(trd, tile2);
}

void tredavgRow(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t col) {
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

        RawData sum_val = redavgImpl(data, trs1.dtype);

        if (broadcast) {
            LOOPJ {
                tile2[i][j] = sum_val;
            }
        } else {
            tile2[i][col] = sum_val;
        }
    }

    setRawTile(trd, tile2);
}

void tredavgCol(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t row) {
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

        RawData sum_val = redavgImpl(data, trs1.dtype);

        if (broadcast) {
            LOOPI {
                tile2[i][j] = sum_val;
            }
        } else {
            tile2[row][j] = sum_val;
        }
    }

    setRawTile(trd, tile2);
}

void tpol2avg(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, uint8_t mask, uint8_t subtile_row, uint8_t subtile_col) {

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

            RawData avg_val = redavgImpl(data, trd.dtype);

            tile2[subtile_row * subtile_size + i][subtile_col * subtile_size + j] = avg_val;
        }
    }

    setRawTile(trd, tile2);
}



}
