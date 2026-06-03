
#include "regfile.hpp"

#include "res/arithm.hpp"
#include "res/exactfp.hpp"

namespace oootpu {

inline TDType redsumDstDType(TDType src_dtype) {
    if (src_dtype == TDType::SINT8 || src_dtype == TDType::SINT16 || src_dtype == TDType::SINT32) {
        return TDType::SINT32;
    }
    if (src_dtype == TDType::UINT8 || src_dtype == TDType::UINT16 || src_dtype == TDType::UINT32) {
        return TDType::UINT32;
    }
    if (src_dtype == TDType::F8E4M3 || src_dtype == TDType::F8E5M2 || src_dtype == TDType::F16 || src_dtype == TDType::F32) {
        return TDType::F32;
    }
    assert(false && "不支持的reduction数据类型");
    return TDType::SINT32;
}

inline RawData redsumImpl(const vector<RawData> &data, TDType src_dtype, TDType dst_dtype) {
    ExactFloat sum = ExactFloat::ZERO();
    if (src_dtype == TDType::UINT8) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<uint8_t>(val & 0xFF));
    } else if (src_dtype == TDType::UINT16) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<uint16_t>(val & 0xFFFF));
    } else if (src_dtype == TDType::UINT32) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<uint32_t>(val));
    } else if (src_dtype == TDType::SINT8) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<int8_t>(val & 0xFF));
    } else if (src_dtype == TDType::SINT16) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<int16_t>(val & 0xFFFF));
    } else if (src_dtype == TDType::SINT32) {
        for (const auto &val : data) sum = sum + intToEF(static_cast<int32_t>(val));
    } else if (src_dtype == TDType::F8E4M3) {
        for (const auto &val : data) sum = sum + floatToEF<4, 3>(val & 0xFF);
    } else if (src_dtype == TDType::F8E5M2) {
        for (const auto &val : data) sum = sum + floatToEF<5, 2>(val & 0xFF);
    } else if (src_dtype == TDType::F16) {
        for (const auto &val : data) sum = sum + floatToEF<5, 10>(val & 0xFFFF);
    } else if (src_dtype == TDType::F32) {
        for (const auto &val : data) sum = sum + floatToEF<8, 23>(val);
    } else {
        assert(false && "不支持的reduction数据类型");
    }

    if (dst_dtype == TDType::F32) {
        return static_cast<RawData>(floatFromEF<8, 23>(sum, RoundingMode::RNE, false));
    }
    if (dst_dtype == TDType::SINT32) {
        return static_cast<uint32_t>(intFromEF<int32_t>(sum, RoundingMode::RNE, true));
    }
    if (dst_dtype == TDType::UINT32) {
        return static_cast<uint32_t>(intFromEF<uint32_t>(sum, RoundingMode::RNE, true));
    }
    assert(false && "tredsum输出类型必须是F32、SINT32或UINT32");
    return 0;
}

void tredsum(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t row, uint8_t col) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    CHECK_TREG_Z(trs2z);

    TDType expected_dtype = redsumDstDType(trs1.dtype);
    assert(trd.dtype == expected_dtype && trs2z.dtype == expected_dtype && "tredsum输出类型必须与trs1对应的32位类型一致");

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

    RawData sum_val = redsumImpl(data, trs1.dtype, trd.dtype);

    if (broadcast) {
        LOOPIJ {
            tile2[i][j] = sum_val;
        }
    } else {
        tile2[row][col] = sum_val;
    }

    setRawTile(trd, tile2);
}

void tredsumRow(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t col) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    CHECK_TREG_Z(trs2z);

    TDType expected_dtype = redsumDstDType(trs1.dtype);
    assert(trd.dtype == expected_dtype && trs2z.dtype == expected_dtype && "tredsum输出类型必须与trs1对应的32位类型一致");

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

        RawData sum_val = redsumImpl(data, trs1.dtype, trd.dtype);

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

void tredsumCol(const TOPRand &trd, const TOPRand &trs1, const TOPRand &trs2z, bool broadcast, uint8_t mask, uint8_t row) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    CHECK_TREG_Z(trs2z);

    TDType expected_dtype = redsumDstDType(trs1.dtype);
    assert(trd.dtype == expected_dtype && trs2z.dtype == expected_dtype && "tredsum输出类型必须与trs1对应的32位类型一致");

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

        RawData sum_val = redsumImpl(data, trs1.dtype, trd.dtype);

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

}
