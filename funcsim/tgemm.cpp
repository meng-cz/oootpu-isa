
#include "regfile.hpp"

#include "res/exactfp.hpp"

namespace oootpu {

using TileEFP = array<array<ExactFloat, TILE_SIZE>, TILE_SIZE>;

constexpr uint8_t TACC_NUM = 4;
static thread_local array<TileEFP, TACC_NUM> TAccs;

inline TileEFP &getTAcc(uint8_t tacc_idx) {
    assert(tacc_idx < TACC_NUM && "逻辑累加寄存器编号必须在0~3范围内");
    return TAccs[tacc_idx];
}

inline TileEFP prepareGEMM(const TOPRand &trsz, bool trans) {
    if (trsz.type == TOPRandType::ZERO) {
        TileEFP tile;
        LOOPIJ {
            tile[i][j] = ExactFloat::ZERO();
        }
        return tile;
    }
    TileEFP tile;
    RawTile raw_tile = getRawTile(trsz);
    if (trsz.dtype == TDType::SINT8) {
        LOOPIJ {
            tile[i][j] = intToEF(static_cast<int8_t>((trans ? raw_tile[j][i] : raw_tile[i][j]) & 0xFF));
        }
    } else if (trsz.dtype == TDType::UINT8) {
        LOOPIJ {
            tile[i][j] = intToEF(static_cast<uint8_t>((trans ? raw_tile[j][i] : raw_tile[i][j]) & 0xFF));
        }
    } else if (trsz.dtype == TDType::SINT16) {
        LOOPIJ {
            tile[i][j] = intToEF(static_cast<int16_t>((trans ? raw_tile[j][i] : raw_tile[i][j]) & 0xFFFF));
        }
    } else if (trsz.dtype == TDType::UINT16) {
        LOOPIJ {
            tile[i][j] = intToEF(static_cast<uint16_t>((trans ? raw_tile[j][i] : raw_tile[i][j]) & 0xFFFF));
        }
    } else if (trsz.dtype == TDType::F8E4M3) {
        LOOPIJ {
            tile[i][j] = floatToEF<4, 3>(trans ? raw_tile[j][i] : raw_tile[i][j]);
        }
    } else if (trsz.dtype == TDType::F8E5M2) {
        LOOPIJ {
            tile[i][j] = floatToEF<5, 2>(trans ? raw_tile[j][i] : raw_tile[i][j]);
        }
    } else if (trsz.dtype == TDType::F16) {
        LOOPIJ {
            tile[i][j] = floatToEF<5, 10>(trans ? raw_tile[j][i] : raw_tile[i][j]);
        }
    } else {
        assert(false && "不支持的GEMM数据类型");
    }

    return tile;
}

void tgemm(uint8_t tacc_idx, const TOPRand &trs1z, const TOPRand &trs2z, bool trans1, bool trans2, bool accum) {

    assert(trs1z.dtype == trs2z.dtype && "输入矩阵数据类型必须相同");
    CHECK_TREG_Z(trs1z);
    CHECK_TREG_Z(trs2z);

    TileEFP A = prepareGEMM(trs1z, trans1);
    TileEFP B = prepareGEMM(trs2z, trans2);
    TileEFP &TAcc = getTAcc(tacc_idx);

    if (!accum) {
        LOOPIJ {
            TAcc[i][j] = ExactFloat::ZERO();
        }
    }
    LOOPIJ {
        for (TRegIdx k = 0; k < TILE_SIZE; ++k) {
            TAcc[i][j] = TAcc[i][j] + (A[i][k] * B[k][j]);
        }
    }
}

void taccget(uint8_t tacc_idx, const TOPRand &trdt, const TOPRand &trs1zt, TRM rm, bool sat) {

    CHECK_TREG_T(trdt);
    CHECK_TREG_ZT(trs1zt);
    assert(trdt.dtype == trs1zt.dtype && "输出数据类型必须与输入矩阵数据类型相同");
    const TileEFP &TAcc = getTAcc(tacc_idx);

    RoundingMode rounding_mode;
    switch (rm) {
        case TRM::RNE: rounding_mode = RoundingMode::RNE; break;
        case TRM::RTZ: rounding_mode = RoundingMode::RTZ; break;
        case TRM::RDN: rounding_mode = RoundingMode::RDN; break;
        case TRM::RUP: rounding_mode = RoundingMode::RUP; break;
        default: assert(false && "无效的舍入模式"); break;
    }

    RawTile tile = getRawTile(trs1zt);

    if (trs1zt.dtype == TDType::SINT32) {
        LOOPIJ {
            ExactFloat val = TAcc[i][j] + intToEF(static_cast<int32_t>(tile[i][j]));
            tile[i][j] = static_cast<uint32_t>(intFromEF<int32_t>(val, rounding_mode, sat));
        }
    } else if (trs1zt.dtype == TDType::UINT32) {
        LOOPIJ {
            ExactFloat val = TAcc[i][j] + intToEF(static_cast<uint32_t>(tile[i][j]));
            tile[i][j] = static_cast<uint32_t>(intFromEF<uint32_t>(val, rounding_mode, sat));
        }
    } else if (trs1zt.dtype == TDType::F32) {
        LOOPIJ {
            ExactFloat val = TAcc[i][j] + floatToEF<8, 23>(tile[i][j]);
            tile[i][j] = static_cast<uint32_t>(floatFromEF<8, 23>(val, rounding_mode, sat));
        }
    } else if (trs1zt.dtype == TDType::F16) {
        LOOPIJ {
            ExactFloat val = TAcc[i][j] + floatToEF<5, 10>(tile[i][j]);
            tile[i][j] = static_cast<uint32_t>(floatFromEF<5, 10>(val, rounding_mode, sat));
        }
    } else if (trs1zt.dtype == TDType::SINT16) {
        LOOPIJ {
            ExactFloat val = TAcc[i][j] + intToEF(static_cast<int16_t>((tile[i][j]) & 0xFFFF));
            tile[i][j] = static_cast<uint32_t>(intFromEF<int16_t>(val, rounding_mode, sat));
        }
    } else if (trs1zt.dtype == TDType::UINT16) {
        LOOPIJ {
            ExactFloat val = TAcc[i][j] + intToEF(static_cast<uint16_t>((tile[i][j]) & 0xFFFF));
            tile[i][j] = static_cast<uint32_t>(intFromEF<uint16_t>(val, rounding_mode, sat));
        }
    } else {
        assert(false && "不支持的taccget数据类型");
    }
    
    setRawTile(trdt, tile);
}





} // namespace oootpu
