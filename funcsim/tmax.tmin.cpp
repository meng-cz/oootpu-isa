
#include "regfile.hpp"

#include "res/arithm.hpp"
#include "res/exactfp.hpp"

namespace oootpu {


void tmax(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_ZT(trdt);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trdt.dtype == trs1zt.dtype && trdt.dtype == trs2z.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (trdt.dtype == TDType::SINT8) {
        LOOPIJ {
            int8_t a = static_cast<int8_t>(tile1[i][j] & 0xFF);
            int8_t b = static_cast<int8_t>(tile2[i][j] & 0xFF);
            result[i][j] = static_cast<uint32_t>(static_cast<uint8_t>(a > b ? a : b));
        }
    } else if (trdt.dtype == TDType::UINT8) {
        LOOPIJ {
            uint8_t a = static_cast<uint8_t>(tile1[i][j]);
            uint8_t b = static_cast<uint8_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(a > b ? a : b);
        }
    } else if (trdt.dtype == TDType::SINT16) {
        LOOPIJ {
            int16_t a = static_cast<int16_t>(tile1[i][j] & 0xFFFF);
            int16_t b = static_cast<int16_t>(tile2[i][j] & 0xFFFF);
            result[i][j] = static_cast<uint32_t>(static_cast<uint16_t>(a > b ? a : b));
        }
    } else if (trdt.dtype == TDType::UINT16) {
        LOOPIJ {
            uint16_t a = static_cast<uint16_t>(tile1[i][j]);
            uint16_t b = static_cast<uint16_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(a > b ? a : b);
        }
    } else if (trdt.dtype == TDType::SINT32) {
        LOOPIJ {
            int32_t a = static_cast<int32_t>(tile1[i][j]);
            int32_t b = static_cast<int32_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(a > b ? a : b);
        }
    } else if (trdt.dtype == TDType::UINT32) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = a > b ? a : b;
        }
    } else if (trdt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = tpuarithm::fgt<4, 3>(a, b) ? a : b;
        }
    } else if (trdt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = tpuarithm::fgt<5, 2>(a, b) ? a : b;
        }
    } else if (trdt.dtype == TDType::F16) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = tpuarithm::fgt<5, 10>(a, b) ? a : b;
        }
    } else if (trdt.dtype == TDType::F32) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = tpuarithm::fgt<8, 23>(a, b) ? a : b;
        }
    } else {
        assert(false && "不支持的tmax数据类型");
    }

    setRawTile(trdt, result);
}

void tmin(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_ZT(trdt);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trdt.dtype == trs1zt.dtype && trdt.dtype == trs2z.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (trdt.dtype == TDType::SINT8) {
        LOOPIJ {
            int8_t a = static_cast<int8_t>(tile1[i][j] & 0xFF);
            int8_t b = static_cast<int8_t>(tile2[i][j] & 0xFF);
            result[i][j] = static_cast<uint32_t>(static_cast<uint8_t>(a < b ? a : b));
        }
    } else if (trdt.dtype == TDType::UINT8) {
        LOOPIJ {
            uint8_t a = static_cast<uint8_t>(tile1[i][j]);
            uint8_t b = static_cast<uint8_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(a < b ? a : b);
        }
    } else if (trdt.dtype == TDType::SINT16) {
        LOOPIJ {
            int16_t a = static_cast<int16_t>(tile1[i][j] & 0xFFFF);
            int16_t b = static_cast<int16_t>(tile2[i][j] & 0xFFFF);
            result[i][j] = static_cast<uint32_t>(static_cast<uint16_t>(a < b ? a : b));
        }
    } else if (trdt.dtype == TDType::UINT16) {
        LOOPIJ {
            uint16_t a = static_cast<uint16_t>(tile1[i][j]);
            uint16_t b = static_cast<uint16_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(a < b ? a : b);
        }
    } else if (trdt.dtype == TDType::SINT32) {
        LOOPIJ {
            int32_t a = static_cast<int32_t>(tile1[i][j]);
            int32_t b = static_cast<int32_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(a < b ? a : b);
        }
    } else if (trdt.dtype == TDType::UINT32) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = a < b ? a : b;
        }
    } else if (trdt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = tpuarithm::flt<4, 3>(a, b) ? a : b;
        }
    } else if (trdt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = tpuarithm::flt<5, 2>(a, b) ? a : b;
        }
    } else if (trdt.dtype == TDType::F16) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = tpuarithm::flt<5, 10>(a, b) ? a : b;
        }
    } else if (trdt.dtype == TDType::F32) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = tpuarithm::flt<8, 23>(a, b) ? a : b;
        }
    } else {
        assert(false && "不支持的tmin数据类型");
    }

    setRawTile(trdt, result);
}



} // namespace oootpu
