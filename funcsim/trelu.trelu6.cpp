
#include "regfile.hpp"

#include "res/arithm.hpp"

namespace oootpu {

void trelu(const TOPRand &trdt, const TOPRand &trs1zt) {
    CHECK_TREG_ZT(trdt);
    CHECK_TREG_ZT(trs1zt);
    assert(trdt.dtype == trs1zt.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile result;

    if (trdt.dtype == TDType::SINT8 || trdt.dtype == TDType::UINT8) {
        LOOPIJ {
            int8_t a = static_cast<int8_t>(tile1[i][j] & 0xFF);
            result[i][j] = static_cast<uint32_t>(a < 0 ? 0 : a);
        }
    } else if (trdt.dtype == TDType::SINT16 || trdt.dtype == TDType::UINT16) {
        LOOPIJ {
            int16_t a = static_cast<int16_t>(tile1[i][j] & 0xFFFF);
            result[i][j] = static_cast<uint32_t>(a < 0 ? 0 : a);
        }
    } else if (trdt.dtype == TDType::SINT32 || trdt.dtype == TDType::UINT32) {
        LOOPIJ {
            int32_t a = static_cast<int32_t>(tile1[i][j]);
            result[i][j] = static_cast<RawData>(a < 0 ? 0 : a);
        }
    } else if (trdt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<4, 3>((tile1[i][j]), 0) ? tile1[i][j] : 0;
        }
    } else if (trdt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<5, 2>((tile1[i][j]), 0) ? tile1[i][j] : 0;
        }
    } else if (trdt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<5, 10>((tile1[i][j]), 0) ? tile1[i][j] : 0;
        }
    } else if (trdt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<8, 23>((tile1[i][j]), 0) ? tile1[i][j] : 0;
        }
    } else {
        assert(false && "不支持的trelu数据类型");
    }

    setRawTile(trdt, result);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline constexpr uint32_t floatEnc6() {
    constexpr uint32_t bias = (1u << (ExpWidth - 1)) - 1u;
    constexpr uint32_t exp = 2u + bias;
    constexpr uint32_t mant = MantWidth == 0 ? 0u : (1u << (MantWidth - 1));
    return (exp << MantWidth) | mant;
}

void trelu6(const TOPRand &trdt, const TOPRand &trs1zt) {
    CHECK_TREG_ZT(trdt);
    CHECK_TREG_ZT(trs1zt);
    assert(trdt.dtype == trs1zt.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile result;

    if (trdt.dtype == TDType::SINT8 || trdt.dtype == TDType::UINT8) {
        LOOPIJ {
            int8_t a = static_cast<int8_t>(tile1[i][j] & 0xFF);
            result[i][j] = static_cast<uint32_t>(a < 0 ? 0 : (a > 6 ? 6 : a));
        }
    } else if (trdt.dtype == TDType::SINT16 || trdt.dtype == TDType::UINT16) {
        LOOPIJ {
            int16_t a = static_cast<int16_t>(tile1[i][j] & 0xFFFF);
            result[i][j] = static_cast<uint32_t>(a < 0 ? 0 : (a > 6 ? 6 : a));
        }
    } else if (trdt.dtype == TDType::SINT32 || trdt.dtype == TDType::UINT32) {
        LOOPIJ {
            int32_t a = static_cast<int32_t>(tile1[i][j]);
            result[i][j] = static_cast<RawData>(a < 0 ? 0 : (a > 6 ? 6 : a));
        }
    } else if (trdt.dtype == TDType::F8E4M3) {
        constexpr uint32_t six = floatEnc6<4, 3>();
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<4, 3>((tile1[i][j]), six) ? six : ((tpuarithm::fgt<4, 3>((tile1[i][j]), 0) ? tile1[i][j] : 0));
        }
    } else if (trdt.dtype == TDType::F8E5M2) {
        constexpr uint32_t six = floatEnc6<5, 2>();
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<5, 2>((tile1[i][j]), six) ? six : ((tpuarithm::fgt<5, 2>((tile1[i][j]), 0) ? tile1[i][j] : 0));
        }
    } else if (trdt.dtype == TDType::F16) {
        constexpr uint32_t six = floatEnc6<5, 10>();
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<5, 10>((tile1[i][j]), six) ? six : ((tpuarithm::fgt<5, 10>((tile1[i][j]), 0) ? tile1[i][j] : 0));
        }
    } else if (trdt.dtype == TDType::F32) {
        constexpr uint32_t six = floatEnc6<8, 23>();
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<8, 23>((tile1[i][j]), six) ? six : ((tpuarithm::fgt<8, 23>((tile1[i][j]), 0) ? tile1[i][j] : 0));
        }
    } else {
        assert(false && "不支持的trelu6数据类型");
    }

    setRawTile(trdt, result);
}



} 
