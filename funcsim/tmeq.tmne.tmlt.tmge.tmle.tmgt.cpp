
#include "regfile.hpp"

#include "res/arithm.hpp"

namespace oootpu {

void tmeq(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trs1zt.dtype == trs2z.dtype && "输入矩阵数据类型必须相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (dtypeIsInt(trs1zt.dtype)) {
        uint32_t width = dtypeWidth(trs1zt.dtype);
        if (width == 1) {
            LOOPIJ {
                uint8_t a = static_cast<uint8_t>(tile1[i][j]);
                uint8_t b = static_cast<uint8_t>(tile2[i][j]);
                result[i][j] = (a == b) ? 1 : 0;
            }
        } else if (width == 2) {
            LOOPIJ {
                uint16_t a = static_cast<uint16_t>(tile1[i][j]);
                uint16_t b = static_cast<uint16_t>(tile2[i][j]);
                result[i][j] = (a == b) ? 1 : 0;
            }
        } else if (width == 4) {
            LOOPIJ {
                result[i][j] = (tile1[i][j] == tile2[i][j]) ? 1 : 0;
            }
        } else {
            assert(false && "不支持的tmeq整数数据类型");
        }
    } else if (trs1zt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::feq<4, 3>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::feq<5, 2>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::feq<5, 10>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::feq<8, 23>(tile1[i][j], tile2[i][j]) ? 1 : 0;
        }
    } else {
        assert(false && "不支持的tmeq数据类型");
    }

    setRawTile(trdt_m8, result);
}

void tmne(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trs1zt.dtype == trs2z.dtype && "输入矩阵数据类型必须相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (dtypeIsInt(trs1zt.dtype)) {
        uint32_t width = dtypeWidth(trs1zt.dtype);
        if (width == 1) {
            LOOPIJ {
                uint8_t a = static_cast<uint8_t>(tile1[i][j]);
                uint8_t b = static_cast<uint8_t>(tile2[i][j]);
                result[i][j] = (a == b) ? 0 : 1;
            }
        } else if (width == 2) {
            LOOPIJ {
                uint16_t a = static_cast<uint16_t>(tile1[i][j]);
                uint16_t b = static_cast<uint16_t>(tile2[i][j]);
                result[i][j] = (a == b) ? 0 : 1;
            }
        } else if (width == 4) {
            LOOPIJ {
                result[i][j] = (tile1[i][j] == tile2[i][j]) ? 0 : 1;
            }
        } else {
            assert(false && "不支持的tmne整数数据类型");
        }
    } else if (trs1zt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::fne<4, 3>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::fne<5, 2>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::fne<5, 10>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::fne<8, 23>(tile1[i][j], tile2[i][j]) ? 1 : 0;
        }
    } else {
        assert(false && "不支持的tmne数据类型");
    }

    setRawTile(trdt_m8, result);
}

void tmlt(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trs1zt.dtype == trs2z.dtype && "输入矩阵数据类型必须相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (trs1zt.dtype == TDType::SINT8) {
        LOOPIJ {
            int8_t a = static_cast<int8_t>(tile1[i][j]);
            int8_t b = static_cast<int8_t>(tile2[i][j]);
            result[i][j] = (a < b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT8) {
        LOOPIJ {
            uint8_t a = static_cast<uint8_t>(tile1[i][j]);
            uint8_t b = static_cast<uint8_t>(tile2[i][j]);
            result[i][j] = (a < b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::SINT16) {
        LOOPIJ {
            int16_t a = static_cast<int16_t>(tile1[i][j]);
            int16_t b = static_cast<int16_t>(tile2[i][j]);
            result[i][j] = (a < b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT16) {
        LOOPIJ {
            uint16_t a = static_cast<uint16_t>(tile1[i][j]);
            uint16_t b = static_cast<uint16_t>(tile2[i][j]);
            result[i][j] = (a < b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::SINT32) {
        LOOPIJ {
            int32_t a = static_cast<int32_t>(tile1[i][j]);
            int32_t b = static_cast<int32_t>(tile2[i][j]);
            result[i][j] = (a < b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT32) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = (a < b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::flt<4, 3>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::flt<5, 2>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::flt<5, 10>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::flt<8, 23>(tile1[i][j], tile2[i][j]) ? 1 : 0;
        }
    } else {
        assert(false && "不支持的tmlt数据类型");
    }

    setRawTile(trdt_m8, result);
}

void tmge(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trs1zt.dtype == trs2z.dtype && "输入矩阵数据类型必须相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (trs1zt.dtype == TDType::SINT8) {
        LOOPIJ {
            int8_t a = static_cast<int8_t>(tile1[i][j]);
            int8_t b = static_cast<int8_t>(tile2[i][j]);
            result[i][j] = (a >= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT8) {
        LOOPIJ {
            uint8_t a = static_cast<uint8_t>(tile1[i][j]);
            uint8_t b = static_cast<uint8_t>(tile2[i][j]);
            result[i][j] = (a >= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::SINT16) {
        LOOPIJ {
            int16_t a = static_cast<int16_t>(tile1[i][j]);
            int16_t b = static_cast<int16_t>(tile2[i][j]);
            result[i][j] = (a >= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT16) {
        LOOPIJ {
            uint16_t a = static_cast<uint16_t>(tile1[i][j]);
            uint16_t b = static_cast<uint16_t>(tile2[i][j]);
            result[i][j] = (a >= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::SINT32) {
        LOOPIJ {
            int32_t a = static_cast<int32_t>(tile1[i][j]);
            int32_t b = static_cast<int32_t>(tile2[i][j]);
            result[i][j] = (a >= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT32) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = (a >= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::fge<4, 3>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::fge<5, 2>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::fge<5, 10>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::fge<8, 23>(tile1[i][j], tile2[i][j]) ? 1 : 0;
        }
    } else {
        assert(false && "不支持的tmge数据类型");
    }

    setRawTile(trdt_m8, result);
}

void tmle(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trs1zt.dtype == trs2z.dtype && "输入矩阵数据类型必须相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (trs1zt.dtype == TDType::SINT8) {
        LOOPIJ {
            int8_t a = static_cast<int8_t>(tile1[i][j]);
            int8_t b = static_cast<int8_t>(tile2[i][j]);
            result[i][j] = (a <= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT8) {
        LOOPIJ {
            uint8_t a = static_cast<uint8_t>(tile1[i][j]);
            uint8_t b = static_cast<uint8_t>(tile2[i][j]);
            result[i][j] = (a <= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::SINT16) {
        LOOPIJ {
            int16_t a = static_cast<int16_t>(tile1[i][j]);
            int16_t b = static_cast<int16_t>(tile2[i][j]);
            result[i][j] = (a <= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT16) {
        LOOPIJ {
            uint16_t a = static_cast<uint16_t>(tile1[i][j]);
            uint16_t b = static_cast<uint16_t>(tile2[i][j]);
            result[i][j] = (a <= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::SINT32) {
        LOOPIJ {
            int32_t a = static_cast<int32_t>(tile1[i][j]);
            int32_t b = static_cast<int32_t>(tile2[i][j]);
            result[i][j] = (a <= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT32) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = (a <= b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::fle<4, 3>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::fle<5, 2>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::fle<5, 10>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::fle<8, 23>(tile1[i][j], tile2[i][j]) ? 1 : 0;
        }
    } else {
        assert(false && "不支持的tmle数据类型");
    }

    setRawTile(trdt_m8, result);
}

void tmgt(const TOPRand &trdt_m8, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trs1zt.dtype == trs2z.dtype && "输入矩阵数据类型必须相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (trs1zt.dtype == TDType::SINT8) {
        LOOPIJ {
            int8_t a = static_cast<int8_t>(tile1[i][j]);
            int8_t b = static_cast<int8_t>(tile2[i][j]);
            result[i][j] = (a > b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT8) {
        LOOPIJ {
            uint8_t a = static_cast<uint8_t>(tile1[i][j]);
            uint8_t b = static_cast<uint8_t>(tile2[i][j]);
            result[i][j] = (a > b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::SINT16) {
        LOOPIJ {
            int16_t a = static_cast<int16_t>(tile1[i][j]);
            int16_t b = static_cast<int16_t>(tile2[i][j]);
            result[i][j] = (a > b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT16) {
        LOOPIJ {
            uint16_t a = static_cast<uint16_t>(tile1[i][j]);
            uint16_t b = static_cast<uint16_t>(tile2[i][j]);
            result[i][j] = (a > b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::SINT32) {
        LOOPIJ {
            int32_t a = static_cast<int32_t>(tile1[i][j]);
            int32_t b = static_cast<int32_t>(tile2[i][j]);
            result[i][j] = (a > b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::UINT32) {
        LOOPIJ {
            uint32_t a = tile1[i][j];
            uint32_t b = tile2[i][j];
            result[i][j] = (a > b) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<4, 3>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<5, 2>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<5, 10>((tile1[i][j]), (tile2[i][j])) ? 1 : 0;
        }
    } else if (trs1zt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::fgt<8, 23>(tile1[i][j], tile2[i][j]) ? 1 : 0;
        }
    } else {
        assert(false && "不支持的tmgt数据类型");
    }

    setRawTile(trdt_m8, result);
}


}
