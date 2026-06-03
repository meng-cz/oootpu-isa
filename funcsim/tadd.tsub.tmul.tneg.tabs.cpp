
#include "regfile.hpp"

#include "res/arithm.hpp"
#include "res/exactfp.hpp"

namespace oootpu {

void tadd(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_ZT(trdt);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trdt.dtype == trs1zt.dtype && trdt.dtype == trs2z.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (trdt.dtype == TDType::SINT8 || trdt.dtype == TDType::UINT8) {
        LOOPIJ {
            uint8_t a = static_cast<uint8_t>(tile1[i][j]);
            uint8_t b = static_cast<uint8_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(static_cast<uint8_t>(a + b));
        }
    } else if (trdt.dtype == TDType::SINT16 || trdt.dtype == TDType::UINT16) {
        LOOPIJ {
            uint16_t a = static_cast<uint16_t>(tile1[i][j]);
            uint16_t b = static_cast<uint16_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(static_cast<uint16_t>(a + b));
        }
    } else if (trdt.dtype == TDType::SINT32 || trdt.dtype == TDType::UINT32) {
        LOOPIJ {
            result[i][j] = tile1[i][j] + tile2[i][j];
        }
    } else if (trdt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::fadd_rne<4, 3>((tile1[i][j]), (tile2[i][j]));
        }
    } else if (trdt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::fadd_rne<5, 2>((tile1[i][j]), (tile2[i][j]));
        }
    } else if (trdt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::fadd_rne<5, 10>((tile1[i][j]), (tile2[i][j]));
        }
    } else if (trdt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::fadd_rne<8, 23>(tile1[i][j], tile2[i][j]);
        }
    } else {
        assert(false && "不支持的tadd数据类型");
    }

    setRawTile(trdt, result);
}

void tsub(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_ZT(trdt);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trdt.dtype == trs1zt.dtype && trdt.dtype == trs2z.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (trdt.dtype == TDType::SINT8 || trdt.dtype == TDType::UINT8) {
        LOOPIJ {
            uint8_t a = static_cast<uint8_t>(tile1[i][j]);
            uint8_t b = static_cast<uint8_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(static_cast<uint8_t>(a - b));
        }
    } else if (trdt.dtype == TDType::SINT16 || trdt.dtype == TDType::UINT16) {
        LOOPIJ {
            uint16_t a = static_cast<uint16_t>(tile1[i][j]);
            uint16_t b = static_cast<uint16_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(static_cast<uint16_t>(a - b));
        }
    } else if (trdt.dtype == TDType::SINT32 || trdt.dtype == TDType::UINT32) {
        LOOPIJ {
            result[i][j] = tile1[i][j] - tile2[i][j];
        }
    } else if (trdt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::fsub_rne<4, 3>((tile1[i][j]), (tile2[i][j]));
        }
    } else if (trdt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::fsub_rne<5, 2>((tile1[i][j]), (tile2[i][j]));
        }
    } else if (trdt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::fsub_rne<5, 10>((tile1[i][j]), (tile2[i][j]));
        }
    } else if (trdt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::fsub_rne<8, 23>(tile1[i][j], tile2[i][j]);
        }
    } else {
        assert(false && "不支持的tsub数据类型");
    }

    setRawTile(trdt, result);
}


void tmul(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_ZT(trdt);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trs1zt.dtype == trs2z.dtype && "输入矩阵数据类型必须相同");
    assert((trdt.dtype == trs1zt.dtype || 
        (trs1zt.dtype == TDType::SINT8 && trdt.dtype == TDType::SINT16) ||
        (trs1zt.dtype == TDType::UINT8 && trdt.dtype == TDType::UINT16) ||
        (trs1zt.dtype == TDType::SINT16 && trdt.dtype == TDType::SINT32) ||
        (trs1zt.dtype == TDType::UINT16 && trdt.dtype == TDType::UINT32) ||
        (trs1zt.dtype == TDType::F8E4M3 && trdt.dtype == TDType::F16) ||
        (trs1zt.dtype == TDType::F8E5M2 && trdt.dtype == TDType::F16) ||
        (trs1zt.dtype == TDType::F16 && trdt.dtype == TDType::F32)
    ) && "输出数据类型必须与输入矩阵数据类型相同或为输入类型两倍宽度");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    if (trs1zt.dtype == TDType::SINT8) {
        if (trdt.dtype == TDType::SINT8) {
            LOOPIJ {
                int16_t a = static_cast<int8_t>(tile1[i][j] & 0xFF);
                int16_t b = static_cast<int8_t>(tile2[i][j] & 0xFF);
                result[i][j] = static_cast<uint32_t>(static_cast<uint8_t>(a * b));
            }
        } else if (trdt.dtype == TDType::SINT16) {
            LOOPIJ {
                int16_t a = static_cast<int8_t>(tile1[i][j] & 0xFF);
                int16_t b = static_cast<int8_t>(tile2[i][j] & 0xFF);
                result[i][j] = static_cast<uint32_t>(static_cast<uint16_t>(a * b));
            }
        } else {
            assert(false && "不支持的tmul输出数据类型");
        }
    } else if (trs1zt.dtype == TDType::UINT8) {
        if (trdt.dtype == TDType::UINT8) {
            LOOPIJ {
                uint16_t a = static_cast<uint8_t>(tile1[i][j]);
                uint16_t b = static_cast<uint8_t>(tile2[i][j]);
                result[i][j] = static_cast<uint32_t>(static_cast<uint8_t>(a * b));
            }
        } else if (trdt.dtype == TDType::UINT16) {
            LOOPIJ {
                uint16_t a = static_cast<uint8_t>(tile1[i][j]);
                uint16_t b = static_cast<uint8_t>(tile2[i][j]);
                result[i][j] = static_cast<uint32_t>(static_cast<uint16_t>(a * b));
            }
        } else {
            assert(false && "不支持的tmul输出数据类型");
        }
    } else if (trs1zt.dtype == TDType::SINT16) {
        if (trdt.dtype == TDType::SINT16) {
            LOOPIJ {
                int32_t a = static_cast<int16_t>(tile1[i][j] & 0xFFFF);
                int32_t b = static_cast<int16_t>(tile2[i][j] & 0xFFFF);
                result[i][j] = static_cast<uint32_t>(static_cast<uint16_t>(a * b));
            }
        } else if (trdt.dtype == TDType::SINT32) {
            LOOPIJ {
                int32_t a = static_cast<int16_t>(tile1[i][j] & 0xFFFF);
                int32_t b = static_cast<int16_t>(tile2[i][j] & 0xFFFF);
                result[i][j] = static_cast<uint32_t>(a * b);
            }
        } else {
            assert(false && "不支持的tmul输出数据类型");
        }
    } else if (trs1zt.dtype == TDType::UINT16) {
        if (trdt.dtype == TDType::UINT16) {
            LOOPIJ {
                uint32_t a = static_cast<uint16_t>(tile1[i][j]);
                uint32_t b = static_cast<uint16_t>(tile2[i][j]);
                result[i][j] = static_cast<uint32_t>(static_cast<uint16_t>(a * b));
            }
        } else if (trdt.dtype == TDType::UINT32) {
            LOOPIJ {
                uint32_t a = static_cast<uint16_t>(tile1[i][j]);
                uint32_t b = static_cast<uint16_t>(tile2[i][j]);
                result[i][j] = a * b;
            }
        } else {
            assert(false && "不支持的tmul输出数据类型");
        }
    } else if (trs1zt.dtype == TDType::SINT32) {
        LOOPIJ {
            int64_t a = static_cast<int32_t>(tile1[i][j]);
            int64_t b = static_cast<int32_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(a * b);
        }
    } else if (trs1zt.dtype == TDType::UINT32) {
        LOOPIJ {
            uint64_t a = static_cast<uint32_t>(tile1[i][j]);
            uint64_t b = static_cast<uint32_t>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(a * b);
        }
    } else if (trs1zt.dtype == TDType::F8E4M3) {
        if (trdt.dtype == TDType::F8E4M3) {
            LOOPIJ {
                ExactFloat a = floatToEF<4, 3>(tile1[i][j]);
                ExactFloat b = floatToEF<4, 3>(tile2[i][j]);
                result[i][j] = static_cast<uint32_t>(floatFromEF<4, 3>(a * b, RoundingMode::RNE, false));
            }
        } else if (trdt.dtype == TDType::F16) {
            LOOPIJ {
                ExactFloat a = floatToEF<4, 3>(tile1[i][j]);
                ExactFloat b = floatToEF<4, 3>(tile2[i][j]);
                result[i][j] = static_cast<uint32_t>(floatFromEF<5, 10>(a * b, RoundingMode::RNE, false));
            }
        } else {
            assert(false && "不支持的tmul输出数据类型");
        }
    } else if (trs1zt.dtype == TDType::F8E5M2) {
        if (trdt.dtype == TDType::F8E5M2) {
            LOOPIJ {
                ExactFloat a = floatToEF<5, 2>(tile1[i][j]);
                ExactFloat b = floatToEF<5, 2>(tile2[i][j]);
                result[i][j] = static_cast<uint32_t>(floatFromEF<5, 2>(a * b, RoundingMode::RNE, false));
            }
        } else if (trdt.dtype == TDType::F16) {
            LOOPIJ {
                ExactFloat a = floatToEF<5, 2>(tile1[i][j]);
                ExactFloat b = floatToEF<5, 2>(tile2[i][j]);
                result[i][j] = static_cast<uint32_t>(floatFromEF<5, 10>(a * b, RoundingMode::RNE, false));
            }
        } else {
            assert(false && "不支持的tmul输出数据类型");
        }
    } else if (trs1zt.dtype == TDType::F16) {
        if (trdt.dtype == TDType::F16) {
            LOOPIJ {
                ExactFloat a = floatToEF<5, 10>(tile1[i][j]);
                ExactFloat b = floatToEF<5, 10>(tile2[i][j]);
                result[i][j] = static_cast<uint32_t>(floatFromEF<5, 10>(a * b, RoundingMode::RNE, false));
            }
        } else if (trdt.dtype == TDType::F32) {
            LOOPIJ {
                ExactFloat a = floatToEF<5, 10>(tile1[i][j]);
                ExactFloat b = floatToEF<5, 10>(tile2[i][j]);
                result[i][j] = static_cast<uint32_t>(floatFromEF<8, 23>(a * b, RoundingMode::RNE, false));
            }
        } else {
            assert(false && "不支持的tmul输出数据类型");
        }
    } else if (trs1zt.dtype == TDType::F32) {
        LOOPIJ {
            ExactFloat a = floatToEF<8, 23>(tile1[i][j]);
            ExactFloat b = floatToEF<8, 23>(tile2[i][j]);
            result[i][j] = static_cast<uint32_t>(floatFromEF<8, 23>(a * b, RoundingMode::RNE, false));
        }
    } else {
        assert(false && "不支持的tmul数据类型");
    }

    setRawTile(trdt, result);
}

void tneg(const TOPRand &trdt, const TOPRand &trs1zt) {
    CHECK_TREG_ZT(trdt);
    CHECK_TREG_ZT(trs1zt);
    assert(trdt.dtype == trs1zt.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile result;

    if (trdt.dtype == TDType::SINT8 || trdt.dtype == TDType::UINT8) {
        LOOPIJ {
            uint8_t a = static_cast<uint8_t>(tile1[i][j]);
            result[i][j] = static_cast<uint32_t>(static_cast<uint8_t>(-a));
        }
    } else if (trdt.dtype == TDType::SINT16 || trdt.dtype == TDType::UINT16) {
        LOOPIJ {
            uint16_t a = static_cast<uint16_t>(tile1[i][j]);
            result[i][j] = static_cast<uint32_t>(static_cast<uint16_t>(-a));
        }
    } else if (trdt.dtype == TDType::SINT32 || trdt.dtype == TDType::UINT32) {
        LOOPIJ {
            result[i][j] = -tile1[i][j];
        }
    } else if (trdt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::fneg<4, 3>((tile1[i][j]));
        }
    } else if (trdt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::fneg<5, 2>((tile1[i][j]));
        }
    } else if (trdt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::fneg<5, 10>((tile1[i][j]));
        }
    } else if (trdt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::fneg<8, 23>(tile1[i][j]);
        }
    } else {
        assert(false && "不支持的tabs数据类型");
    }

    setRawTile(trdt, result);
}

void tabs(const TOPRand &trdt, const TOPRand &trs1zt) {
    CHECK_TREG_ZT(trdt);
    CHECK_TREG_ZT(trs1zt);
    assert(trdt.dtype == trs1zt.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile result;

    if (trdt.dtype == TDType::SINT8 || trdt.dtype == TDType::UINT8) {
        LOOPIJ {
            int8_t a = static_cast<int8_t>(tile1[i][j] & 0xFF);
            result[i][j] = static_cast<int32_t>(a < 0 ? -a : a);
        }
    } else if (trdt.dtype == TDType::SINT16 || trdt.dtype == TDType::UINT16) {
        LOOPIJ {
            int16_t a = static_cast<int16_t>(tile1[i][j] & 0xFFFF);
            result[i][j] = static_cast<int32_t>(a < 0 ? -a : a);
        }
    } else if (trdt.dtype == TDType::SINT32 || trdt.dtype == TDType::UINT32) {
        LOOPIJ {
            int32_t a = static_cast<int32_t>(tile1[i][j]);
            result[i][j] = static_cast<RawData>(a < 0 ? -a : a);
        }
    } else if (trdt.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuarithm::fabs<4, 3>((tile1[i][j]));
        }
    } else if (trdt.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuarithm::fabs<5, 2>((tile1[i][j]));
        }
    } else if (trdt.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuarithm::fabs<5, 10>((tile1[i][j]));
        }
    } else if (trdt.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuarithm::fabs<8, 23>(tile1[i][j]);
        }
    } else {
        assert(false && "不支持的tabs数据类型");
    }

    setRawTile(trdt, result);
}


} // namespace oootpu
