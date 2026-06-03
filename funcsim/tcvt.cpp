
#include "regfile.hpp"

#include "res/arithm.hpp"
#include "res/exactfp.hpp"

namespace oootpu {

template <typename IntType>
inline IntType clipInt(IntType value, uint32_t max, uint32_t min) {
    using T = std::remove_cv_t<std::remove_reference_t<IntType>>;
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                  "clipInt requires a non-bool integral type");

    const T max_t = static_cast<T>(max);
    const T min_t = static_cast<T>(min);
    if (value > max_t) return max_t;
    if (value < min_t) return min_t;
    return value;
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t clipFloat(uint32_t value, uint32_t max, uint32_t min) {
    if (tpuarithm::fgt<ExpWidth, MantWidth>(value, max)) return max;
    if (tpuarithm::flt<ExpWidth, MantWidth>(value, min)) return min;
    return value;
}

void tcvt(const TOPRand &trdt, const TOPRand &trs1t, TRM rm, TOV ov, uint32_t max, uint32_t min) {
    CHECK_TREG_T(trdt);
    CHECK_TREG_T(trs1t);

    RawTile tile = getRawTile(trs1t);
    RawTile result;
    
    array<array<ExactFloat, TILE_SIZE>, TILE_SIZE> efTile;

    RoundingMode rounding_mode;
    switch (rm) {
        case TRM::RNE: rounding_mode = RoundingMode::RNE; break;
        case TRM::RTZ: rounding_mode = RoundingMode::RTZ; break;
        case TRM::RDN: rounding_mode = RoundingMode::RDN; break;
        case TRM::RUP: rounding_mode = RoundingMode::RUP; break;
        default: assert(false && "无效的舍入模式"); break;
    }
    bool saturate = (ov != TOV::DEF);

    // 先转换成ExactFloat
    if (trs1t.dtype == TDType::SINT8) {
        LOOPIJ {
            efTile[i][j] = intToEF(static_cast<int8_t>(tile[i][j] & 0xFF));
        }
    } else if (trs1t.dtype == TDType::UINT8) {
        LOOPIJ {
            efTile[i][j] = intToEF(static_cast<uint8_t>(tile[i][j] & 0xFF));
        }
    } else if (trs1t.dtype == TDType::SINT16) {
        LOOPIJ {
            efTile[i][j] = intToEF(static_cast<int16_t>(tile[i][j] & 0xFFFF));
        }
    } else if (trs1t.dtype == TDType::UINT16) {
        LOOPIJ {
            efTile[i][j] = intToEF(static_cast<uint16_t>(tile[i][j]));
        }
    } else if (trs1t.dtype == TDType::SINT32) {
        LOOPIJ {
            efTile[i][j] = intToEF(static_cast<int32_t>(tile[i][j]));
        }
    } else if (trs1t.dtype == TDType::UINT32) {
        LOOPIJ {
            efTile[i][j] = intToEF(tile[i][j]);
        }
    } else if (trs1t.dtype == TDType::F8E4M3) {
        LOOPIJ {
            efTile[i][j] = floatToEF<4, 3>(tile[i][j]);
        }
    } else if (trs1t.dtype == TDType::F8E5M2) {
        LOOPIJ {
            efTile[i][j] = floatToEF<5, 2>(tile[i][j]);
        }
    } else if (trs1t.dtype == TDType::F16) {
        LOOPIJ {
            efTile[i][j] = floatToEF<5, 10>(tile[i][j]);
        }
    } else if (trs1t.dtype == TDType::F32) {
        LOOPIJ {
            efTile[i][j] = floatToEF<8, 23>(tile[i][j]);
        }
    } else {
        assert(false && "不支持的tcvt输入数据类型");
    }

    if (trdt.dtype == TDType::SINT8) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(
                    clipInt<int8_t>(intFromEF<int8_t>(efTile[i][j], rounding_mode, saturate), max, min)
                );
            }
        } else {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(intFromEF<int8_t>(efTile[i][j], rounding_mode, saturate));
            }
        }
    } else if (trdt.dtype == TDType::UINT8) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                result[i][j] = clipInt<uint8_t>(intFromEF<uint8_t>(efTile[i][j], rounding_mode, saturate), max, min);
            }
        } else {
            LOOPIJ {
                result[i][j] = intFromEF<uint8_t>(efTile[i][j], rounding_mode, saturate);
            }
        }
    } else if (trdt.dtype == TDType::SINT16) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(
                    clipInt<int16_t>(intFromEF<int16_t>(efTile[i][j], rounding_mode, saturate), max, min)
                );
            }
        } else {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(intFromEF<int16_t>(efTile[i][j], rounding_mode, saturate));
            }
        }
    } else if (trdt.dtype == TDType::UINT16) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                result[i][j] = clipInt<uint16_t>(intFromEF<uint16_t>(efTile[i][j], rounding_mode, saturate), max, min);
            }
        } else {
            LOOPIJ {
                result[i][j] = intFromEF<uint16_t>(efTile[i][j], rounding_mode, saturate);
            }
        }
    } else if (trdt.dtype == TDType::SINT32) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(
                    clipInt<int32_t>(intFromEF<int32_t>(efTile[i][j], rounding_mode, saturate), max, min)
                );
            }
        } else {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(intFromEF<int32_t>(efTile[i][j], rounding_mode, saturate));
            }
        }
    } else if (trdt.dtype == TDType::UINT32) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                result[i][j] = clipInt<uint32_t>(intFromEF<uint32_t>(efTile[i][j], rounding_mode, saturate), max, min);
            }
        } else {
            LOOPIJ {
                result[i][j] = intFromEF<uint32_t>(efTile[i][j], rounding_mode, saturate);
            }
        }
    } else if (trdt.dtype == TDType::F8E4M3) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                const auto value = static_cast<uint32_t>(floatFromEF<4, 3>(efTile[i][j], rounding_mode, saturate));
                result[i][j] = static_cast<RawData>(clipFloat<4, 3>(value, max, min));
            }
        } else {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(floatFromEF<4, 3>(efTile[i][j], rounding_mode, saturate));
            }
        }
    } else if (trdt.dtype == TDType::F8E5M2) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                const auto value = static_cast<uint32_t>(floatFromEF<5, 2>(efTile[i][j], rounding_mode, saturate));
                result[i][j] = static_cast<RawData>(clipFloat<5, 2>(value, max, min));
            }
        } else {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(floatFromEF<5, 2>(efTile[i][j], rounding_mode, saturate));
            }
        }
    } else if (trdt.dtype == TDType::F16) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                const auto value = static_cast<uint32_t>(floatFromEF<5, 10>(efTile[i][j], rounding_mode, saturate));
                result[i][j] = static_cast<RawData>(clipFloat<5, 10>(value, max, min));
            }
        } else {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(floatFromEF<5, 10>(efTile[i][j], rounding_mode, saturate));
            }
        }
    } else if (trdt.dtype == TDType::F32) {
        if (ov == TOV::CLIP) {
            LOOPIJ {
                const auto value = static_cast<uint32_t>(floatFromEF<8, 23>(efTile[i][j], rounding_mode, saturate));
                result[i][j] = static_cast<RawData>(clipFloat<8, 23>(value, max, min));
            }
        } else {
            LOOPIJ {
                result[i][j] = static_cast<RawData>(floatFromEF<8, 23>(efTile[i][j], rounding_mode, saturate));
            }
        }
    } else {
        assert(false && "不支持的tcvt输出数据类型");
    }

    setRawTile(trdt, result);
}


} 
