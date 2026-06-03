
#include "regfile.hpp"

namespace oootpu {

void tsll(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i) {
    CHECK_TREG_T(trdt_i);
    CHECK_TREG_ZT(trs1zt_i);
    CHECK_TREG_Z(trs2z_i);
    assert(trdt_i.dtype == trs1zt_i.dtype && trdt_i.dtype == trs2z_i.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt_i);
    RawTile tile2 = getRawTile(trs2z_i);
    RawTile result;

    LOOPIJ {
        uint32_t a = tile1[i][j];
        uint32_t b = tile2[i][j];
        result[i][j] = a << b;
    }

    setRawTile(trdt_i, result);
}

void tsrl(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i) {
    CHECK_TREG_T(trdt_i);
    CHECK_TREG_ZT(trs1zt_i);
    CHECK_TREG_Z(trs2z_i);
    assert(trdt_i.dtype == trs1zt_i.dtype && trdt_i.dtype == trs2z_i.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt_i);
    RawTile tile2 = getRawTile(trs2z_i);
    RawTile result;

    LOOPIJ {
        uint32_t a = tile1[i][j];
        uint32_t b = tile2[i][j];
        result[i][j] = a >> b;
    }

    setRawTile(trdt_i, result);
}

void tsra(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i) {
    CHECK_TREG_T(trdt_i);
    CHECK_TREG_ZT(trs1zt_i);
    CHECK_TREG_Z(trs2z_i);
    assert(trdt_i.dtype == trs1zt_i.dtype && trdt_i.dtype == trs2z_i.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt_i);
    RawTile tile2 = getRawTile(trs2z_i);
    RawTile result;

    LOOPIJ {
        uint32_t a = tile1[i][j];
        uint32_t b = tile2[i][j];
        if (trdt_i.dtype == TDType::SINT8) {
            result[i][j] = static_cast<uint32_t>(static_cast<int8_t>(a & 0xFF) >> b);
        } else if (trdt_i.dtype == TDType::SINT16) {
            result[i][j] = static_cast<uint32_t>(static_cast<int16_t>(a & 0xFFFF) >> b);
        } else if (trdt_i.dtype == TDType::SINT32) {
            result[i][j] = static_cast<uint32_t>(static_cast<int32_t>(a) >> b);
        } else {
            result[i][j] = a >> b;
        }
    }
}


} 
