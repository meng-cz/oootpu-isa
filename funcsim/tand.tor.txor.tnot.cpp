
#include "regfile.hpp"

namespace oootpu {

void tand(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i) {
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
        result[i][j] = a & b;
    }

    setRawTile(trdt_i, result);
}

void tor(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i) {
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
        result[i][j] = a | b;
    }

    setRawTile(trdt_i, result);
}

void txor(const TOPRand &trdt_i, const TOPRand &trs1zt_i, const TOPRand &trs2z_i) {
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
        result[i][j] = a ^ b;
    }
}

void tnot(const TOPRand &trdt_i, const TOPRand &trs1zt_i) {
    CHECK_TREG_T(trdt_i);
    CHECK_TREG_ZT(trs1zt_i);
    assert(trdt_i.dtype == trs1zt_i.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt_i);
    RawTile result;

    LOOPIJ {
        uint32_t a = tile1[i][j];
        result[i][j] = ~a;
    }

    setRawTile(trdt_i, result);
}


} // namespace oootpu
