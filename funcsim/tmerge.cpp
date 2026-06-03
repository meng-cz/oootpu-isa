
#include "regfile.hpp"

namespace oootpu {

void tmerge(const TOPRand &trdt, const TOPRand &trs1zt, const TOPRand &trs2z) {
    CHECK_TREG_T(trdt);
    CHECK_TREG_ZT(trs1zt);
    CHECK_TREG_Z(trs2z);
    assert(trdt.dtype == trs1zt.dtype && trdt.dtype == trs2z.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt);
    RawTile tile2 = getRawTile(trs2z);
    RawTile result;

    auto &TMask = maskTile(1);

    LOOPIJ {
        uint32_t a = tile1[i][j];
        uint32_t b = tile2[i][j];
        result[i][j] = (TMask[i][j]) ? a : b;
    }

    setRawTile(trdt, result);
}


} 
