
#include "regfile.hpp"

#include "res/approx.hpp"

namespace oootpu {

void tsqrt(const TOPRand &trdt_f, const TOPRand &trs1zt_f) {
    CHECK_TREG_T(trdt_f);
    CHECK_TREG_ZT(trs1zt_f);
    assert(trdt_f.dtype == trs1zt_f.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt_f);
    RawTile result;

    if (trdt_f.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuapprox::fsqrt_appr<4, 3>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuapprox::fsqrt_appr<5, 2>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuapprox::fsqrt_appr<5, 10>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuapprox::fsqrt_appr<8, 23>(tile1[i][j]);
        }
    } else {
        assert(false && "不支持的tsqrt数据类型");
    }

    setRawTile(trdt_f, result);
}

void trsqrt(const TOPRand &trdt_f, const TOPRand &trs1zt_f) {
    CHECK_TREG_T(trdt_f);
    CHECK_TREG_ZT(trs1zt_f);
    assert(trdt_f.dtype == trs1zt_f.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt_f);
    RawTile result;

    if (trdt_f.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuapprox::frsqrt_appr<4, 3>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuapprox::frsqrt_appr<5, 2>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuapprox::frsqrt_appr<5, 10>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuapprox::frsqrt_appr<8, 23>(tile1[i][j]);
        }
    } else {
        assert(false && "不支持的trsqrt数据类型");
    }

    setRawTile(trdt_f, result);
}

void texp(const TOPRand &trdt_f, const TOPRand &trs1zt_f) {
    CHECK_TREG_T(trdt_f);
    CHECK_TREG_ZT(trs1zt_f);
    assert(trdt_f.dtype == trs1zt_f.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt_f);
    RawTile result;

    if (trdt_f.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuapprox::fexp_appr<4, 3>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuapprox::fexp_appr<5, 2>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuapprox::fexp_appr<5, 10>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuapprox::fexp_appr<8, 23>(tile1[i][j]);
        }
    } else {
        assert(false && "不支持的texp数据类型");
    }

    setRawTile(trdt_f, result);
}

void trcp(const TOPRand &trdt_f, const TOPRand &trs1zt_f) {
    CHECK_TREG_T(trdt_f);
    CHECK_TREG_ZT(trs1zt_f);
    assert(trdt_f.dtype == trs1zt_f.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1zt_f);
    RawTile result;

    if (trdt_f.dtype == TDType::F8E4M3) {
        LOOPIJ {
            result[i][j] = tpuapprox::frcp_appr<4, 3>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F8E5M2) {
        LOOPIJ {
            result[i][j] = tpuapprox::frcp_appr<5, 2>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F16) {
        LOOPIJ {
            result[i][j] = tpuapprox::frcp_appr<5, 10>(tile1[i][j]);
        }
    } else if (trdt_f.dtype == TDType::F32) {
        LOOPIJ {
            result[i][j] = tpuapprox::frcp_appr<8, 23>(tile1[i][j]);
        }
    } else {
        assert(false && "不支持的trcp数据类型");
    }

    setRawTile(trdt_f, result);
}



} 


