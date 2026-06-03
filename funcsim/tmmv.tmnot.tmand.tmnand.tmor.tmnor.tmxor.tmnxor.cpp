
#include "regfile.hpp"

namespace oootpu {

void tmmv(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZTM8(trs1zt_m8);

    RawTile tile1 = getRawTile(trs1zt_m8);
    RawTile result;

    LOOPIJ {
        result[i][j] = (tile1[i][j] != 0) ? 1 : 0;
    }

    setRawTile(trdt_m8, result);
}

void tmnot(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZTM8(trs1zt_m8);

    RawTile tile1 = getRawTile(trs1zt_m8);
    RawTile result;

    LOOPIJ {
        result[i][j] = (tile1[i][j] != 0) ? 0 : 1;
    }

    setRawTile(trdt_m8, result);
}

void tmand(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZTM8(trs1zt_m8);
    CHECK_TREG_ZM8(trs2z_m8);

    RawTile tile1 = getRawTile(trs1zt_m8);
    RawTile tile2 = getRawTile(trs2z_m8);
    RawTile result;

    LOOPIJ {
        result[i][j] = (tile1[i][j] != 0 && tile2[i][j] != 0) ? 1 : 0;
    }

    setRawTile(trdt_m8, result);
}

void tmnand(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZTM8(trs1zt_m8);
    CHECK_TREG_ZM8(trs2z_m8);

    RawTile tile1 = getRawTile(trs1zt_m8);
    RawTile tile2 = getRawTile(trs2z_m8);
    RawTile result;

    LOOPIJ {
        result[i][j] = (tile1[i][j] != 0 && tile2[i][j] != 0) ? 0 : 1;
    }

    setRawTile(trdt_m8, result);
}

void tmor(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZTM8(trs1zt_m8);
    CHECK_TREG_ZM8(trs2z_m8);

    RawTile tile1 = getRawTile(trs1zt_m8);
    RawTile tile2 = getRawTile(trs2z_m8);
    RawTile result;

    LOOPIJ {
        result[i][j] = (tile1[i][j] != 0 || tile2[i][j] != 0) ? 1 : 0;
    }

    setRawTile(trdt_m8, result);
}

void tmnor(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZTM8(trs1zt_m8);
    CHECK_TREG_ZM8(trs2z_m8);

    RawTile tile1 = getRawTile(trs1zt_m8);
    RawTile tile2 = getRawTile(trs2z_m8);
    RawTile result;

    LOOPIJ {
        result[i][j] = (tile1[i][j] != 0 || tile2[i][j] != 0) ? 0 : 1;
    }

    setRawTile(trdt_m8, result);
}

void tmxor(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZTM8(trs1zt_m8);
    CHECK_TREG_ZM8(trs2z_m8);

    RawTile tile1 = getRawTile(trs1zt_m8);
    RawTile tile2 = getRawTile(trs2z_m8);
    RawTile result;

    LOOPIJ {
        result[i][j] = ((tile1[i][j] != 0) == (tile2[i][j] != 0)) ? 0 : 1;
    }

    setRawTile(trdt_m8, result);
}

void tmnxor(const TOPRand &trdt_m8, const TOPRand &trs1zt_m8, const TOPRand &trs2z_m8) {
    CHECK_TREG_TM8(trdt_m8);
    CHECK_TREG_ZTM8(trs1zt_m8);
    CHECK_TREG_ZM8(trs2z_m8);

    RawTile tile1 = getRawTile(trs1zt_m8);
    RawTile tile2 = getRawTile(trs2z_m8);
    RawTile result;

    LOOPIJ {
        result[i][j] = ((tile1[i][j] != 0) == (tile2[i][j] != 0)) ? 1 : 0;
    }

    setRawTile(trdt_m8, result);
}



}
