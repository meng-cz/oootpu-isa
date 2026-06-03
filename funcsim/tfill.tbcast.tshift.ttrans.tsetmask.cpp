
#include "regfile.hpp"

#include <cstddef>

namespace oootpu {

void tfill(const TOPRand &trd, uint32_t value) {
    CHECK_TREG_NORM(trd);

    RawTile result;
    LOOPIJ {
        result[i][j] = value;
    }

    setRawTile(trd, result);
}

void tbcast(const TOPRand &trd, const TOPRand &trs1, uint8_t row, uint8_t col) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    assert(row < TILE_SIZE && col < TILE_SIZE && "行列索引越界");
    assert(trd.dtype == trs1.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1);
    RawTile result;

    uint32_t value = tile1[row][col];
    LOOPIJ {
        result[i][j] = value;
    }

    setRawTile(trd, result);
}

void tbcastRow(const TOPRand &trd, const TOPRand &trs1, uint8_t col) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    assert(col < TILE_SIZE && "列索引越界");
    assert(trd.dtype == trs1.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1);
    RawTile result;

    LOOPI {
        uint32_t value = tile1[i][col];
        LOOPJ {
            result[i][j] = value;
        }
    }

    setRawTile(trd, result);
}

void tbcastCol(const TOPRand &trd, const TOPRand &trs1, uint8_t row) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    assert(row < TILE_SIZE && "行索引越界");
    assert(trd.dtype == trs1.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1);
    RawTile result;

    LOOPJ {
        uint32_t value = tile1[row][j];
        LOOPI {
            result[i][j] = value;
        }
    }

    setRawTile(trd, result);
}

void tshift(const TOPRand &trd, const TOPRand &trs1, int8_t row_offset, int8_t col_offset) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    assert(trd.dtype == trs1.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1);
    RawTile result;

    LOOPIJ {
        const int32_t src_row = static_cast<int32_t>(i) - static_cast<int32_t>(row_offset);
        const int32_t src_col = static_cast<int32_t>(j) - static_cast<int32_t>(col_offset);
        if (src_row >= 0 &&
            src_row < static_cast<int32_t>(TILE_SIZE) &&
            src_col >= 0 &&
            src_col < static_cast<int32_t>(TILE_SIZE)) {
            result[i][j] = tile1[static_cast<size_t>(src_row)][static_cast<size_t>(src_col)];
        } else {
            result[i][j] = 0;
        }
    }

    setRawTile(trd, result);
}

void ttrans(const TOPRand &trd, const TOPRand &trs1) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    assert(trd.dtype == trs1.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1);
    RawTile result;

    LOOPIJ {
        result[i][j] = tile1[j][i];
    }

    setRawTile(trd, result);
}

void ttransRevRow(const TOPRand &trd, const TOPRand &trs1) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    assert(trd.dtype == trs1.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1);
    RawTile result;

    LOOPIJ {
        result[i][j] = tile1[TILE_SIZE - 1 - i][j];
    }

    setRawTile(trd, result);
}

void ttransRevCol(const TOPRand &trd, const TOPRand &trs1) {
    CHECK_TREG_NORM(trd);
    CHECK_TREG_NORM(trs1);
    assert(trd.dtype == trs1.dtype && "输出数据类型必须与输入矩阵数据类型相同");

    RawTile tile1 = getRawTile(trs1);
    RawTile result;

    LOOPIJ {
        result[i][j] = tile1[i][TILE_SIZE - 1 - j];
    }

    setRawTile(trd, result);
}

void tsetmask(const TOPRand &trd_m8, uint8_t top, uint8_t bottom, uint8_t left, uint8_t right, bool invert) {
    CHECK_TREG_M8(trd_m8);
    assert(top + bottom < TILE_SIZE && left + right < TILE_SIZE && "边界参数过大");
    
    RawTile result;

    LOOPIJ {
        bool in_mask = (i >= top && i < TILE_SIZE - bottom && j >= left && j < TILE_SIZE - right);
        if (invert) {
            result[i][j] = in_mask ? 0 : 1;
        } else {
            result[i][j] = in_mask ? 1 : 0;
        }
    }
    setRawTile(trd_m8, result);
}

void tsetmaskLowTri(const TOPRand &trd_m8, uint8_t top, uint8_t bottom, uint8_t left, uint8_t right, bool diagonal, bool invert) {
    CHECK_TREG_M8(trd_m8);
    assert(top + bottom < TILE_SIZE && left + right < TILE_SIZE && "边界参数过大");
    
    RawTile result;

    LOOPIJ {
        bool in_mask = (i >= top && i < TILE_SIZE - bottom && j >= left && j < TILE_SIZE - right);
        if (in_mask) {
            const uint32_t row_idx = i - static_cast<uint32_t>(top);
            const uint32_t col_idx = j - static_cast<uint32_t>(left);
            bool in_tri = diagonal ? (row_idx >= col_idx) : (row_idx > col_idx);
            in_mask = in_mask && in_tri;
        }
        if (invert) {
            result[i][j] = in_mask ? 0 : 1;
        } else {
            result[i][j] = in_mask ? 1 : 0;
        }
    }
    setRawTile(trd_m8, result);
}

void tsetmaskUpTri(const TOPRand &trd_m8, uint8_t top, uint8_t bottom, uint8_t left, uint8_t right, bool diagonal, bool invert) {
    CHECK_TREG_M8(trd_m8);
    assert(top + bottom < TILE_SIZE && left + right < TILE_SIZE && "边界参数过大");
    
    RawTile result;

    LOOPIJ {
        bool in_mask = (i >= top && i < TILE_SIZE - bottom && j >= left && j < TILE_SIZE - right);
        if (in_mask) {
            const uint32_t row_idx = i - static_cast<uint32_t>(top);
            const uint32_t col_idx = j - static_cast<uint32_t>(left);
            bool in_tri = diagonal ? (row_idx <= col_idx) : (row_idx < col_idx);
            in_mask = in_mask && in_tri;
        }
        if (invert) {
            result[i][j] = in_mask ? 0 : 1;
        } else {
            result[i][j] = in_mask ? 1 : 0;
        }
    }
    setRawTile(trd_m8, result);
}







}
