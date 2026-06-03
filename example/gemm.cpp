
#include "oootpu.hpp"
#include "funcsim/funcsim.hpp"

#include "simmem.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

namespace oootpu {

constexpr inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

constexpr inline uint32_t align(uint32_t a, uint32_t b) {
    return ceil_div(a, b) * b;
}

void tgemm_i8i32_align32(
    TMemAddr A_addr, TMemAddr B_addr, TMemAddr C_addr,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t lda_ele, uint32_t ldb_ele, uint32_t ldc_ele
) {
    const uint32_t treg_num = getTRegNum();

    const uint32_t m_tile = align(M, TILE_SIZE) / TILE_SIZE;
    const uint32_t n_tile = align(N, TILE_SIZE) / TILE_SIZE;
    const uint32_t k_tile = align(K, TILE_SIZE) / TILE_SIZE;

    assert(lda_ele % TILE_SIZE == 0 && ldb_ele % TILE_SIZE == 0 && ldc_ele % TILE_SIZE == 0 && "要求 lda, ldb, ldc 必须是 Tile 大小的整数倍");

    const TOPRand a_reg1_base = S8(TRegIdx(0));
    const TOPRand b_reg1 = S8(static_cast<TRegIdx>(k_tile + 4));
    const TOPRand c_reg4 = S32(static_cast<TRegIdx>(k_tile));

    assert(treg_num >= k_tile + 6 && "寄存器数量不足以加载 A 的一个行");

    const TMemAddr a_mt_stride = static_cast<TMemAddr>(TILE_SIZE) * static_cast<TMemAddr>(lda_ele);
    const TMemAddr a_kt_stride = static_cast<TMemAddr>(TILE_SIZE);
    const TMemAddr b_kt_stride = static_cast<TMemAddr>(TILE_SIZE) * static_cast<TMemAddr>(ldb_ele);
    const TMemAddr b_nt_stride = static_cast<TMemAddr>(TILE_SIZE);
    const TMemAddr c_mt_stride = static_cast<TMemAddr>(TILE_SIZE) * static_cast<TMemAddr>(ldc_ele) * static_cast<TMemAddr>(sizeof(int32_t));
    const TMemAddr c_nt_stride = static_cast<TMemAddr>(TILE_SIZE) * static_cast<TMemAddr>(sizeof(int32_t));

    TMemAddr a_mt_addr = A_addr;
    TMemAddr c_mt_addr = C_addr;
    for (uint32_t mt = 0; mt < m_tile; ++mt, a_mt_addr += a_mt_stride, c_mt_addr += c_mt_stride) {
        // 把 A 的一行 Tile 大小的数据加载到寄存器，后续循环复用
        TMemAddr a_tile_addr = a_mt_addr;
        for (uint32_t kt = 0; kt < k_tile; ++kt, a_tile_addr += a_kt_stride) {
            tloadRow(a_reg1_base + kt, a_tile_addr, lda_ele);
        }
        TMemAddr c_tile_addr = c_mt_addr;
        TMemAddr b_nt_addr = B_addr;
        for (uint32_t nt = 0; nt < n_tile; ++nt, c_tile_addr += c_nt_stride, b_nt_addr += b_nt_stride) {
            TMemAddr b_tile_addr = b_nt_addr;
            for (uint32_t kt = 0; kt < k_tile; ++kt, b_tile_addr += b_kt_stride) {
                tloadRow(b_reg1, b_tile_addr, ldb_ele);
                tgemm(0, a_reg1_base + kt, b_reg1, false, false, kt != 0);
            }
            taccget(0, c_reg4, S32(TZERO), TRM::RNE, false);
            tstoreRow(c_reg4, c_tile_addr, ldc_ele);
        }
    }
}

} // namespace oootpu

constexpr uint32_t GEMM_M = 512;
constexpr uint32_t GEMM_K = 512;
constexpr uint32_t GEMM_N = 512;

int main() {
    constexpr uint32_t M = GEMM_M;
    constexpr uint32_t K = GEMM_K;
    constexpr uint32_t N = GEMM_N;
    constexpr uint32_t lda = K;
    constexpr uint32_t ldb = N;
    constexpr uint32_t ldc = N;

    printf("Testing tgemm_i8i32_align32 with M=%u, K=%u, N=%u\n", M, K, N);
    printf("Setup test data...\n");

    std::vector<int8_t> A(M * lda);
    std::vector<int8_t> B(K * ldb);
    std::vector<int32_t> C_ref(M * ldc, 0);
    std::vector<int32_t> C_out(M * ldc, 0);

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < K; ++j) {
            A[i * lda + j] = static_cast<int8_t>((i * 7 + j * 3 + 11) % 31 - 15);
        }
    }
    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            B[i * ldb + j] = static_cast<int8_t>((i * 5 + j * 9 + 13) % 29 - 14);
        }
    }

    printf("Init test memory...\n");

    const oootpu::TMemAddr A_addr = 0;
    const oootpu::TMemAddr B_addr = static_cast<oootpu::TMemAddr>(A.size());
    const oootpu::TMemAddr C_addr = B_addr + static_cast<oootpu::TMemAddr>(B.size());
    const size_t mem_size = static_cast<size_t>(C_addr) + C_out.size() * sizeof(int32_t);
    SimMemory sim_mem(mem_size);
    oootpu::setMemProxy(&sim_mem);

    memcpy(sim_mem.memory.data() + A_addr, A.data(), A.size());
    memcpy(sim_mem.memory.data() + B_addr, B.data(), B.size());
    memset(sim_mem.memory.data() + C_addr, 0, C_out.size() * sizeof(int32_t));

    printf("Start tgemm_i8i32_align32...\n");

    oootpu::tgemm_i8i32_align32(A_addr, B_addr, C_addr, M, N, K, lda, ldb, ldc);

    memcpy(C_out.data(), sim_mem.memory.data() + C_addr, C_out.size() * sizeof(int32_t));

    printf("Checking results...\n");

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[m * lda + k]) *
                       static_cast<int32_t>(B[k * ldb + n]);
            }
            C_ref[m * ldc + n] = acc;
        }
    }

    for (size_t idx = 0; idx < C_out.size(); ++idx) {
        if (C_out[idx] != C_ref[idx]) {
            std::cerr << "GEMM mismatch at idx=" << idx
                      << ", got=" << C_out[idx]
                      << ", expect=" << C_ref[idx] << '\n';
            return 1;
        }
    }

    std::cout << "tgemm_i8i32_align32 test PASS\n";
    return 0;
}
