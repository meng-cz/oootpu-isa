#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>
#include <utility>

namespace tpuarithm {

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t fadd_rne(uint32_t a, uint32_t b) {
    static_assert(ExpWidth > 0, "ExpWidth must be greater than 0");
    static_assert(1 + ExpWidth + MantWidth <= 32, "Format must fit in uint32_t");

    constexpr uint32_t kTotalBits = 1 + ExpWidth + MantWidth;
    constexpr uint32_t kSignShift = ExpWidth + MantWidth;
    constexpr uint32_t kRoundBits = 3;

    constexpr uint32_t kMask32 = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t kAllMask = (kTotalBits == 32) ? kMask32 : ((uint32_t{1} << kTotalBits) - 1u);
    constexpr uint32_t kExpMask = (ExpWidth == 32) ? kMask32 : ((uint32_t{1} << ExpWidth) - 1u);
    constexpr uint32_t kFracMask = (MantWidth == 32) ? kMask32 : ((MantWidth == 0) ? 0u : ((uint32_t{1} << MantWidth) - 1u));
    constexpr uint32_t kExpMax = kExpMask;
    constexpr uint64_t kHidden = uint64_t{1} << MantWidth;
    constexpr uint64_t kHiddenWithGRS = kHidden << kRoundBits;
    constexpr uint32_t kQuietBit = (MantWidth == 0) ? 0u : (uint32_t{1} << (MantWidth - 1));

    auto shift_right_jam = [](uint64_t v, uint32_t dist) -> uint64_t {
        if (dist == 0) return v;
        if (dist < 64) {
            const uint64_t mask = (uint64_t{1} << dist) - 1u;
            return (v >> dist) | ((v & mask) ? 1u : 0u);
        }
        return v ? 1u : 0u;
    };

    auto make_qnan = [&]() -> uint32_t {
        return ((kExpMax << MantWidth) | kQuietBit) & kAllMask;
    };

    a &= kAllMask;
    b &= kAllMask;

    uint32_t sign_a = (a >> kSignShift) & 1u;
    uint32_t sign_b = (b >> kSignShift) & 1u;
    uint32_t exp_a = (a >> MantWidth) & kExpMask;
    uint32_t exp_b = (b >> MantWidth) & kExpMask;
    uint32_t frac_a = a & kFracMask;
    uint32_t frac_b = b & kFracMask;

    const bool a_nan = (MantWidth != 0) && (exp_a == kExpMax) && (frac_a != 0);
    const bool b_nan = (MantWidth != 0) && (exp_b == kExpMax) && (frac_b != 0);
    if (a_nan) return (a | kQuietBit) & kAllMask;
    if (b_nan) return (b | kQuietBit) & kAllMask;

    const bool a_inf = (exp_a == kExpMax) && (frac_a == 0);
    const bool b_inf = (exp_b == kExpMax) && (frac_b == 0);
    if (a_inf || b_inf) {
        if (a_inf && b_inf && (sign_a != sign_b)) return make_qnan();
        return a_inf ? a : b;
    }

    uint32_t exp_x = exp_a ? exp_a : 1u;
    uint32_t exp_y = exp_b ? exp_b : 1u;
    uint64_t sig_x = (exp_a ? (kHidden | frac_a) : frac_a) << kRoundBits;
    uint64_t sig_y = (exp_b ? (kHidden | frac_b) : frac_b) << kRoundBits;
    uint32_t sign_x = sign_a;
    uint32_t sign_y = sign_b;

    if ((exp_x < exp_y) || ((exp_x == exp_y) && (sig_x < sig_y))) {
        std::swap(exp_x, exp_y);
        std::swap(sig_x, sig_y);
        std::swap(sign_x, sign_y);
    }

    sig_y = shift_right_jam(sig_y, exp_x - exp_y);

    uint32_t sign_z = sign_x;
    uint32_t exp_z = exp_x;
    uint64_t sig_z;

    if (sign_x == sign_y) {
        sig_z = sig_x + sig_y;
        if (sig_z >= (kHiddenWithGRS << 1)) {
            sig_z = shift_right_jam(sig_z, 1);
            ++exp_z;
        }
    } else {
        sig_z = sig_x - sig_y;
        if (sig_z == 0) return 0;

        if ((exp_z > 1) && (sig_z < kHiddenWithGRS)) {
            const uint32_t lead = 63u - static_cast<uint32_t>(__builtin_clzll(sig_z));
            const uint32_t target = MantWidth + kRoundBits;
            const uint32_t need = target > lead ? (target - lead) : 0u;
            const uint32_t sh = (need < (exp_z - 1)) ? need : (exp_z - 1);
            sig_z <<= sh;
            exp_z -= sh;
        }
    }

    if (exp_z >= kExpMax) {
        return ((sign_z << kSignShift) | (kExpMax << MantWidth)) & kAllMask;
    }

    if (exp_z < 1) {
        sig_z = shift_right_jam(sig_z, 1u - exp_z);
        exp_z = 1;
    }

    if (exp_z == 1 && sig_z < kHiddenWithGRS) {
        // already subnormal range
    }

    uint64_t sig_main = sig_z >> kRoundBits;
    const uint32_t round = static_cast<uint32_t>(sig_z & ((1u << kRoundBits) - 1u));
    const bool incr = (round > 0b100u) || ((round == 0b100u) && (sig_main & 1u));
    if (incr) ++sig_main;

    if (sig_main >= (kHidden << 1)) {
        sig_main >>= 1;
        ++exp_z;
        if (exp_z >= kExpMax) {
            return ((sign_z << kSignShift) | (kExpMax << MantWidth)) & kAllMask;
        }
    }

    uint32_t exp_field;
    uint32_t frac_field;
    if (sig_main < kHidden) {
        exp_field = 0;
        frac_field = static_cast<uint32_t>(sig_main) & kFracMask;
    } else {
        exp_field = exp_z;
        frac_field = static_cast<uint32_t>(sig_main & (kHidden - 1u));
    }

    return ((sign_z << kSignShift) | (exp_field << MantWidth) | frac_field) & kAllMask;
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t fsub_rne(uint32_t a, uint32_t b) {
    return fadd_rne<ExpWidth, MantWidth>(a, b ^ (1u << (ExpWidth + MantWidth)));
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline bool fgt(uint32_t a, uint32_t b) {
    static_assert(ExpWidth > 0, "ExpWidth must be greater than 0");
    static_assert(1 + ExpWidth + MantWidth <= 32, "Format must fit in uint32_t");
    constexpr uint32_t kTotalBits = 1 + ExpWidth + MantWidth;
    constexpr uint32_t kSignShift = ExpWidth + MantWidth;
    constexpr uint32_t kMask32 = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t kAllMask = (kTotalBits == 32) ? kMask32 : ((uint32_t{1} << kTotalBits) - 1u);
    constexpr uint32_t kSignMask = uint32_t{1} << kSignShift;
    constexpr uint32_t kMagMask = kAllMask ^ kSignMask;
    constexpr uint32_t kExpMask = (uint32_t{1} << ExpWidth) - 1u;
    constexpr uint32_t kFracMask = (MantWidth == 0) ? 0u : ((uint32_t{1} << MantWidth) - 1u);
    constexpr uint32_t kExpMax = kExpMask;

    a &= kAllMask;
    b &= kAllMask;

    const uint32_t exp_a = (a >> MantWidth) & kExpMask;
    const uint32_t exp_b = (b >> MantWidth) & kExpMask;
    const uint32_t frac_a = a & kFracMask;
    const uint32_t frac_b = b & kFracMask;

    const bool a_nan = (MantWidth != 0) && (exp_a == kExpMax) && (frac_a != 0);
    const bool b_nan = (MantWidth != 0) && (exp_b == kExpMax) && (frac_b != 0);
    if (a_nan || b_nan) return false;

    if (((a & kMagMask) == 0) && ((b & kMagMask) == 0)) return false;
    if (a == b) return false;

    const uint32_t sign_a = (a >> kSignShift) & 1u;
    const uint32_t sign_b = (b >> kSignShift) & 1u;
    if (sign_a != sign_b) return sign_a == 0;

    const uint32_t mag_a = a & kMagMask;
    const uint32_t mag_b = b & kMagMask;
    return sign_a ? (mag_a < mag_b) : (mag_a > mag_b);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline bool feq(uint32_t a, uint32_t b) {
    static_assert(ExpWidth > 0, "ExpWidth must be greater than 0");
    static_assert(1 + ExpWidth + MantWidth <= 32, "Format must fit in uint32_t");
    constexpr uint32_t kTotalBits = 1 + ExpWidth + MantWidth;
    constexpr uint32_t kSignShift = ExpWidth + MantWidth;
    constexpr uint32_t kMask32 = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t kAllMask = (kTotalBits == 32) ? kMask32 : ((uint32_t{1} << kTotalBits) - 1u);
    constexpr uint32_t kSignMask = uint32_t{1} << kSignShift;
    constexpr uint32_t kMagMask = kAllMask ^ kSignMask;
    constexpr uint32_t kExpMask = (uint32_t{1} << ExpWidth) - 1u;
    constexpr uint32_t kFracMask = (MantWidth == 0) ? 0u : ((uint32_t{1} << MantWidth) - 1u);
    constexpr uint32_t kExpMax = kExpMask;

    a &= kAllMask;
    b &= kAllMask;

    const uint32_t exp_a = (a >> MantWidth) & kExpMask;
    const uint32_t exp_b = (b >> MantWidth) & kExpMask;
    const uint32_t frac_a = a & kFracMask;
    const uint32_t frac_b = b & kFracMask;
    const bool a_nan = (MantWidth != 0) && (exp_a == kExpMax) && (frac_a != 0);
    const bool b_nan = (MantWidth != 0) && (exp_b == kExpMax) && (frac_b != 0);
    if (a_nan || b_nan) return false;

    if (((a & kMagMask) == 0) && ((b & kMagMask) == 0)) return true;
    return a == b;
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline bool fne(uint32_t a, uint32_t b) {
    return !feq<ExpWidth, MantWidth>(a, b);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline bool flt(uint32_t a, uint32_t b) {
    return fgt<ExpWidth, MantWidth>(b, a);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline bool fge(uint32_t a, uint32_t b) {
    return fgt<ExpWidth, MantWidth>(a, b) || feq<ExpWidth, MantWidth>(a, b);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline bool fle(uint32_t a, uint32_t b) {
    return flt<ExpWidth, MantWidth>(a, b) || feq<ExpWidth, MantWidth>(a, b);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t fneg(uint32_t a) {
    static_assert(ExpWidth > 0, "ExpWidth must be greater than 0");
    static_assert(1 + ExpWidth + MantWidth <= 32, "Format must fit in uint32_t");
    constexpr uint32_t kTotalBits = 1 + ExpWidth + MantWidth;
    constexpr uint32_t kSignShift = ExpWidth + MantWidth;
    constexpr uint32_t kMask32 = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t kAllMask = (kTotalBits == 32) ? kMask32 : ((uint32_t{1} << kTotalBits) - 1u);
    return (a ^ (uint32_t{1} << kSignShift)) & kAllMask;
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t fabs(uint32_t a) {
    static_assert(ExpWidth > 0, "ExpWidth must be greater than 0");
    static_assert(1 + ExpWidth + MantWidth <= 32, "Format must fit in uint32_t");
    constexpr uint32_t kTotalBits = 1 + ExpWidth + MantWidth;
    constexpr uint32_t kSignShift = ExpWidth + MantWidth;
    constexpr uint32_t kMask32 = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t kAllMask = (kTotalBits == 32) ? kMask32 : ((uint32_t{1} << kTotalBits) - 1u);
    return a & (kAllMask ^ (uint32_t{1} << kSignShift));
}


} // namespace tpuarithm
