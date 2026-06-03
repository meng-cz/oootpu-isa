#pragma once

#include <cstdint>
#include <array>
#include <cstddef>
#include <vector>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <limits>

using std::vector;

#if defined(__GNUC__) || defined(__clang__)
__extension__ typedef unsigned __int128 tpu_uint128_t;
#endif

struct ExactFloat {
    vector<uint64_t> mant; // 标准大整数视图，小数点在最右边，每个元素存储64位有效数字，mant[0]为最低有效位
    int64_t exp;
    bool sign;
    bool is_inf;
    bool is_nan;

    inline static ExactFloat ZERO(bool sign = false) {
        ExactFloat r{.mant = {}, .exp = 0, .sign = sign, .is_inf = false, .is_nan = false};
        return r;
    }
    inline static ExactFloat NZERO() {
        return ZERO(true);
    }
    inline static ExactFloat INF(bool sign = false) {
        ExactFloat r{.mant = {}, .exp = 0, .sign = sign, .is_inf = true, .is_nan = false};
        return r;
    }
    inline static ExactFloat NINF() {
        return INF(true);
    }
    inline static ExactFloat NAN() {
        ExactFloat r{.mant = {}, .exp = 0, .sign = false, .is_inf = false, .is_nan = true};
        return r;
    }
};

template <uint32_t ExpWidth, uint32_t MantWidth, typename T = uint64_t>
inline ExactFloat floatToEF(T value) {
    static_assert(ExpWidth + MantWidth < 64, "Total width of exponent and mantissa must be less than 64 bits");
    static_assert(ExpWidth > 0, "Exponent width must be greater than 0");
    static_assert(sizeof(T) <= sizeof(uint64_t), "Input width must be <= 64 bits");
    static_assert((ExpWidth + MantWidth + 1) <= (sizeof(T) * 8), "Type width is too small for the specified layout");

    using U = uint64_t;
    using RawU =
        std::conditional_t<sizeof(T) <= 2, uint16_t,
        std::conditional_t<sizeof(T) <= 4, uint32_t, uint64_t>>;
    U bits = 0;
    if constexpr (std::is_floating_point_v<T>) {
        RawU raw = 0;
        std::memcpy(&raw, &value, sizeof(T));
        bits = static_cast<U>(raw);
    } else {
        bits = static_cast<U>(value);
    }

    const U mant_mask = (MantWidth == 64) ? ~U(0) : ((U(1) << MantWidth) - 1);
    const U exp_mask = (U(1) << ExpWidth) - 1;
    const uint32_t exp_shift = MantWidth;
    const uint32_t sign_shift = ExpWidth + MantWidth;

    const bool sign = ((bits >> sign_shift) & U(1)) != 0;
    const U exp_bits = (bits >> exp_shift) & exp_mask;
    const U frac_bits = bits & mant_mask;

    ExactFloat r{
        .mant = {},
        .exp = 0,
        .sign = sign,
        .is_inf = false,
        .is_nan = false
    };

    if (exp_bits == exp_mask) {
        if (frac_bits == 0) {
            r.is_inf = true;
        } else {
            r.is_nan = true;
        }
        return r;
    }

    const int64_t bias = (int64_t(1) << (ExpWidth - 1)) - 1;
    const int64_t mant_w = static_cast<int64_t>(MantWidth);

    if (exp_bits == 0) {
        if (frac_bits == 0) {
            return r;
        }
        r.mant.push_back(frac_bits);
        r.exp = (1 - bias) - mant_w;
        return r;
    }

    const U int_mant = (U(1) << MantWidth) | frac_bits;
    r.mant.push_back(int_mant);
    r.exp = static_cast<int64_t>(exp_bits) - bias - mant_w;
    return r;
}

enum class RoundingMode {
    RNE, // Round to Nearest, ties to Even
    RTZ, // Round towards Zero
    RDN, // Round Down (towards -inf)
    RUP  // Round Up (towards +inf)
};

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint64_t floatFromEF(const ExactFloat& ef, RoundingMode rm, bool saturate) {
    static_assert(ExpWidth + MantWidth < 64, "Total width of exponent and mantissa must be less than 64 bits");
    const uint64_t sign_bit = uint64_t(ef.sign) << (ExpWidth + MantWidth);
    const uint64_t exp_all_ones = (uint64_t(1) << ExpWidth) - 1;
    const uint64_t mant_mask = (MantWidth == 0) ? 0 : ((uint64_t(1) << MantWidth) - 1);
    const int64_t bias = (int64_t(1) << (ExpWidth - 1)) - 1;
    const int64_t emin = 1 - bias;
    const int64_t emax = bias;

    const auto pack = [&](uint64_t exp_bits, uint64_t frac_bits) -> uint64_t {
        return sign_bit | (exp_bits << MantWidth) | (frac_bits & mant_mask);
    };
    const auto pack_inf = [&]() -> uint64_t {
        return pack(exp_all_ones, 0);
    };
    const auto pack_max_finite = [&]() -> uint64_t {
        return pack(exp_all_ones - 1, mant_mask);
    };

    const auto overflow_result = [&]() -> uint64_t {
        if (saturate) return pack_max_finite();
        const bool to_inf =
            (rm == RoundingMode::RNE) ||
            (rm == RoundingMode::RUP && !ef.sign) ||
            (rm == RoundingMode::RDN && ef.sign);
        return to_inf ? pack_inf() : pack_max_finite();
    };

    if (ef.is_nan) {
        if constexpr (MantWidth == 0) return pack(exp_all_ones, 0);
        return pack(exp_all_ones, uint64_t(1) << (MantWidth - 1));
    }
    if (ef.is_inf) return saturate ? pack_max_finite() : pack_inf();
    if (ef.mant.empty()) return sign_bit;

    const auto& m = ef.mant;
    size_t n = m.size();
    while (n > 0 && m[n - 1] == 0) --n;
    if (n == 0) return sign_bit;

    const auto get_bit = [&](int64_t idx) -> bool {
        if (idx < 0) return false;
        const size_t wi = static_cast<size_t>(idx >> 6);
        if (wi >= n) return false;
        return ((m[wi] >> (idx & 63)) & 1u) != 0;
    };
    const auto any_bits_below = [&](int64_t lim) -> bool {
        if (lim <= 0) return false;
        const size_t full_words = static_cast<size_t>(lim >> 6);
        const uint32_t rem_bits = static_cast<uint32_t>(lim & 63);
        const size_t words_to_check = (full_words < n) ? full_words : n;
        for (size_t i = 0; i < words_to_check; ++i) {
            if (m[i] != 0) return true;
        }
        if (rem_bits != 0 && full_words < n) {
            const uint64_t mask = (uint64_t(1) << rem_bits) - 1;
            if ((m[full_words] & mask) != 0) return true;
        }
        return false;
    };

    const uint32_t msb_in_top = 63u - static_cast<uint32_t>(__builtin_clzll(m[n - 1]));
    const int64_t k = static_cast<int64_t>((n - 1) * 64 + msb_in_top);
    int64_t E = ef.exp + k;

    auto round_right_shift = [&](int64_t sh) -> uint64_t {
        if (sh == 0) return m[0];
        if (sh < 0) return 0;
        uint64_t t = 0;
        const size_t ws = static_cast<size_t>(sh >> 6);
        const uint32_t bs = static_cast<uint32_t>(sh & 63);
        if (ws < n) {
            t = m[ws] >> bs;
            if (bs != 0 && ws + 1 < n) t |= (m[ws + 1] << (64 - bs));
        }
        const bool rem_nonzero = any_bits_below(sh);
        bool round_up = false;
        switch (rm) {
            case RoundingMode::RNE: {
                const bool half_bit = get_bit(sh - 1);
                const bool low_bits = any_bits_below(sh - 1);
                const bool gt_half = half_bit && low_bits;
                const bool eq_half = half_bit && !low_bits;
                round_up = gt_half || (eq_half && ((t & 1u) != 0));
                break;
            }
            case RoundingMode::RTZ:
                break;
            case RoundingMode::RUP:
                round_up = !ef.sign && rem_nonzero;
                break;
            case RoundingMode::RDN:
                round_up = ef.sign && rem_nonzero;
                break;
        }
        return t + static_cast<uint64_t>(round_up);
    };

    if (E > emax) return overflow_result();

    if (E >= emin) {
        uint64_t q = 0;
        if (k >= static_cast<int64_t>(MantWidth)) {
            q = round_right_shift(k - static_cast<int64_t>(MantWidth));
        } else {
            q = m[0] << (MantWidth - static_cast<uint32_t>(k));
        }

        const uint64_t carry_threshold = uint64_t(1) << (MantWidth + 1);
        if (q >= carry_threshold) {
            q >>= 1;
            ++E;
            if (E > emax) return overflow_result();
        }

        const uint64_t exp_bits = static_cast<uint64_t>(E + bias);
        return pack(exp_bits, q & mant_mask);
    }

    const int64_t sub_scale = emin - static_cast<int64_t>(MantWidth);
    const int64_t sh = sub_scale - ef.exp;
    uint64_t frac = 0;
    if (sh > 0) {
        frac = round_right_shift(sh);
    } else {
        const uint32_t ls = static_cast<uint32_t>(-sh);
        frac = (ls >= 64) ? (mant_mask + 1) : (m[0] << ls);
    }

    if (frac == 0) return sign_bit;
    if (frac > mant_mask) {
        return pack(1, 0); // Rounded into the minimum normal value.
    }
    return pack(0, frac);
}

template <uint32_t ExpW, uint32_t MantW>
struct FPFormat {
    static constexpr uint32_t kExpWidth = ExpW;
    static constexpr uint32_t kMantWidth = MantW;
};

using FP8E4M3 = FPFormat<4, 3>;
using FP8E5M2 = FPFormat<5, 2>;
using FP16 = FPFormat<5, 10>;
using BF16 = FPFormat<8, 7>;
using FP32 = FPFormat<8, 23>;
using FP64 = FPFormat<11, 52>;

template <typename FP>
inline ExactFloat floatToEF(uint64_t value) {
    return floatToEF<FP::kExpWidth, FP::kMantWidth, uint64_t>(value);
}

template <typename FP>
inline uint64_t floatFromEF(const ExactFloat& ef, RoundingMode rm, bool saturate) {
    return floatFromEF<FP::kExpWidth, FP::kMantWidth>(ef, rm, saturate);
}

template <typename IntType>
inline ExactFloat intToEF(IntType value) {
    using T = std::remove_cv_t<std::remove_reference_t<IntType>>;
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                  "intToEF requires a non-bool integral type");

    using U = std::make_unsigned_t<T>;
    bool sign = false;
    U mag = 0;

    if constexpr (std::is_signed_v<T>) {
        sign = (value < 0);
        const U bits = static_cast<U>(value);
        mag = sign ? (U(0) - bits) : bits; // Safe absolute value in two's complement.
    } else {
        mag = static_cast<U>(value);
    }

    if (mag == 0) {
        return ExactFloat::ZERO();
    }

    ExactFloat r{
        .mant = {},
        .exp = 0,
        .sign = sign,
        .is_inf = false,
        .is_nan = false
    };

    if constexpr (sizeof(U) <= sizeof(uint64_t)) {
        r.mant.push_back(static_cast<uint64_t>(mag));
    } else {
        U v = mag;
        while (v != 0) {
            r.mant.push_back(static_cast<uint64_t>(v));
            v >>= 64;
        }
    }
    return r;
}

template <typename IntType>
inline IntType intFromEF(const ExactFloat& ef, RoundingMode rm, bool saturate) {
    using T = std::remove_cv_t<std::remove_reference_t<IntType>>;
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>,
                  "intFromEF requires a non-bool integral type");
    using U = std::make_unsigned_t<T>;

    constexpr bool kSigned = std::is_signed_v<T>;
    constexpr int kBits = std::numeric_limits<U>::digits;
    constexpr int kWords = (kBits + 63) / 64;
    constexpr U kUMax = std::numeric_limits<U>::max();
    constexpr U kSignedPosMaxMag = kSigned ? (kUMax >> 1) : kUMax;
    constexpr U kSignedNegMaxMag = kSigned ? (U(1) << (kBits - 1)) : U(0);

    const auto clamp_overflow = [&]() -> T {
        if constexpr (kSigned) {
            return ef.sign ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
        } else {
            return ef.sign ? T(0) : std::numeric_limits<T>::max();
        }
    };
    const auto from_mag = [&](U mag) -> T {
        if constexpr (kSigned) {
            if (!ef.sign) return static_cast<T>(mag);
            if (mag == 0) return T(0);
            if (mag == kSignedNegMaxMag) return std::numeric_limits<T>::min();
            return static_cast<T>(-static_cast<T>(mag));
        } else {
            return ef.sign ? T(0) : static_cast<T>(mag);
        }
    };

    if (ef.is_nan) return T(0);
    if (ef.mant.empty()) return T(0);
    if (ef.is_inf) return clamp_overflow();

    const auto& m = ef.mant;
    size_t n = m.size();
    while (n > 0 && m[n - 1] == 0) --n;
    if (n == 0) return T(0);

    const auto get_bit = [&](int64_t idx) -> bool {
        if (idx < 0) return false;
        const size_t wi = static_cast<size_t>(idx >> 6);
        if (wi >= n) return false;
        return ((m[wi] >> (idx & 63)) & 1u) != 0;
    };
    const auto any_bits_below = [&](int64_t lim) -> bool {
        if (lim <= 0) return false;
        const size_t full_words = static_cast<size_t>(lim >> 6);
        const uint32_t rem_bits = static_cast<uint32_t>(lim & 63);
        const size_t words_to_check = (full_words < n) ? full_words : n;
        for (size_t i = 0; i < words_to_check; ++i) {
            if (m[i] != 0) return true;
        }
        if (rem_bits != 0 && full_words < n) {
            const uint64_t mask = (uint64_t(1) << rem_bits) - 1;
            if ((m[full_words] & mask) != 0) return true;
        }
        return false;
    };

    const uint32_t msb_in_top = 63u - static_cast<uint32_t>(__builtin_clzll(m[n - 1]));
    const int64_t bitlen_m = static_cast<int64_t>((n - 1) * 64 + msb_in_top + 1);

    U mag = 0;
    bool overflow = false;
    const int64_t max_bits = kSigned ? (ef.sign ? kBits : (kBits - 1))
                                     : (ef.sign ? 0 : kBits);

    if (ef.exp >= 0) {
        const int64_t total_bits = bitlen_m + ef.exp;
        if (total_bits > max_bits) {
            overflow = true;
        } else {
            const int64_t ws = ef.exp >> 6;
            const uint32_t bs = static_cast<uint32_t>(ef.exp & 63);
            for (int ow = 0; ow < kWords; ++ow) {
                const int64_t src = static_cast<int64_t>(ow) - ws;
                uint64_t chunk = 0;
                if (src >= 0 && static_cast<size_t>(src) < n) {
                    chunk |= (m[static_cast<size_t>(src)] << bs);
                }
                if (bs != 0 && (src - 1) >= 0 && static_cast<size_t>(src - 1) < n) {
                    chunk |= (m[static_cast<size_t>(src - 1)] >> (64 - bs));
                }
                if (ow == 0) {
                    mag = static_cast<U>(chunk);
                } else if constexpr (kWords > 1) {
                    const int shift = ow * 64;
                    if (shift < kBits) {
                        mag = static_cast<U>(mag | static_cast<U>(static_cast<U>(chunk) << shift));
                    }
                }
            }
        }
    } else {
        const int64_t sh = -ef.exp;
        const int64_t int_bits = bitlen_m - sh;
        if (int_bits > max_bits) overflow = true;
        const int64_t ws = sh >> 6;
        const uint32_t bs = static_cast<uint32_t>(sh & 63);
        if (!overflow) {
            for (int ow = 0; ow < kWords; ++ow) {
                const int64_t src = ws + ow;
                uint64_t chunk = 0;
                if (src >= 0 && static_cast<size_t>(src) < n) {
                    chunk = (m[static_cast<size_t>(src)] >> bs);
                    if (bs != 0 && static_cast<size_t>(src + 1) < n) {
                        chunk |= (m[static_cast<size_t>(src + 1)] << (64 - bs));
                    }
                }
                if (ow == 0) {
                    mag = static_cast<U>(chunk);
                } else if constexpr (kWords > 1) {
                    const int shift = ow * 64;
                    if (shift < kBits) {
                        mag = static_cast<U>(mag | static_cast<U>(static_cast<U>(chunk) << shift));
                    }
                }
            }
        }

        const bool rem_nonzero = any_bits_below(sh);
        bool round_up = false;
        switch (rm) {
            case RoundingMode::RNE: {
                const bool half_bit = get_bit(sh - 1);
                const bool low_bits = any_bits_below(sh - 1);
                const bool gt_half = half_bit && low_bits;
                const bool eq_half = half_bit && !low_bits;
                round_up = gt_half || (eq_half && ((mag & U(1)) != 0));
                break;
            }
            case RoundingMode::RTZ:
                break;
            case RoundingMode::RUP:
                round_up = !ef.sign && rem_nonzero;
                break;
            case RoundingMode::RDN:
                round_up = ef.sign && rem_nonzero;
                break;
        }
        if (round_up) {
            if (mag == kUMax) {
                overflow = true;
            } else {
                ++mag;
            }
        }
    }

    if (!overflow) {
        if constexpr (!kSigned) {
            if (ef.sign && mag != 0) overflow = true;
        } else {
            if (ef.sign) {
                if (mag > kSignedNegMaxMag) overflow = true;
            } else {
                if (mag > kSignedPosMaxMag) overflow = true;
            }
        }
    }

    if (overflow) {
        if (saturate) return clamp_overflow();
        return clamp_overflow();
    }

    return from_mag(mag);
}

inline void trimMant(vector<uint64_t>& v) {
    while (!v.empty() && v.back() == 0) v.pop_back();
}

inline vector<uint64_t> shlMant(const vector<uint64_t>& src, int64_t shift_bits) {
    if (src.empty()) return {};
    if (shift_bits <= 0) return src;

    const size_t ws = static_cast<size_t>(shift_bits >> 6);
    const uint32_t bs = static_cast<uint32_t>(shift_bits & 63);
    vector<uint64_t> out(src.size() + ws + (bs ? 1 : 0), 0);

    if (bs == 0) {
        for (size_t i = 0; i < src.size(); ++i) out[i + ws] = src[i];
    } else {
        for (size_t i = 0; i < src.size(); ++i) {
            const uint64_t x = src[i];
            out[i + ws] |= (x << bs);
            out[i + ws + 1] |= (x >> (64 - bs));
        }
    }
    trimMant(out);
    return out;
}

inline int cmpMant(const vector<uint64_t>& a, const vector<uint64_t>& b) {
    size_t na = a.size();
    size_t nb = b.size();
    while (na > 0 && a[na - 1] == 0) --na;
    while (nb > 0 && b[nb - 1] == 0) --nb;
    if (na != nb) return (na < nb) ? -1 : 1;
    while (na > 0) {
        const uint64_t av = a[na - 1];
        const uint64_t bv = b[na - 1];
        if (av != bv) return (av < bv) ? -1 : 1;
        --na;
    }
    return 0;
}

inline vector<uint64_t> addMant(const vector<uint64_t>& a, const vector<uint64_t>& b) {
    const size_t n = (a.size() > b.size()) ? a.size() : b.size();
    vector<uint64_t> out(n + 1, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t av = (i < a.size()) ? a[i] : 0;
        const uint64_t bv = (i < b.size()) ? b[i] : 0;
        const tpu_uint128_t sum = static_cast<tpu_uint128_t>(av) + bv + carry;
        out[i] = static_cast<uint64_t>(sum);
        carry = static_cast<uint64_t>(sum >> 64);
    }
    out[n] = carry;
    trimMant(out);
    return out;
}

inline vector<uint64_t> subMant(const vector<uint64_t>& a, const vector<uint64_t>& b) {
    vector<uint64_t> out(a.size(), 0);
    uint64_t borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const uint64_t av = a[i];
        const uint64_t bv = (i < b.size()) ? b[i] : 0;
        const uint64_t t = av - borrow;
        const uint64_t b1 = (av < borrow) ? 1 : 0;
        const uint64_t r = t - bv;
        const uint64_t b2 = (t < bv) ? 1 : 0;
        out[i] = r;
        borrow = b1 | b2;
    }
    trimMant(out);
    return out;
}

inline ExactFloat makeFiniteEF(vector<uint64_t>&& mant, int64_t exp, bool sign) {
    trimMant(mant);
    if (mant.empty()) {
        return ExactFloat{.mant = {}, .exp = 0, .sign = false, .is_inf = false, .is_nan = false};
    }
    return ExactFloat{.mant = std::move(mant), .exp = exp, .sign = sign, .is_inf = false, .is_nan = false};
}

inline ExactFloat operator+(const ExactFloat& lhs, const ExactFloat& rhs) {
    if (lhs.is_nan || rhs.is_nan) {
        return ExactFloat::NAN();
    }
    if (lhs.is_inf || rhs.is_inf) {
        if (lhs.is_inf && rhs.is_inf && (lhs.sign != rhs.sign)) {
            return ExactFloat::NAN();
        }
        const bool sign = lhs.is_inf ? lhs.sign : rhs.sign;
        return ExactFloat::INF(sign);
    }

    const bool lzero = lhs.mant.empty();
    const bool rzero = rhs.mant.empty();
    if (lzero && rzero) {
        const bool sign = lhs.sign & rhs.sign;
        return ExactFloat::ZERO(sign);
    }
    if (lzero) return rhs;
    if (rzero) return lhs;

    const int64_t base_exp = (lhs.exp < rhs.exp) ? lhs.exp : rhs.exp;
    const int64_t lshift = lhs.exp - base_exp;
    const int64_t rshift = rhs.exp - base_exp;
    vector<uint64_t> lm = shlMant(lhs.mant, lshift);
    vector<uint64_t> rm = shlMant(rhs.mant, rshift);

    if (lhs.sign == rhs.sign) {
        return makeFiniteEF(addMant(lm, rm), base_exp, lhs.sign);
    }

    const int cmp = cmpMant(lm, rm);
    if (cmp == 0) {
        return ExactFloat::ZERO();
    }
    if (cmp > 0) {
        return makeFiniteEF(subMant(lm, rm), base_exp, lhs.sign);
    }
    return makeFiniteEF(subMant(rm, lm), base_exp, rhs.sign);
}

inline ExactFloat operator-(const ExactFloat& lhs, const ExactFloat& rhs) {
    if (rhs.is_nan) {
        return ExactFloat::NAN();
    }
    ExactFloat neg_rhs = rhs;
    neg_rhs.sign = !neg_rhs.sign;
    return lhs + neg_rhs;
}

inline ExactFloat operator*(const ExactFloat& lhs, const ExactFloat& rhs) {
    if (lhs.is_nan || rhs.is_nan) {
        return ExactFloat::NAN();
    }

    const bool sign = lhs.sign ^ rhs.sign;

    if (lhs.is_inf || rhs.is_inf) {
        const bool lzero = lhs.mant.empty();
        const bool rzero = rhs.mant.empty();
        if (lzero || rzero) {
            return ExactFloat::NAN();
        }
        return ExactFloat::INF(sign);
    }

    const bool lzero = lhs.mant.empty();
    const bool rzero = rhs.mant.empty();
    if (lzero || rzero) {
        return ExactFloat::ZERO(sign);
    }

    const vector<uint64_t>& a = lhs.mant;
    const vector<uint64_t>& b = rhs.mant;
    vector<uint64_t> out(a.size() + b.size(), 0);

    for (size_t i = 0; i < a.size(); ++i) {
        const uint64_t av = a[i];
        if (av == 0) continue;
        tpu_uint128_t carry = 0;
        size_t k = i;
        for (size_t j = 0; j < b.size(); ++j, ++k) {
            const tpu_uint128_t acc =
                static_cast<tpu_uint128_t>(av) * b[j] + out[k] + carry;
            out[k] = static_cast<uint64_t>(acc);
            carry = acc >> 64;
        }
        while (carry != 0) {
            const tpu_uint128_t acc = static_cast<tpu_uint128_t>(out[k]) + carry;
            out[k] = static_cast<uint64_t>(acc);
            carry = acc >> 64;
            ++k;
        }
    }

    return makeFiniteEF(std::move(out), lhs.exp + rhs.exp, sign);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t divUIntAndRound(const ExactFloat &f, uint32_t divisor, RoundingMode rm) {
    static_assert(ExpWidth + MantWidth < 32, "Output width must fit in uint32_t");
    const uint64_t sign_bit = uint64_t(f.sign) << (ExpWidth + MantWidth);
    const uint64_t exp_all_ones = (uint64_t(1) << ExpWidth) - 1;
    const uint64_t mant_mask = (MantWidth == 0) ? 0 : ((uint64_t(1) << MantWidth) - 1);
    const int64_t bias = (int64_t(1) << (ExpWidth - 1)) - 1;
    const int64_t emin = 1 - bias;
    const int64_t emax = bias;

    const auto pack = [&](uint64_t exp_bits, uint64_t frac_bits) -> uint32_t {
        return static_cast<uint32_t>(sign_bit | (exp_bits << MantWidth) | (frac_bits & mant_mask));
    };
    const auto pack_inf = [&]() -> uint32_t {
        return pack(exp_all_ones, 0);
    };
    const auto pack_max_finite = [&]() -> uint32_t {
        return pack(exp_all_ones - 1, mant_mask);
    };
    const auto overflow_result = [&]() -> uint32_t {
        const bool to_inf =
            (rm == RoundingMode::RNE) ||
            (rm == RoundingMode::RUP && !f.sign) ||
            (rm == RoundingMode::RDN && f.sign);
        return to_inf ? pack_inf() : pack_max_finite();
    };

    if (f.is_nan) {
        if constexpr (MantWidth == 0) return pack(exp_all_ones, 0);
        return pack(exp_all_ones, uint64_t(1) << (MantWidth - 1));
    }
    if (f.is_inf) return pack_inf();
    if (divisor == 0) return pack_inf();

    size_t n = f.mant.size();
    while (n > 0 && f.mant[n - 1] == 0) --n;
    if (n == 0) return static_cast<uint32_t>(sign_bit);

    const auto any_bits_below = [&](int64_t lim) -> bool {
        if (lim <= 0) return false;
        const size_t full_words = static_cast<size_t>(lim >> 6);
        const uint32_t rem_bits = static_cast<uint32_t>(lim & 63);
        const size_t words_to_check = (full_words < n) ? full_words : n;
        for (size_t i = 0; i < words_to_check; ++i) {
            if (f.mant[i] != 0) return true;
        }
        if (rem_bits != 0 && full_words < n) {
            const uint64_t mask = (uint64_t(1) << rem_bits) - 1;
            if ((f.mant[full_words] & mask) != 0) return true;
        }
        return false;
    };

    const auto shr_mant = [&](int64_t sh) -> vector<uint64_t> {
        const auto mant_end = std::next(f.mant.begin(), static_cast<std::ptrdiff_t>(n));
        if (sh <= 0) {
            return vector<uint64_t>(f.mant.begin(), mant_end);
        }
        const size_t ws = static_cast<size_t>(sh >> 6);
        const uint32_t bs = static_cast<uint32_t>(sh & 63);
        if (ws >= n) return {};
        vector<uint64_t> out(n - ws, 0);
        if (bs == 0) {
            for (size_t i = ws; i < n; ++i) out[i - ws] = f.mant[i];
        } else {
            for (size_t i = ws; i < n; ++i) {
                const size_t oi = i - ws;
                out[oi] |= (f.mant[i] >> bs);
                if (i + 1 < n) out[oi] |= (f.mant[i + 1] << (64 - bs));
            }
        }
        trimMant(out);
        return out;
    };

    const auto div_u32_to_u64 = [&](const vector<uint64_t>& src, uint64_t& q, uint32_t& r, bool& overflow64) {
        size_t sn = src.size();
        while (sn > 0 && src[sn - 1] == 0) --sn;
        if (sn == 0) {
            q = 0;
            r = 0;
            overflow64 = false;
            return;
        }
        overflow64 = false;
        uint64_t rem = 0;
        uint64_t q0 = 0;
        for (size_t i = sn; i-- > 0;) {
            const tpu_uint128_t cur = (static_cast<tpu_uint128_t>(rem) << 64) | src[i];
            const uint64_t qw = static_cast<uint64_t>(cur / divisor);
            rem = static_cast<uint64_t>(cur % divisor);
            if (i == 0) {
                q0 = qw;
            } else if (qw != 0) {
                overflow64 = true;
            }
        }
        q = q0;
        r = static_cast<uint32_t>(rem);
    };

    const uint32_t msb_in_top = 63u - static_cast<uint32_t>(__builtin_clzll(f.mant[n - 1]));
    const int64_t bitlen_n = static_cast<int64_t>((n - 1) * 64 + msb_in_top + 1);
    const uint32_t bitlen_d = 32u - static_cast<uint32_t>(__builtin_clz(divisor));

    const auto cmp_n_vs_d_shl = [&](int64_t sh) -> int {
        const int64_t bitlen_b = static_cast<int64_t>(bitlen_d) + sh;
        if (bitlen_n != bitlen_b) return (bitlen_n < bitlen_b) ? -1 : 1;

        const int64_t ws = sh >> 6;
        const uint32_t bs = static_cast<uint32_t>(sh & 63);
        for (size_t i = n; i-- > 0;) {
            const uint64_t nv = f.mant[i];
            uint64_t bv = 0;
            if (static_cast<int64_t>(i) == ws) bv |= (uint64_t(divisor) << bs);
            if (bs != 0 && static_cast<int64_t>(i) == ws + 1) bv |= (uint64_t(divisor) >> (64 - bs));
            if (nv != bv) return (nv < bv) ? -1 : 1;
        }
        return 0;
    };

    const auto cmp_n_shl_vs_d = [&](int64_t sh) -> int {
        const int64_t bitlen_a = bitlen_n + sh;
        if (bitlen_a != static_cast<int64_t>(bitlen_d)) {
            return (bitlen_a < static_cast<int64_t>(bitlen_d)) ? -1 : 1;
        }
        const uint64_t av = (f.mant[0] << static_cast<uint32_t>(sh));
        const uint64_t dv = divisor;
        if (av == dv) return 0;
        return (av < dv) ? -1 : 1;
    };

    const auto ge_pow2 = [&](int64_t k) -> bool {
        const int64_t t = k - f.exp;
        if (t >= 0) return cmp_n_vs_d_shl(t) >= 0;
        return cmp_n_shl_vs_d(-t) >= 0;
    };

    int64_t k = (bitlen_n - 1) + f.exp - (static_cast<int64_t>(bitlen_d) - 1);
    while (!ge_pow2(k)) --k;
    while (ge_pow2(k + 1)) ++k;

    if (k > emax) return overflow_result();

    const auto rounded_from_shift = [&](int64_t sh, uint64_t& out_q, bool& overflow64) {
        uint64_t q_floor = 0;
        bool rem_nonzero = false;
        bool gt_half = false;
        bool eq_half = false;
        overflow64 = false;

        if (sh >= 0) {
            const auto mant_end = std::next(f.mant.begin(), static_cast<std::ptrdiff_t>(n));
            vector<uint64_t> num = shlMant(vector<uint64_t>(f.mant.begin(), mant_end), sh);
            uint32_t rem = 0;
            div_u32_to_u64(num, q_floor, rem, overflow64);
            rem_nonzero = (rem != 0);
            const uint64_t twice = uint64_t(rem) << 1;
            gt_half = twice > divisor;
            eq_half = twice == divisor;
        } else {
            const int64_t rsh = -sh;
            vector<uint64_t> upper = shr_mant(rsh);
            uint32_t r1 = 0;
            div_u32_to_u64(upper, q_floor, r1, overflow64);
            const bool low_nonzero = any_bits_below(rsh);
            rem_nonzero = (r1 != 0) || low_nonzero;
            const uint64_t twice_r1 = uint64_t(r1) << 1;
            if (twice_r1 > divisor) {
                gt_half = true;
            } else if (twice_r1 == divisor) {
                gt_half = low_nonzero;
                eq_half = !low_nonzero;
            }
        }

        bool round_up = false;
        switch (rm) {
            case RoundingMode::RNE:
                round_up = gt_half || (eq_half && ((q_floor & 1u) != 0));
                break;
            case RoundingMode::RTZ:
                break;
            case RoundingMode::RUP:
                round_up = !f.sign && rem_nonzero;
                break;
            case RoundingMode::RDN:
                round_up = f.sign && rem_nonzero;
                break;
        }
        out_q = q_floor + static_cast<uint64_t>(round_up);
        if (round_up && out_q < q_floor) overflow64 = true;
    };

    if (k >= emin) {
        const int64_t sh = f.exp + static_cast<int64_t>(MantWidth) - k;
        uint64_t q = 0;
        bool of64 = false;
        rounded_from_shift(sh, q, of64);
        if (of64) return overflow_result();

        int64_t E = k;
        const uint64_t carry_threshold = uint64_t(1) << (MantWidth + 1);
        if (q >= carry_threshold) {
            q >>= 1;
            ++E;
            if (E > emax) return overflow_result();
        }
        return pack(static_cast<uint64_t>(E + bias), q & mant_mask);
    }

    const int64_t sh = f.exp + static_cast<int64_t>(MantWidth) - emin;
    uint64_t frac = 0;
    bool of64 = false;
    rounded_from_shift(sh, frac, of64);
    if (of64) return pack(1, 0);
    if (frac == 0) return static_cast<uint32_t>(sign_bit);
    if (frac > mant_mask) return pack(1, 0);
    return pack(0, frac);
}
