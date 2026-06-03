#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tpuapprox {

template <uint32_t ExpWidth, uint32_t MantWidth>
inline double toHardF64(uint32_t value) {
    static_assert(ExpWidth > 0, "ExpWidth must be greater than 0");
    static_assert(1u + ExpWidth + MantWidth <= 32u, "Input float format must fit in 32 bits");

    constexpr auto mask = [](uint32_t bits) -> uint64_t {
        return bits == 64u ? ~0ull : ((1ull << bits) - 1ull);
    };

    constexpr uint32_t kSignShift = ExpWidth + MantWidth;
    constexpr uint32_t kExpMask = static_cast<uint32_t>(mask(ExpWidth));
    constexpr uint32_t kMantMask = static_cast<uint32_t>(mask(MantWidth));
    constexpr int32_t kBias = static_cast<int32_t>((1u << (ExpWidth - 1u)) - 1u);

    const uint32_t sign = (value >> kSignShift) & 1u;
    const uint32_t exp = (value >> MantWidth) & kExpMask;
    const uint32_t mant = value & kMantMask;

    if (exp == kExpMask) {
        if (mant == 0u) {
            return sign ? -std::numeric_limits<double>::infinity()
                        : std::numeric_limits<double>::infinity();
        }
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return sign ? -nan : nan;
    }

    if (exp == 0u) {
        if (mant == 0u) {
            return sign ? -0.0 : 0.0;
        }
        const double frac = static_cast<double>(mant) *
                            std::ldexp(1.0, -static_cast<int>(MantWidth));
        const int e = 1 - kBias;
        const double v = std::ldexp(frac, e);
        return sign ? -v : v;
    }

    const double frac = static_cast<double>(mant) *
                        std::ldexp(1.0, -static_cast<int>(MantWidth));
    const int e = static_cast<int>(exp) - kBias;
    const double v = std::ldexp(1.0 + frac, e);
    return sign ? -v : v;
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t fromHardF64(double value) {
    static_assert(ExpWidth > 0, "ExpWidth must be greater than 0");
    static_assert(1u + ExpWidth + MantWidth <= 32u, "Output float format must fit in 32 bits");

    constexpr auto mask = [](uint32_t bits) -> uint32_t {
        return bits == 32u ? 0xffffffffu : ((1u << bits) - 1u);
    };

    constexpr uint32_t kSignShift = ExpWidth + MantWidth;
    constexpr uint32_t kExpMask = mask(ExpWidth);
    constexpr uint32_t kMantMask = mask(MantWidth);
    constexpr int32_t kBias = static_cast<int32_t>((1u << (ExpWidth - 1u)) - 1u);
    constexpr int32_t kMinNormalExp = 1 - kBias;
    constexpr int32_t kMaxNormalExp = static_cast<int32_t>(kExpMask - 1u) - kBias;
    constexpr uint32_t kPrecision = MantWidth + 1u;

    uint64_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));

    const uint32_t sign = static_cast<uint32_t>(raw >> 63);
    const uint32_t exp = static_cast<uint32_t>((raw >> 52) & 0x7ffu);
    const uint64_t frac = raw & ((1ull << 52) - 1ull);

    const uint32_t signPart = sign << kSignShift;

    if (exp == 0x7ffu) {
        if (frac == 0ull) {
            return signPart | (kExpMask << MantWidth);
        }
        const uint32_t qnanMant = MantWidth ? (1u << (MantWidth - 1u)) : 0u;
        return signPart | (kExpMask << MantWidth) | qnanMant;
    }

    if (exp == 0u && frac == 0ull) {
        return signPart;
    }

    int32_t e = 0;
    uint64_t sig = 0;
    if (exp != 0u) {
        e = static_cast<int32_t>(exp) - 1023;
        sig = (1ull << 52) | frac;
    } else {
        const int lz = __builtin_clzll(frac);
        const int msb = 63 - lz;
        const int shift = 52 - msb;
        sig = frac << shift;
        e = -1022 - shift;
    }

    if (e > kMaxNormalExp) {
        return signPart | (kExpMask << MantWidth);
    }

    if (e >= kMinNormalExp) {
        constexpr uint32_t kDropBits = 53u - kPrecision;
        uint64_t mainSig = sig >> kDropBits;
        const uint64_t remMask = (1ull << kDropBits) - 1ull;
        const uint64_t rem = sig & remMask;
        const uint64_t half = 1ull << (kDropBits - 1u);
        if (rem > half || (rem == half && (mainSig & 1ull))) {
            ++mainSig;
        }

        if (mainSig == (1ull << kPrecision)) {
            mainSig >>= 1u;
            ++e;
            if (e > kMaxNormalExp) {
                return signPart | (kExpMask << MantWidth);
            }
        }

        const uint32_t outExp = static_cast<uint32_t>(e + kBias);
        const uint32_t outMant = static_cast<uint32_t>(mainSig) & kMantMask;
        return signPart | (outExp << MantWidth) | outMant;
    }

    const int32_t shift = static_cast<int32_t>(52u - MantWidth) + (kMinNormalExp - e);
    if (shift >= 64) {
        return signPart;
    }

    uint64_t outMant64 = sig >> shift;
    if (shift > 0) {
        const uint64_t remMask = (1ull << shift) - 1ull;
        const uint64_t rem = sig & remMask;
        const uint64_t half = 1ull << (shift - 1);
        if (rem > half || (rem == half && (outMant64 & 1ull))) {
            ++outMant64;
        }
    }

    const uint64_t normalThreshold = (MantWidth == 0u) ? 1ull : (1ull << MantWidth);
    if (outMant64 >= normalThreshold) {
        return signPart | (1u << MantWidth);
    }

    const uint32_t outMant = static_cast<uint32_t>(outMant64) & kMantMask;
    return signPart | outMant;
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t fsqrt_appr(uint32_t value) {
    double v = toHardF64<ExpWidth, MantWidth>(value);
    double sqrt_v = std::sqrt(v);
    return fromHardF64<ExpWidth, MantWidth>(sqrt_v);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t frsqrt_appr(uint32_t value) {
    double v = toHardF64<ExpWidth, MantWidth>(value);
    double rsqrt_v = 1.0 / std::sqrt(v);
    return fromHardF64<ExpWidth, MantWidth>(rsqrt_v);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t fexp_appr(uint32_t value) {
    double v = toHardF64<ExpWidth, MantWidth>(value);
    double exp_v = std::exp(v);
    return fromHardF64<ExpWidth, MantWidth>(exp_v);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t frcp_appr(uint32_t value) {
    double v = toHardF64<ExpWidth, MantWidth>(value);
    double rcp_v = 1.0 / v;
    return fromHardF64<ExpWidth, MantWidth>(rcp_v);
}


template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t fsigmoid_appr(uint32_t value) {
    double v = toHardF64<ExpWidth, MantWidth>(value);
    double sigmoid_v = 1.0 / (1.0 + std::exp(-v));
    return fromHardF64<ExpWidth, MantWidth>(sigmoid_v);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t ftanh_appr(uint32_t value) {
    double v = toHardF64<ExpWidth, MantWidth>(value);
    double tanh_v = std::tanh(v);
    return fromHardF64<ExpWidth, MantWidth>(tanh_v);
}

template <uint32_t ExpWidth, uint32_t MantWidth>
inline uint32_t fgelu_appr(uint32_t value) {
    double v = toHardF64<ExpWidth, MantWidth>(value);
    double gelu_v = 0.5 * v * (1.0 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
    return fromHardF64<ExpWidth, MantWidth>(gelu_v);
}

}
