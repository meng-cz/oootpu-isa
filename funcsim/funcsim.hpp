#pragma once

#include "oootpu.hpp"

namespace oootpu {

constexpr TRegIdx TREG_NUM = 512; // 假设有512个Tensor寄存器，实际数量根据具体实现调整

class TMemProxy {
public:
    virtual ~TMemProxy() = default;
    virtual void read(TMemAddr addr, void *data, uint32_t size) const = 0;
    virtual void write(TMemAddr addr, const void *data, uint32_t size) = 0;
};

void setMemProxy(TMemProxy *proxy);

}

