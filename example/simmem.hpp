
#pragma once

#include "oootpu.hpp"
#include "funcsim/funcsim.hpp"

#include <vector>

#include <assert.h>
#include <string.h>

class SimMemory : public oootpu::TMemProxy {
public:
    void read(oootpu::TMemAddr addr, void *data, uint32_t size) const override {
        assert(addr >= 0 && "访问地址不能为负");
        const size_t offset = static_cast<size_t>(addr);
        assert((offset + size) <= memory.size() && "访问越界");
        memcpy(data, memory.data() + offset, size);
    }

    void write(oootpu::TMemAddr addr, const void *data, uint32_t size) override {
        assert(addr >= 0 && "访问地址不能为负");
        const size_t offset = static_cast<size_t>(addr);
        assert((offset + size) <= memory.size() && "访问越界");
        memcpy(memory.data() + offset, data, size);
    }

    SimMemory(size_t size) : memory(size, 0) {}

    std::vector<uint8_t> memory;
};
