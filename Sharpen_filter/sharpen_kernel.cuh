#pragma once
#include <cstdint>

void sharpen_rgb_u8(const std::uint8_t* d_in,
                    std::uint8_t* d_out,
                    int width,
                    int height,
                    float amount);
