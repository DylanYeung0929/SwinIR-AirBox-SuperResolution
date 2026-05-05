#include <iostream>
#include <algorithm>
#include <cmath>

extern "C" {
    // 真实的高性能 C++ 后处理函数
    // 直接对内存指针进行操作，完全跳过 Python 层的内存再分配
    void cpp_hardware_postprocess(float* input_fp32, unsigned char* output_uint8, int total_elements, float scale_factor, float bias) {
        
        // 使用 OpenMP 甚至简单的单线程内存连续遍历，都比 Numpy 快得多
        for(int i = 0; i < total_elements; ++i) {
            // 1. 尺度还原
            float val = input_fp32[i] * scale_factor + bias;
            
            // 2. 硬件级 Saturate (截断)
            val = std::max(0.0f, std::min(255.0f, val));
            
            // 3. 强制类型转换
            output_uint8[i] = static_cast<unsigned char>(val);
        }
    }
}
