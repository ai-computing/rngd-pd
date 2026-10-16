/* Rebel 0.11.0 DLFloat16 conversion. No device pointers or DMA operations. */
#include <stdint.h>
#include <stddef.h>
int bf16_to_dlfloat(const uint16_t *restrict input, uint16_t *restrict output, size_t n) {
    unsigned invalid = 0;
    for (size_t i = 0; i < n; ++i) {
        unsigned x = input[i], exp = (x >> 7) & 255u, sign = x & 0x8000u;
        unsigned normal = sign | ((exp - 96u) << 9) | ((x & 127u) << 2);
        output[i] = (uint16_t)(exp < 96u ? sign : exp > 159u ? sign | 0x7fffu : normal);
        invalid |= exp == 255u;
    }
    return invalid ? -1 : 0;
}
