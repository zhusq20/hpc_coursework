#include "aplusb.h"
#include <x86intrin.h>

void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        // Load 8 floats from array a and b at the index i
        __m256 a_vec = _mm256_load_ps(a + i);
        __m256 b_vec = _mm256_load_ps(b + i);

        // Add the two vector registers element-wise
        __m256 c_vec = _mm256_add_ps(a_vec, b_vec);

        // Store the result back into array c at the index i
        _mm256_store_ps(c + i, c_vec);
    }
}