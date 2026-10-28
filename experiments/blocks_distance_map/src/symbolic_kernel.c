#include <stdint.h>
#include <stddef.h>

static int contains(const int16_t *level, const int32_t *low,
                    const int32_t *high, int32_t node, uint64_t mask) {
    while (node >= 2) node = (mask >> level[node]) & 1 ? high[node] : low[node];
    return node == 1;
}

int bdd_rank(const int16_t *level, const int32_t *low, const int32_t *high,
             const uint64_t *counts, int32_t root, int width, size_t n,
             const uint64_t *masks, int64_t *result) {
    for (size_t i = 0; i < n; ++i) {
        int32_t node = root;
        uint64_t index = 0;
        for (int bit = 0; bit < width; ++bit) {
            int split = level[node] == bit;
            int32_t left = split ? low[node] : node;
            if ((masks[i] >> bit) & 1) {
                index += counts[left] << (level[left] - bit - 1);
                node = split ? high[node] : node;
            } else node = left;
        }
        if (node != 1) return 1;
        result[i] = (int64_t)index;
    }
    return 0;
}

int bdd_unrank(const int16_t *level, const int32_t *low, const int32_t *high,
               const uint64_t *counts, int32_t root, int width, size_t n,
               const uint64_t *indices, uint64_t *result) {
    for (size_t i = 0; i < n; ++i) {
        int32_t node = root;
        uint64_t index = indices[i], mask = 0;
        for (int bit = 0; bit < width; ++bit) {
            int split = level[node] == bit;
            int32_t left = split ? low[node] : node;
            uint64_t count = counts[left] << (level[left] - bit - 1);
            if (index >= count) {
                mask |= UINT64_C(1) << bit;
                index -= count;
                node = split ? high[node] : node;
            } else node = left;
        }
        if (node != 1) return 1;
        result[i] = mask;
    }
    return 0;
}

void bdd_contains(const int16_t *level, const int32_t *low, const int32_t *high,
                  int32_t root, size_t n, const uint64_t *masks, uint8_t *result) {
    for (size_t i = 0; i < n; ++i) result[i] = contains(level, low, high, root, masks[i]);
}

void bdd_distance(const int16_t *level, const int32_t *low, const int32_t *high,
                  const int32_t *layers, int layer_count, int width, size_t n,
                  const uint64_t *source, const uint64_t *goal, int16_t *result) {
    uint64_t full = width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1;
    for (size_t i = 0; i < n; ++i) {
        result[i] = -1;
        if ((source[i] & goal[i]) != goal[i]) continue;
        uint64_t target = full ^ (source[i] ^ goal[i]);
        for (int d = 0; d < layer_count; ++d) {
            if (contains(level, low, high, layers[d], target)) { result[i] = d; break; }
        }
    }
}
