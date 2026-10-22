// Full labelled-permutation Cayley graph; exact sequential-edge-swap BFS.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void write(const std::string& path, const void* ptr, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(ptr), bytes);
    if (!out) throw std::runtime_error("Cannot write oracle array");
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream in(argv[1]);
    int n, m; in >> n >> m;
    if (!in || n < 2 || n > 10 || m < 1) return 2;
    std::vector<std::pair<int,int>> edges(m);
    for (auto& edge : edges) {
        in >> edge.first >> edge.second;
        if (edge.first < 0 || edge.second >= n || edge.first >= edge.second) return 2;
    }
    uint32_t factorial[11] = {1};
    for (int i = 1; i <= n; ++i) factorial[i] = factorial[i-1]*i;
    const uint32_t count = factorial[n];
    std::vector<uint8_t> states(size_t(count)*n);
    std::vector<int32_t> children(size_t(count)*m);
    auto rank = [&](const uint8_t* p) {
        uint32_t id = 0, mask = (1u<<n)-1;
        for (int i = 0; i < n; ++i) {
            id += __builtin_popcount(mask & ((1u<<p[i])-1))*factorial[n-1-i];
            mask &= ~(1u<<p[i]);
        }
        return id;
    };
    for (uint32_t id = 0; id < count; ++id) {
        uint32_t left = id, mask = (1u<<n)-1;
        uint8_t* p = &states[size_t(id)*n];
        for (int i = 0; i < n; ++i) {
            uint32_t digit = left/factorial[n-1-i];
            left %= factorial[n-1-i];
            int value = 0;
            for (; value < n; ++value) if (mask & (1u<<value)) {
                if (digit == 0) break;
                --digit;
            }
            p[i] = value; mask &= ~(1u<<value);
        }
        for (int e = 0; e < m; ++e) {
            const auto [a,b] = edges[e];
            std::swap(p[a],p[b]); children[size_t(id)*m+e] = rank(p); std::swap(p[a],p[b]);
        }
    }
    std::vector<uint8_t> distances(count,255);
    std::vector<uint32_t> queue(count);
    size_t head=0,tail=1; queue[0]=0; distances[0]=0;
    while (head<tail) {
        uint32_t state=queue[head++];
        for (int e=0;e<m;++e) {
            uint32_t child=children[size_t(state)*m+e];
            if (distances[child]==255) {
                if (distances[state]>=254) throw std::runtime_error("Distance byte overflow");
                distances[child]=distances[state]+1; queue[tail++]=child;
            }
        }
    }
    if (tail!=count) throw std::runtime_error("Graph not connected: no complete permutation oracle");
    std::string out=argv[2];
    write(out+"/states.bin",states.data(),states.size());
    write(out+"/children.bin",children.data(),children.size()*sizeof(int32_t));
    write(out+"/distances.bin",distances.data(),distances.size());
    std::cout<<"{\"states\":"<<count<<",\"edges\":"<<m<<",\"visited\":"<<tail
             <<",\"diameter\":"<<int(*std::max_element(distances.begin(),distances.end()))<<"}\n";
}
