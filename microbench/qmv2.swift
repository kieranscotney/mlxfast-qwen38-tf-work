import Metal
// 2-bit, 2-row qmv for the drafter's gate/up shape: row-major codes (16 a u32, low bits first), per-64 group fp16 scale/bias
let dev = MTLCreateSystemDefaultDevice()!
let N = 34816, K = 5120
func src(_ R: Int) -> String { return """
#include <metal_stdlib>
using namespace metal;
constant constexpr int K = \(K);
constant constexpr int R = \(R);   // output rows a SIMD group
kernel void qmv2(const device float* X [[buffer(0)]], const device uint* W [[buffer(1)]], const device half2* SB [[buffer(2)]],
                 const device float* XS [[buffer(3)]], device float* Y [[buffer(4)]],
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tg [[threadgroup_position_in_grid]]) {
  const uint n0 = (tg * 4 + sg) * R;
  float a0[R], a1[R];
  for (int r = 0; r < R; r++) { a0[r] = 0; a1[r] = 0; }
  // a lane takes 16 consecutive k a step: 32 lanes cover 512 k, 10 steps cover K
  for (int base = 0; base < K; base += 512) {
    const int k0 = base + lane * 16;
    // activations pre-scaled for the in-place masks: code j of a byte is (w & (3 << 2j)) = q * 4^j
    float x0[16], x1[16];
    for (int i = 0; i < 16; i++) { const float sc = 1.0f / float(1 << (2 * (i & 3))); x0[i] = X[k0 + i] * sc; x1[i] = X[K + k0 + i] * sc; }
    const int g = k0 / 64;
    const float xs0 = XS[g], xs1 = XS[K / 64 + g];
    for (int r = 0; r < R; r++) {
      const uint w = W[(size_t)(n0 + r) * (K / 16) + k0 / 16];
      float p0 = 0, p1 = 0;
      for (int i = 0; i < 16; i++) {
        const float q = float(w & (3u << (2 * i)) >> (8 * (i >> 2)));  // byte i>>2, kept scaled by 4^(i&3)
        p0 = fma(x0[i], q, p0); p1 = fma(x1[i], q, p1);
      }
      const half2 sb = SB[(size_t)(n0 + r) * (K / 64) + g];
      a0[r] += float(sb.x) * p0 + float(sb.y) * xs0;
      a1[r] += float(sb.x) * p1 + float(sb.y) * xs1;
    }
  }
  for (int r = 0; r < R; r++) {
    const float s0 = simd_sum(a0[r]), s1 = simd_sum(a1[r]);
    if (lane == 0) { Y[n0 + r] = s0; Y[N_ + n0 + r] = s1; }
  }
}
""".replacingOccurrences(of: "N_", with: "\(N)") }
let X = dev.makeBuffer(length: 2 * K * 4, options: .storageModeShared)!
let XS = dev.makeBuffer(length: 2 * (K / 64) * 4, options: .storageModeShared)!
let W = dev.makeBuffer(length: N * K / 4, options: .storageModePrivate)!
let SB = dev.makeBuffer(length: N * (K / 64) * 4, options: .storageModePrivate)!
let Y = dev.makeBuffer(length: 2 * N * 4, options: .storageModeShared)!
let q = dev.makeCommandQueue()!
for R in [2, 4, 8] {
  do {
    let lib = try dev.makeLibrary(source: src(R), options: nil)
    let p = try dev.makeComputePipelineState(function: lib.makeFunction(name: "qmv2")!)
    var best = 1e9
    for _ in 0..<10 {
      let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
      e.setComputePipelineState(p); e.setBuffer(X, offset: 0, index: 0); e.setBuffer(W, offset: 0, index: 1); e.setBuffer(SB, offset: 0, index: 2); e.setBuffer(XS, offset: 0, index: 3); e.setBuffer(Y, offset: 0, index: 4)
      e.dispatchThreadgroups(MTLSize(width: N / (4 * R), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      if let err = cb.error { print("R=\(R) error \(err)"); break }
      best = min(best, cb.gpuEndTime - cb.gpuStartTime)
    }
    let bytes = Double(W.length + SB.length)
    print("R=\(R): \(String(format: "%.1f", best * 1e6)) us  \(String(format: "%.0f", bytes / best / 1e9)) GB/s  (current tensor path ~120 us, floor ~\(String(format: "%.0f", bytes / 580e9 * 1e6)) us)")
  } catch { print("R=\(R) compile error \(error)") }
}
