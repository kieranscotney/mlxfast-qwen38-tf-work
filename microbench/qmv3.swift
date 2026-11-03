import Metal
// mlx-kquant verify_qmv geometry on our 2-bit affine layout: MLX row-major codes (16 a u32), bf16 scale/bias per 64.
let dev = MTLCreateSystemDefaultDevice()!
let N = 34816, K = 5120
func src(_ R: Int, _ SGS: Int) -> String { return """
#include <metal_stdlib>
using namespace metal;
constant constexpr int K = \(K), N = \(N), R = \(R), SGS = \(SGS), G = K / 64;
kernel void qmv(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]], const device bfloat* S [[buffer(2)]],
                const device bfloat* B [[buffer(3)]], const device float* XS [[buffer(4)]], device bfloat* Y [[buffer(5)]],
                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tg [[threadgroup_position_in_grid]]) {
  const int n0 = (int(tg) * SGS + int(sg)) * R;
  const int ix = lane / 4, it = lane % 4;          // 8 groups a pass, 4 lanes a group, 16 codes a lane
  float acc0[R], acc1[R];
  for (int r = 0; r < R; r++) { acc0[r] = 0; acc1[r] = 0; }
  for (int g = ix; g < G; g += 8) {
    const int k = g * 64 + it * 16;
    const vec<bfloat, 8> l0 = *(const device vec<bfloat, 8>*)(X + k), h0 = *(const device vec<bfloat, 8>*)(X + k + 8);
    const vec<bfloat, 8> l1 = *(const device vec<bfloat, 8>*)(X + K + k), h1 = *(const device vec<bfloat, 8>*)(X + K + k + 8);
    float2 x0[8], x1[8];
    for (int j = 0; j < 8; j++) { x0[j] = float2(float(j < 4 ? l0[2 * j] : h0[2 * j - 8]), float(j < 4 ? l0[2 * j + 1] : h0[2 * j - 7])); x1[j] = float2(float(j < 4 ? l1[2 * j] : h1[2 * j - 8]), float(j < 4 ? l1[2 * j + 1] : h1[2 * j - 7])); }
    const float xs0 = XS[g], xs1 = XS[G + g];
    for (int r = 0; r < R; r++) {
      const uint w = W[(size_t)(n0 + r) * (K / 16) + k / 16];
      float2 p0 = 0, p1 = 0;
      for (int j = 0; j < 8; j++) {
        const uint2 q2 = uint2((w >> (4 * j)) & 3u, (w >> (4 * j + 2)) & 3u);
        const float2 qf = float2(q2);
        p0 = fma(qf, x0[j], p0); p1 = fma(qf, x1[j], p1);
      }
      const float s = float(S[(size_t)(n0 + r) * G + g]), b = float(B[(size_t)(n0 + r) * G + g]);
      acc0[r] += s * (p0.x + p0.y) + (it == 0 ? b * xs0 : 0.0f);
      acc1[r] += s * (p1.x + p1.y) + (it == 0 ? b * xs1 : 0.0f);
    }
  }
  for (int r = 0; r < R; r++) {
    const float a = simd_sum(acc0[r]), c = simd_sum(acc1[r]);
    if (lane == 0) { Y[n0 + r] = bfloat(a); Y[N + n0 + r] = bfloat(c); }
  }
}
""" }
let X = dev.makeBuffer(length: 2 * K * 2, options: .storageModeShared)!
let XS = dev.makeBuffer(length: 2 * (K / 64) * 4, options: .storageModeShared)!
let W = dev.makeBuffer(length: N * K / 4, options: .storageModePrivate)!
let S = dev.makeBuffer(length: N * (K / 64) * 2, options: .storageModePrivate)!
let B = dev.makeBuffer(length: N * (K / 64) * 2, options: .storageModePrivate)!
let Y = dev.makeBuffer(length: 2 * N * 2, options: .storageModeShared)!
let q = dev.makeCommandQueue()!
for (R, SGS) in [(1, 2), (2, 2), (4, 2), (2, 4), (4, 4), (8, 2)] {
  let lib = try! dev.makeLibrary(source: src(R, SGS), options: nil)
  let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "qmv")!)
  var best = 1e9
  for _ in 0..<12 {
    let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    for (i, b) in [X, W, S, B, XS, Y].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.dispatchThreadgroups(MTLSize(width: N / (R * SGS), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * SGS, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  let bytes = Double(W.length + S.length + B.length)
  print("R=\(R) SGS=\(SGS): \(String(format: "%.1f", best * 1e6)) us  \(String(format: "%.0f", bytes / best / 1e9)) GB/s  (tensor path ~120 us)")
}
