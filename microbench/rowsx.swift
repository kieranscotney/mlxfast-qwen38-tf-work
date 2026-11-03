import Metal
// Per-row cost of the 4-bit decode tile loop (gate|up shape, codes streamed from DRAM): one tile per SIMD group with x
// read from device memory each group (m0, as tf_lane_reg), two tiles per SIMD group reading x twice (m1), and two tiles
// sharing one x load in a cooperative left-input tensor (m2). Rows M in {2, 4, 8}.
let dev = MTLCreateSystemDefaultDevice()!
func src(_ mode: Int, _ rows: Int) -> String { return """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
inline void unpack(thread half2* b2, uint4 a, uint4 b) {
  for (int w = 0; w < 8; w++) {
    const uint word = w < 4 ? a[w] : b[w - 4];
    const int e = 16 * (w & 1) + 2 * (w >> 1);
    b2[e] = as_type<half2>(word & 0x000F000Fu); b2[e + 1] = as_type<half2>((word >> 4) & 0x000F000Fu);
    b2[e + 8] = as_type<half2>((word >> 8) & 0x000F000Fu); b2[e + 9] = as_type<half2>((word >> 12) & 0x000F000Fu);
  }
}
kernel void k(const device bfloat* X [[buffer(0)]], const device uint4* W [[buffer(1)]], device float* Y [[buffer(2)]],
  uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int K = 5120, GS = 64, KG = K / GS; const int M = \(rows);
  constexpr auto desc = matmul2d_descriptor(16, 32, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a((device bfloat*)X, dextents<int32_t, 2>(K, M));
  auto a0 = a.slice(0, 0);
  auto cb = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto cb2 = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto P = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a0)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  auto P2 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a0)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  float C[16] = {0}, D[16] = {0};
  const int tiles = \(mode) == 0 ? 1 : 2;
  const device uint4* w = W + (size_t)(tg * tiles) * KG * 64 + lane;
  const array<int32_t, 2> xs = {1, K};
  for (int g = 0; g < KG; g++) {
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> x((device bfloat*)X + g * GS, dextents<int32_t, 2>(GS, M), xs);
    unpack((thread half2*)&cb[0], w[(size_t)g * 64], w[(size_t)g * 64 + 32]);
    if (\(mode) == 0) {
      op.run(x, cb, P);
      for (int j = 0; j < 16; j++) C[j] = fma(1.0f, P[j], C[j]);
    } else {
      unpack((thread half2*)&cb2[0], w[(size_t)(KG + g) * 64], w[(size_t)(KG + g) * 64 + 32]);
      if (\(mode) == 1) {
        op.run(x, cb, P);
        op.run(x, cb2, P2);
      } else {
        auto ca = op.template get_left_input_cooperative_tensor<bfloat, half, float>();
        ca.load(x);
        op.run(ca, cb, P);
        op.run(ca, cb2, P2);
      }
      for (int j = 0; j < 16; j++) { C[j] = fma(1.0f, P[j], C[j]); D[j] = fma(1.0f, P2[j], D[j]); }
    }
  }
  float s = 0; for (int j = 0; j < 16; j++) s += C[j] + D[j];
  Y[tg * 32 + lane] = s;
}
""" }
let N = 34816, K = 5120, tiles = N / 32
let X = dev.makeBuffer(length: K * 16 * 2, options: .storageModeShared)!
let W = dev.makeBuffer(length: tiles * (K / 64) * 64 * 16, options: .storageModePrivate)!   // 89 MB of codes, streamed
let Y = dev.makeBuffer(length: tiles * 32 * 4, options: .storageModeShared)!
let q = dev.makeCommandQueue()!
for rows in [2, 4, 8] {
  for (mode, name) in [(0, "m0 one tile, x per group"), (1, "m1 two tiles, x twice"), (2, "m2 two tiles, x once")] {
    let lib: MTLLibrary
    do { lib = try dev.makeLibrary(source: src(mode, rows), options: nil) } catch { print("rows \(rows) \(name): compile failed: \(error)"); continue }
    let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "k")!)
    var best = 1e9
    for _ in 0..<15 {
      let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
      e.setComputePipelineState(p); e.setBuffer(X, offset: 0, index: 0); e.setBuffer(W, offset: 0, index: 1); e.setBuffer(Y, offset: 0, index: 2)
      e.dispatchThreadgroups(MTLSize(width: mode == 0 ? tiles : tiles / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      best = min(best, cb.gpuEndTime - cb.gpuStartTime)
    }
    let gbs = Double(N * K / 2) / best / 1e9
    print("rows \(rows) \(name): \(String(format: "%.1f", best * 1e6)) us = \(String(format: "%.0f", gbs)) GB/s of codes")
  }
}
