import Metal
// Decomposition of the 2-bit decode kernel's per-group work, weights cache-resident (a small W block reused).
let dev = MTLCreateSystemDefaultDevice()!
func src(_ mode: Int) -> String { return """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void k(const device bfloat* X [[buffer(0)]], const device uint4* W [[buffer(1)]], device float* Y [[buffer(2)]],
  constant uint& G [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int K = 5120; const int M = 2;
  constexpr auto desc = matmul2d_descriptor(16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a((device bfloat*)X, dextents<int32_t, 2>(K, M));
  auto a0 = a.slice(0, 0);
  auto cb = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto P = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a0)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  thread half2* b2 = (thread half2*)&cb[0];
  for (int i = 0; i < 32; i++) b2[i] = half2(0.0h);
  float C[16] = {0};
  const device uint4* w = W + (tg & 63) * 80 * 32 + lane;   // 64 blocks reused: cache-resident
  for (uint g = 0; g < G; g++) {
    const uint4 w0 = w[(g % 80) * 32];
    if (\(mode) != 1) {
      for (int i = 0; i < 8; i++) {
        const uint word = i < 4 ? (w0[i] & 0x33333333u) : ((w0[i - 4] >> 2) & 0x33333333u);
        const int e = 16 * (i & 1) + 2 * (i >> 1);
        b2[e] = as_type<half2>(word & 0x000F000Fu); b2[e + 1] = as_type<half2>((word >> 4) & 0x000F000Fu);
        b2[e + 8] = as_type<half2>((word >> 8) & 0x000F000Fu); b2[e + 9] = as_type<half2>((word >> 12) & 0x000F000Fu);
      }
    } else { b2[g & 31] = as_type<half2>(w0.x); }
    if (\(mode) != 2) {
      const array<int32_t, 2> xs = {1, K};
      tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> x((device bfloat*)X + (g % 80) * 64, dextents<int32_t, 2>(64, M), xs);
      op.run(x, cb, P);
      for (int j = 0; j < 16; j++) C[j] = fma(1.0f, P[j], C[j]);
    } else {
      for (int j = 0; j < 16; j++) C[j] += float(b2[j].x) + float(b2[j + 16].y);
    }
  }
  float s = 0; for (int j = 0; j < 16; j++) s += C[j];
  Y[tg * 32 + lane] = s;
}
""" }
let tgs = 4352, G: UInt32 = 80   // 34816 columns / 32 = 1088 tiles; x4 for a long run
let X = dev.makeBuffer(length: 5120 * 16 * 2, options: .storageModeShared)!
let W = dev.makeBuffer(length: 64 * 80 * 32 * 16, options: .storageModePrivate)!
let Y = dev.makeBuffer(length: tgs * 32 * 4, options: .storageModeShared)!
let q = dev.makeCommandQueue()!
for (mode, name) in [(0, "unpack+MMA"), (1, "MMA only"), (2, "unpack only")] {
  let lib = try! dev.makeLibrary(source: src(mode), options: nil)
  let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "k")!)
  var g = G; var best = 1e9
  for _ in 0..<10 {
    let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p); e.setBuffer(X, offset: 0, index: 0); e.setBuffer(W, offset: 0, index: 1); e.setBuffer(Y, offset: 0, index: 2)
    e.setBytes(&g, length: 4, index: 3)
    e.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  let weights = Double(tgs) * Double(G) * 2048
  print("\(name): \(String(format: "%.1f", best * 1e6)) us for \(String(format: "%.0f", weights / 1e6))M weights = \(String(format: "%.2f", weights / best / 1e12)) T w/s")
}
