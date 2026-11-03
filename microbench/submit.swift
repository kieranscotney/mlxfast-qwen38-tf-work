import Metal
import Foundation
// Submission round trip on this GPU: commit a one-threadgroup kernel that signals a shared event, spin until the host
// sees it. Classic MTLCommandQueue vs Metal 4 MTL4CommandQueue. 300 trials each, median and p10 in microseconds.
let dev = MTLCreateSystemDefaultDevice()!
let src = "kernel void k(device uint* y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] = i; }"
let lib = try! dev.makeLibrary(source: src, options: nil)
let fn = lib.makeFunction(name: "k")!
let pso = try! dev.makeComputePipelineState(function: fn)
let buf = dev.makeBuffer(length: 4096, options: .storageModeShared)!
let ev = dev.makeSharedEvent()!
var value: UInt64 = 0
func now() -> Double { return Double(DispatchTime.now().uptimeNanoseconds) / 1e3 }
func report(_ name: String, _ xs: [Double]) {
  let s = xs.sorted()
  print(String(format: "%@: median %.1f us, p10 %.1f, p90 %.1f", name, s[s.count / 2], s[s.count / 10], s[9 * s.count / 10]))
}
// classic
let q = dev.makeCommandQueue()!
var classic: [Double] = []
for _ in 0..<300 {
  let cb = q.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(pso); e.setBuffer(buf, offset: 0, index: 0)
  e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  e.endEncoding()
  value += 1; cb.encodeSignalEvent(ev, value: value)
  let t0 = now(); cb.commit()
  while ev.signaledValue < value {}
  classic.append(now() - t0)
}
report("classic queue commit -> event seen", classic)
// Metal 4
if #available(macOS 26.0, *) {
  let q4 = dev.makeMTL4CommandQueue()!
  let alloc = dev.makeCommandAllocator()!
  let cb4 = dev.makeCommandBuffer()!
  let table = try! dev.makeArgumentTable(descriptor: { let d = MTL4ArgumentTableDescriptor(); d.maxBufferBindCount = 1; return d }())
  table.setAddress(buf.gpuAddress, index: 0)
  var m4: [Double] = []
  for _ in 0..<300 {
    alloc.reset()
    cb4.beginCommandBuffer(allocator: alloc)
    let e = cb4.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setArgumentTable(table)
    e.dispatchThreadgroups(threadgroupsPerGrid: MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    e.endEncoding()
    cb4.endCommandBuffer()
    value += 1
    let t0 = now()
    q4.commit([cb4])
    q4.signalEvent(ev, value: value)
    while ev.signaledValue < value {}
    m4.append(now() - t0)
  }
  report("Metal 4 queue commit -> event seen", m4)
} else { print("Metal 4 unavailable") }
