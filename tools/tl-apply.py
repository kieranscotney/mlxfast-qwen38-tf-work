#!/usr/bin/env python3
# GPU timeline probe: "TLV start end busy chunks" per verify (its command buffers' GPU times, logged when checked at
# the next pass), "TLF start end" per drafter frame, "TLK start end" per keep. Times are seconds on Metal's clock.
seg = 'Vendor/TensorFold/zig/src/core/segments.zig'
s = open(seg).read()
old = '''pub fn checkUnchecked() !void {
    var failed = false;
'''
assert old in s
s = s.replace(old, '''pub fn checkUnchecked() !void {
    var failed = false;
    if (n_unchecked > 0) {
        var t0: f64 = 1e30;
        var t1: f64 = 0;
        var busy: f64 = 0;
        for (unchecked[0..n_unchecked]) |cb| {
            if (cb.status() != .completed) cb.wait();
            const a = cb.gpuStart();
            const b = cb.gpuEnd();
            t0 = @min(t0, a);
            t1 = @max(t1, b);
            busy += b - a;
        }
        std.log.warn("TLV {d:.6} {d:.6} {d:.6} {d}", .{ t0, t1, busy, n_unchecked });
    }
''', 1)
open(seg, 'w').write(s)
fr = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/runtime_frame.zig'
s = open(fr).read()
old = '''    pub fn checkUnchecked(f: *Frame) !void {
        var bad = false;
'''
assert old in s
s = s.replace(old, '''    pub fn checkUnchecked(f: *Frame) !void {
        var bad = false;
        if (f.n_unchecked > 0) {
            var t0: f64 = 1e30;
            var t1: f64 = 0;
            for (f.unchecked[0..f.n_unchecked]) |cb| {
                if (cb.status() != .completed) cb.wait();
                t0 = @min(t0, cb.gpuStart());
                t1 = @max(t1, cb.gpuEnd());
            }
            std.log.warn("TLF {d:.6} {d:.6}", .{ t0, t1 });
        }
''', 1)
open(fr, 'w').write(s)
g = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/generation.zig'
s = open(g).read()
old = '''        cb.wait();
        if (gpuFailed(cb, "a draft round's keep")) {'''
assert old in s
s = s.replace(old, '''        cb.wait();
        std.log.warn("TLK {d:.6} {d:.6}", .{ cb.gpuStart(), cb.gpuEnd() });
        if (gpuFailed(cb, "a draft round's keep")) {''', 1)
open(g, 'w').write(s)
print('timeline probe applied')
