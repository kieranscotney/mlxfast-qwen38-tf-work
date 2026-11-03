#!/usr/bin/env python3
# Inserts per-round wall timing (p10/p25/p50/mean every 512 rounds, "RT" lines) into dflash/generation.zig of the
# current tree. Same instrumentation on every branch, so builds compare on per-round percentiles robust to bursts.
import sys
p = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/generation.zig'
s = open(p).read()
a1 = 'const accept_step: f32 = 0.125;\n'
assert a1 in s and 'var rt_' not in s
s = s.replace(a1, a1 + '''var rt_ms: [512]f32 = undefined;
var rt_n: usize = 0;
var rt_last: f64 = 0;
var rt_landed: usize = 0;
var rt_rows: usize = 0;
fn rtNote(landed: bool, rows: usize) void {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t);
    const now = @as(f64, @floatFromInt(t.sec)) + @as(f64, @floatFromInt(t.nsec)) * 1e-9;
    defer rt_last = now;
    if (rt_last == 0 or now - rt_last > 0.2) return;
    rt_ms[rt_n] = @floatCast((now - rt_last) * 1e3);
    rt_n += 1;
    rt_landed += @intFromBool(landed);
    rt_rows += rows;
    if (rt_n < rt_ms.len) return;
    var sorted = rt_ms;
    std.mem.sort(f32, &sorted, {}, std.sort.asc(f32));
    var sum: f64 = 0;
    for (sorted) |v| sum += v;
    std.log.warn("RT p10={d:.3} p25={d:.3} p50={d:.3} mean={d:.3} landed={d} rows={d}", .{ sorted[51], sorted[128], sorted[256], sum / 512.0, rt_landed, rt_rows });
    rt_n = 0;
    rt_landed = 0;
    rt_rows = 0;
}
''', 1)
a2 = '        g.draft.backend.precise = g.accept_share < precise_below;\n'
assert a2 in s
s = s.replace(a2, a2 + '        rtNote(accepted.matched > 0, ids.len);\n', 1)
open(p, 'w').write(s)
print('rt applied')
