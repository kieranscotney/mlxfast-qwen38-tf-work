#!/usr/bin/env python3
# After rt-apply.py: logs the first 8 round times after each prompt ("EARLYMS"), round 1's landing ("ROUND1") and the
# landings over the first 64 rounds ("EARLY").
p = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/generation.zig'
s = open(p).read()
assert 'var rt_ms' in s and 'early_rounds' not in s
s = s.replace("var rt_ms: [512]f32 = undefined;\n", """var rt_ms: [512]f32 = undefined;
var early_rounds: usize = 0;
var early_landed: usize = 0;
var early_last: f64 = 0;
var early_ms: [8]f32 = @splat(0);
fn nowS() f64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t);
    return @as(f64, @floatFromInt(t.sec)) + @as(f64, @floatFromInt(t.nsec)) * 1e-9;
}
""", 1)
a = '        rtNote(accepted.matched > 0, ids.len);\n'
s = s.replace(a, a + """        {
            const now = nowS();
            if (early_rounds < 8 and early_last != 0) early_ms[early_rounds] = @floatCast((now - early_last) * 1e3);
            early_last = now;
            if (early_rounds == 7) std.log.warn("EARLYMS {any}", .{early_ms});
        }
        early_rounds += 1;
        early_landed += @intFromBool(accepted.matched > 0);
        if (early_rounds == 64) std.log.warn("EARLY landed={d} of 64", .{early_landed});
        if (early_rounds == 1) std.log.warn("ROUND1 landed={d}", .{@intFromBool(accepted.matched > 0)});
""", 1)
a2 = '    pub fn prefill(g: *Generation, ids: []const u32, chunk: usize) !void {\n'
s = s.replace(a2, a2 + '        early_rounds = 0;\n        early_landed = 0;\n        defer early_last = nowS();\n', 1)
open(p, 'w').write(s)
print('early probe applied')
