#!/usr/bin/env python3
# Probe + cold drafter context: times the prompt catch-up (Model.flush, "FLUSH" lines), counts landings over the first
# 64 rounds after each prompt ("EARLY" lines), and limits the catch-up to the last COLD rows (argv[1]).
import sys
cold = int(sys.argv[1])
p = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/runtime_model.zig'
s = open(p).read()
old = "        const begin = m.committed_end -| (@as(u64, m.graph.config.window) - 1);\n"
assert old in s
s = s.replace(old, "        const begin = m.committed_end -| @min(@as(u64, m.graph.config.window) - 1, %d);\n" % cold, 1)
old = "        try m.flush();\n        const first = try m.session.firstDraw();\n"
assert old in s
s = s.replace(old, '''        {
            var t0: std.c.timespec = undefined;
            _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t0);
            const behind0 = m.committed_end - m.session.context.end;
            try m.flush();
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t1);
            const ms = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) * 1e-6;
            if (behind0 > 128) std.log.warn("FLUSH behind={d} ms={d:.3}", .{ behind0, ms });
        }
        const first = try m.session.firstDraw();
''', 1)
open(p, 'w').write(s)
g = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/generation.zig'
s = open(g).read()
a = '        rtNote(accepted.matched > 0, ids.len);\n'
assert a in s
s = s.replace(a, a + '''        early_rounds += 1;
        early_landed += @intFromBool(accepted.matched > 0);
        if (early_rounds == 64) std.log.warn("EARLY landed={d} of 64", .{early_landed});
''', 1)
s = s.replace('var rt_ms: [512]f32 = undefined;\n', 'var rt_ms: [512]f32 = undefined;\nvar early_rounds: usize = 0;\nvar early_landed: usize = 0;\n', 1)
a2 = '    pub fn prefill(g: *Generation, ids: []const u32, chunk: usize) !void {\n'
assert a2 in s
s = s.replace(a2, a2 + '        early_rounds = 0;\n        early_landed = 0;\n', 1)
open(g, 'w').write(s)
print('cold', cold)
