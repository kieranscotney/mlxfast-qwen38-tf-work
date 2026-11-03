#!/usr/bin/env python3
# After rt-apply.py: one "TR" line per round: the target token's rank in the selector's order of the drafter's top-16
# (16 = not there), the siblings offered, the rows verified, and the 16 in-top-16 drafter probabilities and selector
# log-scores in selector order. For offline replay of sibling policies.
sel = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/selector.zig'
s = open(sel).read()
s = s.replace('pub const mass_stop: f64', '''pub var tr_tokens: [16]u32 = @splat(0xffffffff);
pub var tr_q: [16]f64 = @splat(0);
pub var tr_s: [16]f64 = @splat(0);
pub var tr_offered: u32 = 0;
pub const mass_stop: f64''', 1)
old = '''    const children = @min(options.children, lattice.topk);
    var offered: u32 = 0;'''
assert old in s
s = s.replace(old, '''    if (parent == -1) for (order[0..lattice.topk], 0..) |i, j| {
        tr_tokens[j] = lattice.candidates[row + i];
        tr_q[j] = if (options.floor > 0) drafter[i] else 0;
        tr_s[j] = scores[i];
    };
    const children = @min(options.children, lattice.topk);
    var offered: u32 = 0;''', 1)
old = '''            if (mass >= mass_stop) break;'''
assert old in s
s = s.replace(old, '''            if (mass >= mass_stop) {
                if (parent == -1) tr_offered = offered;
                break;
            }''', 1)
old = '''        try frontier.append(lattice.gpa, .{ .score = cumulative + scores[i], .parent = parent, .token = lattice.candidates[row + i], .depth = depth });
        offered += 1;'''
assert old in s
s = s.replace(old, old + '\n        if (parent == -1) tr_offered = offered;', 1)
open(sel, 'w').write(s)
g = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/generation.zig'
s = open(g).read()
a = '        rtNote(accepted.matched > 0, ids.len);\n'
assert a in s
s = s.replace(a, a + '''        {
            const sl = @import("selector.zig");
            var rank: usize = 16;
            for (sl.tr_tokens, 0..) |id, j| if (id == accepted.tokens[0]) {
                rank = j;
                break;
            };
            var buf: [600]u8 = undefined;
            var w = std.Io.Writer.fixed(&buf);
            for (0..16) |j| w.print(" {d:.4}/{d:.2}", .{ sl.tr_q[j], sl.tr_s[j] }) catch {};
            std.log.warn("TR r={d} o={d} rows={d} m={d}{s}", .{ rank, sl.tr_offered, ids.len, @intFromBool(accepted.matched > 0), w.buffered() });
        }
''', 1)
open(g, 'w').write(s)
print('trace applied')
