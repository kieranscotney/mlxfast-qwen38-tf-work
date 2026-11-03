#!/usr/bin/env python3
# After tl-apply.py: host stamps on Metal's clock per round: h0 frame wait returned (Frame.waitLast), h1 selector done
# (Model.completeNext), h2 the verify's first chunk committed (segments.commitLane) -> "TLH h0 h1 h2".
seg = 'Vendor/TensorFold/zig/src/core/segments.zig'
s = open(seg).read()
s = s.replace('pub fn checkUnchecked() !void {', 'pub var tlh: [2]f64 = .{ 0, 0 };\npub fn checkUnchecked() !void {', 1)
old = '''            if (x.n_done.* == 0) try beforeCommit(x.fam);'''
assert old in s
s = s.replace(old, old + '''
            if (x.n_done.* == 0) std.log.warn("TLH {d:.6} {d:.6} {d:.6}", .{ tlh[0], tlh[1], mtl.clock.seconds() });''', 1)
open(seg, 'w').write(s)
fr = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/runtime_frame.zig'
s = open(fr).read()
old = '''    pub fn waitLast(f: *Frame) !void {
        if (!f.active or f.completed or !f.committed) return error.DraftFrameNotActive;'''
assert old in s
s = s.replace(old, old + '''
        defer @import("core").segments.tlh[0] = mtl.clock.seconds();''', 1)
open(fr, 'w').write(s)
md = 'Vendor/TensorFold/zig/src/families/qwen27/dflash/runtime_model.zig'
s = open(md).read()
old = '''        return selector.propose(m.allocator, &m.graph, lattice, pending, try std.math.add(u64, m.session.context.end, 1), options);
    }
    /// Ends the early frame without drafting from it'''
assert old in s
s = s.replace(old, '''        defer @import("core").segments.tlh[1] = mtl.clock.seconds();
        return selector.propose(m.allocator, &m.graph, lattice, pending, try std.math.add(u64, m.session.context.end, 1), options);
    }
    /// Ends the early frame without drafting from it''', 1)
open(md, 'w').write(s)
print('host stamps applied')
