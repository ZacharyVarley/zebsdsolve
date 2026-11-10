//! The simulator's host-side parts, for the unit tests (`zig build test`): everything of
//! src/sim/ that needs no GPU.
test {
    _ = @import("sim/group.zig");
    _ = @import("sim/cif.zig");
    _ = @import("sim/crystal.zig");
    _ = @import("sim/scatter.zig");
    _ = @import("sim/potential.zig");
    _ = @import("sim/energy.zig");
    _ = @import("sim/kgrid.zig");
    _ = @import("sim/cellframe.zig");
}
