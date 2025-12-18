const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const module = b.addModule("memtest", .{
        .root_source_file = b.path("memtest.zig"),
        .target = target,
        .optimize = optimize,
        .omit_frame_pointer = optimize == .ReleaseFast,
        .link_libc = true,
    });

    module.addImport("regent", b.dependency("regent", .{
        .target = target,
        .optimize = optimize,
    }).module("regent"));
    module.addImport("zcasp", b.dependency("zcasp", .{
        .target = target,
        .optimize = optimize,
    }).module("zcasp"));

    const exe = b.addExecutable(.{
        .name = "memtest",
        .root_module = module,
        .use_llvm = true,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
