const std = @import("std");
const zcasp = @import("zcasp");
const regent = @import("regent");
const codec = zcasp.codec;
const byteUnit = codec.byteUnit;
const CByteUnit = regent.units.ByteUnit;
const c = @cImport({
    @cDefine("__USE_GNU", "");
    @cDefine("_GNU_SOURCE", "");
    @cInclude("pthread.h");
    @cInclude("sched.h");
});

pub fn ArgsCodec(Spec: type) type {
    return struct {
        defaultCodec: zcasp.codec.ArgCodec(Spec) = .{},

        pub const Error = error{
            MissingMemSize,
            ZeroSize,
            InvalidChannelsCount,
        } ||
            codec.PrimitiveCodec.Error ||
            byteUnit.Error;
        pub const CursorT = zcasp.codec.ArgCodec(Spec).CursorT;
        pub const SpecFieldEnum = zcasp.codec.ArgCodec(Spec).SpecFieldEnum;

        pub fn supports(
            comptime T: type,
            comptime tag: SpecFieldEnum,
        ) bool {
            return T == byteUnit.ByteUnit or tag == .channels;
        }

        pub fn parseByType(
            self: *@This(),
            comptime T: type,
            comptime tag: SpecFieldEnum,
            allocator: *const std.mem.Allocator,
            cursor: *CursorT,
        ) Error!T {
            switch (comptime tag) {
                .channels => {
                    const value = try zcasp.codec.PrimitiveCodec.parseInt(
                        T,
                        cursor,
                    );
                    if (value < 1) return Error.InvalidChannelsCount;
                    return value;
                },
                else => {
                    if (T == byteUnit.ByteUnit) {
                        const sizeStr = cursor.next() orelse return Error.MissingMemSize;
                        const bUnit = try zcasp.codec.byteUnit.parse(sizeStr);
                        if (bUnit.size() == 0) return Error.ZeroSize;
                        return bUnit;
                    } else {
                        return try @TypeOf(self.defaultCodec).parseByType(self, T, tag, allocator, cursor);
                    }
                },
            }
        }
    };
}

pub const MemsetType = enum {
    zig,
    assembly,
    manual,
};

const Args = struct {
    mem: byteUnit.ByteUnit = undefined,
    cycles: usize = undefined,
    @"memset-type": MemsetType = .manual,
    channels: u3 = 1,

    pub const Short = .{
        .m = .mem,
        .c = .cycles,
        .mT = .@"memset-type",
        .mC = .channels,
    };

    pub const Positionals = zcasp.positionals.EmptyPositionalsOf;

    pub const Codec = ArgsCodec(@This());

    pub const Help: zcasp.help.HelpData(@This()) = .{
        .usage = &.{"memtest <args>"},
        .description = "Runs memory test and reports gbp/s",
        .examples = &.{
            "memtest --mem 8mb --cycles 3",
            "memtest --mem 10gb --cycles 10",
            "memtest --memset-type zig --mem 10gb --cycles 10",
        },
        .optionsDescription = &.{
            .{ .field = .mem, .description = "Memory amount to use, example: 10gb" },
            .{ .field = .cycles, .description = "How many times to cycle through the memory with memset" },
            .{
                .field = .@"memset-type",
                .defaultHint = false,
                .typeHint = false,
                .description = "Pick Memset implementation to run test with. Options: " ++ zcasp.help.enumValueHint(MemsetType),
            },
            .{ .field = .channels, .description = "Number of channels your mobo supports to be stressed" },
        },
    };

    pub const GroupMatch: zcasp.validate.GroupMatchConfig(@This()) = .{
        .required = &.{ .mem, .cycles },
    };
};

const ArgsResponse = zcasp.spec.SpecResponseWithConfig(Args, zcasp.help.HelpConf{
    .backwardsBranchesQuote = 1000000,
    .simpleTypes = true,
}, true);

pub var scrapAlloc: std.mem.Allocator = undefined;
pub var stderrW: *std.Io.Writer = undefined;

pub fn main() !u8 {
    // var scrapAllocBuff: [4096]u8 = undefied;
    // var fba = std.heap.FixedBufferAllocator.init(&scrapAllocBuff);
    // scrapAlloc = fba.allocator();
    scrapAlloc = std.heap.page_allocator;
    const stderr = std.fs.File.stderr();

    // var stderrBuff: [CByteUnit.kb * 512]u8 = undefined;
    const stderrBuff = try scrapAlloc.alloc(u8, CByteUnit.kb * 512);
    var stderrIO = stderr.writer(stderrBuff);
    stderrW = &stderrIO.interface;

    var argsRes: ArgsResponse = .init(scrapAlloc);
    defer argsRes.deinit();

    if (argsRes.parseArgs()) |parseError| {
        try stderrW.print("Last opt <{?s}>, Last token <{?s}>. ", .{ parseError.lastOpt, parseError.lastToken });
        try stderrW.writeAll(parseError.message orelse unreachable);
        try stderrW.flush();
        return 1;
    }

    const elapsedArray = try scrapAlloc.alloc(u64, argsRes.options.channels);
    const threads = try scrapAlloc.alloc(std.Thread, argsRes.options.channels);
    for (0..argsRes.options.channels) |i| {
        switch (argsRes.options.@"memset-type") {
            inline else => |memsetT| {
                elapsedArray[i] = 0;
                threads[i] = std.Thread.spawn(.{}, runMemTest, .{
                    memsetT,
                    switch (memsetT) {
                        .zig => u64,
                        .assembly => u8,
                        .manual => u8,
                    },
                    &argsRes,
                    &elapsedArray[i],
                    i + 1,
                }) catch |e| {
                    try stderrW.print("Memtest failed with: {s}\n", .{@errorName(e)});
                    try stderrW.flush();
                    return 1;
                };
            },
        }
    }

    for (threads) |*thread| thread.join();

    var elapsed: u64 = 0;
    for (elapsedArray) |elapsedI| elapsed += elapsedI;

    const targetBytes = argsRes.options.mem.size() * argsRes.options.channels;
    const elapsedF: f128 = @floatFromInt(elapsed);
    const gbsProcessed: f128 = @as(f128, @floatFromInt(targetBytes)) * @as(f128, @floatFromInt(argsRes.options.cycles)) / CByteUnit.gb;
    const elapsedInSec = elapsedF / regent.units.NanoUnit.s;
    try stderrW.print("{d:.2} gb/s\n", .{
        gbsProcessed / elapsedInSec,
    });
    try stderrW.flush();

    return 0;
}

extern fn asm_memset(target: ?*anyopaque, c: c_int, n: usize) ?*anyopaque;

pub noinline fn manualmemset(dst: []u8, targetBuff: []const u8) void {
    var ptr = dst.ptr;
    const end = dst.ptr + dst.len;

    asm volatile (
        \\ vmovdqu (%[src]), %%ymm0
        \\ vmovaps %%ymm0, %%ymm1
        \\ vmovaps %%ymm0, %%ymm2
        \\ vmovaps %%ymm0, %%ymm3
        \\ vmovaps %%ymm0, %%ymm4
        \\ vmovaps %%ymm0, %%ymm5
        \\ vmovaps %%ymm0, %%ymm6
        \\ vmovaps %%ymm0, %%ymm7
        \\
        \\ .align 16
        \\ 1:
        \\   vmovntdq %%ymm0, 0(%[ptr])
        \\   vmovntdq %%ymm1, 32(%[ptr])
        \\   vmovntdq %%ymm2, 64(%[ptr])
        \\   vmovntdq %%ymm3, 96(%[ptr])
        \\   vmovntdq %%ymm4, 128(%[ptr])
        \\   vmovntdq %%ymm5, 160(%[ptr])
        \\   vmovntdq %%ymm6, 192(%[ptr])
        \\   vmovntdq %%ymm7, 224(%[ptr])
        \\
        \\   lea 256(%[ptr]), %[ptr]
        \\   cmp %[end], %[ptr]
        \\   jne 1b
        \\
        \\ sfence
        : [ptr] "+&r" (ptr),
        : [end] "r" (end),
          [src] "r" (targetBuff.ptr),
        : .{
          .ymm0 = true,
          .ymm1 = true,
          .ymm2 = true,
          .ymm3 = true,
          .ymm4 = true,
          .ymm5 = true,
          .ymm6 = true,
          .ymm7 = true,
          .memory = true,
          .cc = true,
        });
}

pub fn runMemTest(
    comptime memsetT: MemsetType,
    comptime T: type,
    argsRes: *const ArgsResponse,
    elapsed: *u64,
    i: usize,
) !void {
    var cpuset: c.cpu_set_t = undefined;
    cpuset.__bits = @splat(0);
    const w = i / (@sizeOf(usize) * 8);
    const b = i % (@sizeOf(usize) * 8);
    cpuset.__bits[w] |= (@as(usize, 1) << @intCast(b));

    _ = c.sched_setaffinity(@intCast(std.Thread.getCurrentId()), @sizeOf(c.cpu_set_t), @intFromPtr(&cpuset));

    const seed: u64 = @intCast(std.time.nanoTimestamp() & std.math.maxInt(u64));
    var generator = std.Random.DefaultPrng.init(seed);
    var random = generator.random();

    const alloc = std.heap.c_allocator;
    const allMem = try alloc.alignedAlloc(
        T,
        std.mem.Alignment.@"32",
        argsRes.options.mem.size() / @sizeOf(T),
    );

    var timer = try std.time.Timer.start();
    for (0..argsRes.options.cycles) |_| {
        const target = random.int(T);
        switch (comptime memsetT) {
            .zig => {
                timer.reset();
                @memset(allMem, target);
                elapsed.* += timer.read();
            },
            .assembly => {
                timer.reset();
                _ = asm_memset(@ptrCast(allMem), @intCast(target), allMem.len);
                elapsed.* += timer.read();
            },
            .manual => {
                const targetBuff: []u8 = try alloc.alignedAlloc(u8, std.mem.Alignment.@"32", 32);
                defer alloc.free(targetBuff);
                @memset(targetBuff, target);

                timer.reset();
                manualmemset(allMem, targetBuff);
                elapsed.* += timer.read();
                asm volatile ("vzeroupper");
            },
        }
    }
}
