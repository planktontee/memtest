const std = @import("std");
const zcasp = @import("zcasp");
const regent = @import("regent");
const codec = zcasp.codec;
const byteUnit = codec.byteUnit;
const CByteUnit = regent.units.ByteUnit;

pub fn ArgsCodec(Spec: type) type {
    return struct {
        defaultCodec: zcasp.codec.ArgCodec(Spec) = .{},

        pub const Error = error{
            MissingMemSize,
            ZeroSize,
        } ||
            codec.PrimitiveCodec.Error ||
            byteUnit.Error;
        pub const CursorT = zcasp.codec.ArgCodec(Spec).CursorT;
        pub const SpecFieldEnum = std.meta.FieldEnum(Spec);

        pub fn supports(
            comptime T: type,
            comptime _: SpecFieldEnum,
        ) bool {
            return T == byteUnit.ByteUnit;
        }

        pub fn parseByType(
            self: *@This(),
            comptime T: type,
            comptime tag: SpecFieldEnum,
            allocator: *const std.mem.Allocator,
            cursor: *CursorT,
        ) Error!T {
            if (T == byteUnit.ByteUnit) {
                const sizeStr = cursor.next() orelse return Error.MissingMemSize;
                const bUnit = try zcasp.codec.byteUnit.parse(sizeStr);
                if (bUnit.size() == 0) return Error.ZeroSize;
                return bUnit;
            } else {
                return try @TypeOf(self.defaultCodec).parseByType(self, T, tag, allocator, cursor);
            }
        }
    };
}

const Args = struct {
    mem: byteUnit.ByteUnit = undefined,
    cycles: usize = undefined,

    pub const Short = .{
        .rn = .random,
        .m = .mem,
        .c = .cycles,
    };

    pub const Positionals = zcasp.positionals.EmptyPositionalsOf;

    pub const Codec = ArgsCodec(@This());

    pub const Help: zcasp.help.HelpData(@This()) = .{
        .usage = &.{"memtest <args>"},
        .description = "Runs memory test and reports gbp/s",
        .examples = &.{
            "memtest --mem 8mb --cycles 3",
            "memtest --mem 10gb --cycles 10",
        },
        .optionsDescription = &.{
            .{ .field = .mem, .description = "Memory amount to use, example: 10gb" },
            .{ .field = .cycles, .description = "How many times to cycle through the memory with memset" },
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
    var scrapAllocBuff: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&scrapAllocBuff);
    scrapAlloc = fba.allocator();
    const stderr = std.fs.File.stderr();

    var stderrBuff: [CByteUnit.kb * 512]u8 = undefined;
    var stderrIO = stderr.writer(&stderrBuff);
    stderrW = &stderrIO.interface;

    var argsRes: ArgsResponse = .init(scrapAlloc);
    defer argsRes.deinit();

    if (argsRes.parseArgs()) |parseError| {
        try stderrW.print("Last opt <{?s}>, Last token <{?s}>. ", .{ parseError.lastOpt, parseError.lastToken });
        try stderrW.writeAll(parseError.message orelse unreachable);
        try stderrW.flush();
        return 1;
    }

    runMemTest(&argsRes) catch |e| {
        try stderrW.print("Memtest failed with: {s}\n", .{@errorName(e)});
        try stderrW.flush();
        return 1;
    };

    return 0;
}

pub fn runMemTest(argsRes: *const ArgsResponse) !void {
    const seed: u64 = @intCast(std.time.nanoTimestamp() & std.math.maxInt(u64));
    var generator = std.Random.DefaultPrng.init(seed);
    var random = generator.random();

    const alloc = std.heap.page_allocator;
    const targetBytes = argsRes.options.mem.size();
    const allMem = try alloc.alloc(u8, argsRes.options.mem.size());

    var timer = try std.time.Timer.start();
    var elapsed: u64 = 0;
    for (0..argsRes.options.cycles) |_| {
        const target = random.int(u8);
        timer.reset();
        @memset(allMem, target);
        elapsed += timer.read();
    }
    const elapsedF: f128 = @floatFromInt(elapsed);
    const gbsProcessed: f128 = @as(f128, @floatFromInt(targetBytes)) / CByteUnit.gb * @as(f128, @floatFromInt(argsRes.options.cycles));
    const elapsedInSec = elapsedF / regent.units.NanoUnit.s;
    try stderrW.print("{d:.2} gb/s\n", .{
        gbsProcessed / elapsedInSec,
    });
    try stderrW.flush();
}
