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
            InvalidthreadsCount,
            MemoryNot512BytesAligned,
        } ||
            codec.PrimitiveCodec.Error ||
            byteUnit.Error;
        pub const CursorT = zcasp.codec.ArgCodec(Spec).CursorT;
        pub const SpecFieldEnum = zcasp.codec.ArgCodec(Spec).SpecFieldEnum;

        pub fn supports(
            comptime T: type,
            comptime tag: SpecFieldEnum,
        ) bool {
            return T == byteUnit.ByteUnit or tag == .threads;
        }

        pub fn parseByType(
            self: *@This(),
            comptime T: type,
            comptime tag: SpecFieldEnum,
            allocator: *const std.mem.Allocator,
            cursor: *CursorT,
        ) Error!T {
            switch (comptime tag) {
                .threads => {
                    const value = try zcasp.codec.PrimitiveCodec.parseInt(
                        T,
                        cursor,
                    );
                    if (value < 1) return Error.InvalidthreadsCount;
                    return value;
                },
                else => {
                    if (T == byteUnit.ByteUnit) {
                        const sizeStr = cursor.next() orelse return Error.MissingMemSize;
                        const bUnit = try zcasp.codec.byteUnit.parse(sizeStr);
                        const size = bUnit.size();
                        if (size == 0) return Error.ZeroSize;
                        if (size & 511 != 0) return Error.MemoryNot512BytesAligned;
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
    nonTemporal,

    pub fn unitT(comptime self: @This()) type {
        return switch (self) {
            .zig => u64,
            .assembly => u8,
            .nonTemporal => u8,
        };
    }
};

const Args = struct {
    mem: byteUnit.ByteUnit = undefined,
    cycles: usize = undefined,
    @"memset-type": MemsetType = .nonTemporal,
    threads: u16 = 1,

    pub const Short = .{
        .m = .mem,
        .c = .cycles,
        .mT = .@"memset-type",
        .t = .threads,
    };

    pub const Positionals = zcasp.positionals.EmptyPositionalsOf;

    pub const Codec = ArgsCodec(@This());

    pub const Help: zcasp.help.HelpData(@This()) = .{
        .usage = &.{"memtest <args>"},
        .description = "Runs memory test and reports gbp/s",
        .examples = &.{
            "memtest --mem 8gb --cycles 30",
            "memtest --mem 10gb --cycles 50",
            "memtest --memset-type zig --mem 10gb --cycles 50",
            "memtest --threads 2 --mem 10gb --cycles 50",
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
            .{ .field = .threads, .description = "Number of threads to run writes in parallel, they will be pinned to cpu and you need mem x threads avaiable memory." },
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
    scrapAlloc = std.heap.page_allocator;
    const stderr = std.fs.File.stderr();

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

    const elapsedArray = try scrapAlloc.alloc(u64, argsRes.options.threads);
    const threads = try scrapAlloc.alloc(std.Thread, argsRes.options.threads);
    for (0..argsRes.options.threads) |i| {
        switch (argsRes.options.@"memset-type") {
            inline else => |memsetT| {
                elapsedArray[i] = 0;
                threads[i] = std.Thread.spawn(.{}, runMemTest, .{
                    memsetT,
                    MemsetType.unitT(memsetT),
                    &argsRes,
                    &elapsedArray[i],
                    i,
                }) catch |e| {
                    try stderrW.print("Memtest failed with: {s}\n", .{@errorName(e)});
                    try stderrW.flush();
                    return 1;
                };
            },
        }
    }

    for (threads) |*thread| thread.join();

    const targetBytes = argsRes.options.mem.size();
    const gbsProcessed: f128 = @as(f128, @floatFromInt(targetBytes)) * @as(f128, @floatFromInt(argsRes.options.cycles)) / CByteUnit.gb;
    for (elapsedArray, 1..) |elapsed, id| {
        const elapsedF: f128 = @floatFromInt(elapsed);
        const elapsedInSec = elapsedF / regent.units.NanoUnit.s;
        try stderrW.print("thread {d}: {d:.2} gb/s\n", .{
            id,
            gbsProcessed / elapsedInSec,
        });
    }
    try stderrW.flush();

    return 0;
}

extern fn asm_memset(target: ?*anyopaque, c: c_int, n: usize) ?*anyopaque;

pub noinline fn nonTemporalMemsetAVX2(dst: []u8, targetBuff: []const u8, elapsed: *u64, timer: *std.time.Timer) void {
    var ptr = dst.ptr;
    const end = dst.ptr + dst.len;

    timer.reset();
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
    elapsed.* += timer.read();
}

pub noinline fn nonTemporalMemsetAVX512(dst: []u8, targetBuff: []const u8, elapsed: *u64, timer: *std.time.Timer) void {
    var ptr = dst.ptr;
    const end = dst.ptr + dst.len;

    timer.reset();
    asm volatile (
        \\ vmovdqu64 (%[src]), %%zmm0
        \\ vmovaps %%zmm0, %%zmm1
        \\ vmovaps %%zmm0, %%zmm2
        \\ vmovaps %%zmm0, %%zmm3
        \\ vmovaps %%zmm0, %%zmm4
        \\ vmovaps %%zmm0, %%zmm5
        \\ vmovaps %%zmm0, %%zmm6
        \\ vmovaps %%zmm0, %%zmm7
        \\
        \\ .align 64
        \\ 1:
        \\   vmovntdq %%zmm0, 0(%[ptr])
        \\   vmovntdq %%zmm1, 64(%[ptr])
        \\   vmovntdq %%zmm2, 128(%[ptr])
        \\   vmovntdq %%zmm3, 192(%[ptr])
        \\   vmovntdq %%zmm4, 256(%[ptr])
        \\   vmovntdq %%zmm5, 320(%[ptr])
        \\   vmovntdq %%zmm6, 384(%[ptr])
        \\   vmovntdq %%zmm7, 448(%[ptr])
        \\
        \\   leaq 512(%[ptr]), %[ptr]
        \\   cmp %[end], %[ptr]
        \\   jne 1b
        \\
        \\ sfence
        : [ptr] "+&r" (ptr),
        : [end] "r" (end),
          [src] "r" (targetBuff.ptr),
        : .{
          .zmm0 = true,
          .zmm1 = true,
          .zmm2 = true,
          .zmm3 = true,
          .zmm4 = true,
          .zmm5 = true,
          .zmm6 = true,
          .zmm7 = true,
          .memory = true,
          .cc = true,
        });

    elapsed.* += timer.read();
}

fn hasAvx2() bool {
    // 1. Check max CPUID leaf
    var res = cpuid(0, 0);
    if (res.eax < 7) return false;

    // 2. Check Leaf 1 for OSXSAVE (bit 27 of ECX)
    res = cpuid(1, 0);
    if ((res.ecx & (@as(u32, 1) << 27)) == 0) return false;

    // 3. Check XCR0 for AVX state (Bit 2: YMM)
    // SSE state (Bit 1) must also be set for AVX to work.
    const xcr0 = xgetbv(0);
    if ((xcr0 & 0x06) != 0x06) return false;

    // 4. Check Leaf 7, Sub-leaf 0 for AVX2 (Bit 5 of EBX)
    res = cpuid(7, 0);
    return (res.ebx & (@as(u32, 1) << 5)) != 0;
}

fn hasAvx512Dq() bool {
    // 1. Check max input value for CPUID to ensure Leaf 7 exists
    const leaf0 = cpuid(0, 0);
    if (leaf0.eax < 7) return false;

    // 2. Check for OSXSAVE support (Bit 27 of ECX in Leaf 1)
    const leaf1 = cpuid(1, 0);
    if ((leaf1.ecx & (@as(u32, 1) << 27)) == 0) return false;

    // 3. Check XCR0 via XGETBV (OS support for ZMM state)
    // We must check if the OS handles:
    // Bit 5 (Opmask), Bit 6 (ZMM_Hi256), Bit 7 (Hi16_ZMM)
    const xcr0 = xgetbv(0);
    const zmm_os_bits: u64 = (1 << 5) | (1 << 6) | (1 << 7);
    if ((xcr0 & zmm_os_bits) != zmm_os_bits) return false;

    // 4. Check Leaf 7 for AVX-512F (bit 16) and AVX-512DQ (bit 17)
    const leaf7 = cpuid(7, 0);
    const f_mask = @as(u32, 1) << 16;
    const dq_mask = @as(u32, 1) << 17;

    return (leaf7.ebx & f_mask != 0) and (leaf7.ebx & dq_mask != 0);
}

const CpuIdRes = struct {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
};

fn cpuid(leaf: u32, sub_leaf: u32) CpuIdRes {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [eax] "={eax}" (eax),
          [ebx] "={ebx}" (ebx),
          [ecx] "={ecx}" (ecx),
          [edx] "={edx}" (edx),
        : [leaf] "{eax}" (leaf),
          [sub_leaf] "{ecx}" (sub_leaf),
    );
    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

fn xgetbv(index: u32) u64 {
    var eax: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("xgetbv"
        : [eax] "={eax}" (eax),
          [edx] "={edx}" (edx),
        : [index] "{ecx}" (index),
    );
    return (@as(u64, edx) << 32) | eax;
}

pub fn runMemTest(
    comptime memsetT: MemsetType,
    comptime T: type,
    argsRes: *const ArgsResponse,
    elapsed: *u64,
    id: usize,
) !void {
    var cpuset: c.cpu_set_t = undefined;
    cpuset.__bits = @splat(0);

    const i = id % try std.Thread.getCpuCount();
    const bHi = i / (@sizeOf(usize) * 8);
    const bLo = i % (@sizeOf(usize) * 8);
    cpuset.__bits[bHi] |= (@as(usize, 1) << @intCast(bLo));

    _ = c.sched_setaffinity(
        @intCast(std.Thread.getCurrentId()),
        @sizeOf(c.cpu_set_t),
        @intFromPtr(&cpuset),
    );

    const seed: u64 = @intCast(std.time.nanoTimestamp() & std.math.maxInt(u64));
    var generator = std.Random.DefaultPrng.init(seed);
    var random = generator.random();

    const alloc = std.heap.c_allocator;
    const allMem = try alloc.alignedAlloc(
        T,
        std.mem.Alignment.@"64",
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
            .nonTemporal => {
                const avx2 = hasAvx2();
                const avx512 = hasAvx512Dq();

                if (!avx2 and !avx512) return error{NonTempAVXMemsetUnavailable}.NonTempAVXMemsetUnavailable;

                const targetBuff: []u8 = try alloc.alignedAlloc(u8, std.mem.Alignment.@"64", if (avx512) 64 else 32);
                defer alloc.free(targetBuff);
                @memset(targetBuff, target);

                if (avx512)
                    nonTemporalMemsetAVX512(
                        allMem,
                        targetBuff,
                        elapsed,
                        &timer,
                    )
                else
                    nonTemporalMemsetAVX2(
                        allMem,
                        targetBuff,
                        elapsed,
                        &timer,
                    );
                asm volatile ("vzeroupper");
            },
        }
    }
}
