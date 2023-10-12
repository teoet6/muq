const std = @import("std");
const mnist = @import("mnist.zig");

const fnn = f32;

fn rows(comptime T: type) comptime_int {
    return @typeInfo(T).Array.len;
}

fn cols(comptime T: type) comptime_int {
    return @typeInfo(@typeInfo(T).Array.child).Array.len;
}

fn elem(comptime T: type) type {
    return @typeInfo(@typeInfo(T).Array.child).Array.child;
}

fn MatrixProduct(comptime A: type, comptime B: type) type {
    if (elem(A) != elem(B)) {
        @compileError(std.fmt.comptimePrint("Expected elem(a) == elem(b), got {} != {}", .{ elem(A), elem(B) }));
    }

    if (cols(A) != rows(B)) {
        @compileError(std.fmt.comptimePrint("Expected cols(a) == rows(b), got {} != {}", .{ cols(A), rows(B) }));
    }

    return [rows(A)][cols(B)]elem(A);
}

fn matrix_multiply(a: anytype, b: anytype, c: *MatrixProduct(@TypeOf(a.*), @TypeOf(b.*))) void {
    for (c) |*it| {
        for (it) |*jt| {
            jt.* = 0;
        }
    }

    for (0..c.len) |row| {
        for (0..b.len) |idx| {
            const vec_len = 8;
            const Vector = @Vector(vec_len, elem(@TypeOf(c.*)));

            const a_vec = @as(Vector, @splat(a[row][idx]));

            var col = @as(usize, 0);

            while (col + vec_len <= c[row].len) : (col += vec_len) {
                const b_vec = @as(Vector, b[idx][col..][0..vec_len].*);
                c[row][col..][0..vec_len].* += a_vec * b_vec;
            }

            while (col < c[row].len) : (col += 1) {
                c[row][col] += a[row][idx] * b[idx][col];
            }
        }
    }
}

fn MatrixVectorProduct(comptime A: type, comptime B: type) type {
    const elem_b = @typeInfo(B).Array.child;

    if (elem(A) != elem_b) {
        @compileError(std.fmt.comptimePrint("Expected elem(a) == elem(b), got {} != {}", .{ elem(A), elem_b }));
    }

    if (cols(A) != rows(B)) {
        @compileError(std.fmt.comptimePrint("Expected cols(a) == rows(b), got {} != {}", .{ cols(A), rows(B) }));
    }

    return [rows(A)]elem(A);
}

fn matrix_vector_multiply(a: anytype, b: anytype, c: *MatrixVectorProduct(@TypeOf(a.*), @TypeOf(b.*))) void {
    for (c, a) |*c_it, a_row| {
        c_it.* = 0;
        for (a_row, b) |a_it, b_it| c_it.* += a_it * b_it;
    }
}

fn FullyConnected(comptime input_size_: comptime_int, comptime output_size_: comptime_int) type {
    return struct {
        const input_size = input_size_;
        const output_size = output_size_;
        const param_size = (input_size + 1) * output_size; // +1 because of bias

        fn eval(_: std.mem.Allocator, input: *const [input_size]fnn, param: *const [param_size]fnn, output: *[output_size]fnn) !void {
            for (output, 0..) |*it, i| {
                const param_it = param[i * (input_size + 1) ..][0 .. input_size + 1];
                it.* = 0;
                it.* += param_it[0];
                for (input, param_it[1..]) |x, w| it.* += x * w;
            }
        }

        fn param_derive(_: std.mem.Allocator, input: *const [input_size]fnn, _: *const [param_size]fnn, derivative: *[param_size][output_size]fnn) !void {
            @memset(derivative, [_]fnn{0} ** output_size);

            // XXX maybe bad cache
            for (0..output_size) |i| {
                const derivative_it = derivative[i * (input_size + 1) ..][0 .. input_size + 1];
                derivative_it[0][i] = 1;
                for (input, 1..) |x, j| derivative_it[j][i] = x;
            }
        }

        fn input_derive(_: std.mem.Allocator, _: *const [input_size]fnn, param: *const [param_size]fnn, derivative: *[input_size][output_size]fnn) !void {
            // XXX maybe bad cache
            for (derivative, 0..) |*derivative_row, row_idx| {
                for (derivative_row, 0..) |*it, col_idx| {
                    it.* = param[1 + row_idx + (input_size + 1) * col_idx];
                }
            }
        }
    };
}

fn Tanh(comptime io_size: comptime_int) type {
    return struct {
        const input_size = io_size;
        const output_size = io_size;
        const param_size = 0;

        fn eval(_: std.mem.Allocator, input: *const [input_size]fnn, _: *const [param_size]fnn, output: *[output_size]fnn) !void {
            for (output, input) |*out, in| out.* = std.math.tanh(in);
        }

        fn param_derive(_: std.mem.Allocator, _: *const [input_size]fnn, _: *const [param_size]fnn, _: *[param_size][output_size]fnn) !void {}

        fn input_derive(_: std.mem.Allocator, input: *const [input_size]fnn, _: *const [param_size]fnn, derivative: *[input_size][output_size]fnn) !void {
            @memset(derivative, [_]fnn{0} ** output_size);

            for (input, 0..) |x, i| {
                const sech = 1 / std.math.cosh(x);
                derivative[i][i] = sech * sech;
            }
        }
    };
}

fn Sigmoid(comptime io_size: comptime_int) type {
    return struct {
        const input_size = io_size;
        const output_size = io_size;
        const param_size = 0;

        fn eval(_: std.mem.Allocator, input: *const [input_size]fnn, _: *const [param_size]fnn, output: *[output_size]fnn) !void {
            for (output, input) |*out, in| out.* = 0.5 * (std.math.tanh(0.5 * in) + 1);
        }

        fn param_derive(_: std.mem.Allocator, _: *const [input_size]fnn, _: *const [param_size]fnn, _: *[param_size][output_size]fnn) !void {}

        fn input_derive(_: std.mem.Allocator, input: *const [input_size]fnn, _: *const [param_size]fnn, derivative: *[input_size][output_size]fnn) !void {
            @memset(derivative, [_]fnn{0} ** output_size);

            for (input, 0..) |x, i| {
                const sech = 1 / std.math.cosh(0.5 * x);
                derivative[i][i] = 0.25 * sech * sech;
            }
        }
    };
}

fn Softmax(comptime io_size: comptime_int) type {
    return struct {
        const input_size = io_size;
        const output_size = io_size;
        const param_size = 0;

        fn eval(allocator: std.mem.Allocator, input: *const [input_size]fnn, _: *const [param_size]fnn, output: *[output_size]fnn) !void {
            var max = input[0];
            for (input[1..]) |it| max = @max(max, it);

            const exp = try allocator.create([input_size]fnn);
            defer allocator.destroy(exp);
            for (exp, input) |*e, in| e.* = @exp(in - max);

            var sum: fnn = 0;
            for (exp) |it| sum += it;

            for (output, exp) |*out, exp_it| out.* = exp_it / sum;
        }

        fn param_derive(_: std.mem.Allocator, _: *const [input_size]fnn, _: *const [param_size]fnn, _: *[param_size][output_size]fnn) !void {}

        fn input_derive(allocator: std.mem.Allocator, input: *const [input_size]fnn, _: *const [param_size]fnn, derivative: *[input_size][output_size]fnn) !void {
            var max = input[0];
            for (input[1..]) |it| max = @max(max, it);

            const exp = try allocator.create([input_size]fnn);
            defer allocator.destroy(exp);
            for (exp, input) |*e, in| e.* = @exp(in - max);

            var sum: fnn = 0;
            for (exp) |it| sum += it;

            for (derivative, 0..) |*it, in_idx| {
                for (it, 0..) |*jt, out_idx| {
                    if (in_idx == out_idx) {
                        jt.* = (sum * exp[out_idx] - exp[in_idx] * exp[out_idx]) / (sum * sum);
                    } else {
                        jt.* = -exp[in_idx] * exp[out_idx] / (sum * sum);
                    }
                }
            }
        }
    };
}

fn Chain(comptime models: anytype) type {
    // XXX check models[i-1].output_size == models[i].input_size and throw a nice error message
    return struct {
        const input_size = models[0].input_size;
        const output_size = models[models.len - 1].output_size;
        const param_size = blk: {
            var sum = 0;
            inline for (models) |Model| sum += Model.param_size;
            break :blk sum;
        };

        fn eval(allocator: std.mem.Allocator, input: *const [input_size]fnn, param: *const [param_size]fnn, output: *[output_size]fnn) !void {
            comptime var model_param_idx = 0;

            const max_io_size = comptime blk: {
                var max = models[0].input_size;
                inline for (models) |Model| max = @max(max, Model.output_size);
                break :blk max;
            };

            const io_buf = try allocator.create([2][max_io_size]fnn);
            defer allocator.destroy(io_buf);
            comptime var input_buffer_idx = 0;

            io_buf[input_buffer_idx][0..input.len].* = input.*;

            inline for (models) |Model| {
                const model_input = io_buf[input_buffer_idx][0..Model.input_size];
                const model_output = io_buf[input_buffer_idx ^ 1][0..Model.output_size];

                const model_param = param[model_param_idx..][0..Model.param_size];
                model_param_idx += Model.param_size;

                try Model.eval(allocator, model_input, model_param, model_output);

                input_buffer_idx ^= 1;
            }

            output.* = io_buf[input_buffer_idx][0..output_size].*;
        }

        fn param_derive(allocator: std.mem.Allocator, input: *const [input_size]fnn, param: *const [param_size]fnn, derivative: *[param_size][output_size]fnn) !void {
            // XXX currently we expect that we get an arena allocator.
            // This might not be the case. Maybe it's better to create an arena
            // here on top of the allocator we get.

            const inputs = try allocator.create(blk: {
                comptime var typeInfo = std.builtin.Type{
                    .Struct = .{
                        .layout = .Auto,
                        .fields = undefined,
                        .decls = &[_]std.builtin.Type.Declaration{},
                        .is_tuple = true,
                    },
                };

                comptime var fields: [models.len]std.builtin.Type.StructField = undefined;
                inline for (&fields, models, 0..) |*field, Model, i| {
                    const Type = [Model.input_size]fnn;

                    field.* = .{
                        .name = std.fmt.comptimePrint("{}", .{i}),
                        .type = Type,
                        .default_value = null,
                        .is_comptime = false,
                        .alignment = @alignOf(Type),
                    };
                }

                typeInfo.Struct.fields = &fields;

                break :blk @Type(typeInfo);
            });
            defer allocator.destroy(inputs);

            comptime var model_param_idx: comptime_int = undefined;

            inputs[0] = input.*;
            model_param_idx = 0;
            inline for (0..inputs.len - 1) |i| {
                const param_it = param[model_param_idx..][0..models[i].param_size];
                model_param_idx += models[i].param_size;
                try models[i].eval(allocator, &inputs[i], param_it, &inputs[i + 1]);
            }

            // XXX we can get away with this not being a tuple, but a buffer large enough for the
            // largest element of the chain rule
            const chain_rule = try allocator.create(blk: {
                comptime var typeInfo = std.builtin.Type{
                    .Struct = .{
                        .layout = .Auto,
                        .fields = undefined,
                        .decls = &[_]std.builtin.Type.Declaration{},
                        .is_tuple = true,
                    },
                };

                comptime var fields: [models.len - 1]std.builtin.Type.StructField = undefined;
                inline for (&fields, 0..) |*field, i| {
                    const Type = [models[i].output_size][output_size]fnn;

                    field.* = .{
                        .name = std.fmt.comptimePrint("{}", .{i}),
                        .type = Type,
                        .default_value = null,
                        .is_comptime = false,
                        .alignment = @alignOf(Type),
                    };
                }

                typeInfo.Struct.fields = &fields;

                break :blk @Type(typeInfo);
            });
            defer allocator.destroy(chain_rule);

            // XXX as far as I know zig does not have reverse iteration
            model_param_idx = param.len;
            inline for (1..models.len + 1) |models_len_minus_i| {
                const i = models.len - models_len_minus_i;
                model_param_idx -= models[i].param_size;

                const param_it = param[model_param_idx..][0..models[i].param_size];
                const derivative_it = derivative[model_param_idx..][0..models[i].param_size];

                // XXX this if is in place because zig thinks the alignemnt of an array of size 0 is nonzero
                if (models[i].param_size > 0) {
                    if (i < chain_rule.len) {
                        const param_derivative = try allocator.create([models[i].param_size][models[i].output_size]fnn);
                        defer allocator.destroy(param_derivative);
                        try models[i].param_derive(allocator, &inputs[i], param_it, param_derivative);
                        matrix_multiply(param_derivative, &chain_rule[i], derivative_it);
                    } else {
                        try models[i].param_derive(allocator, &inputs[i], param_it, derivative_it);
                    }
                }

                if (i >= 1) {
                    if (i < chain_rule.len) {
                        const input_derivative = try allocator.create([models[i].input_size][models[i].output_size]fnn);
                        defer allocator.destroy(input_derivative);
                        try models[i].input_derive(allocator, &inputs[i], param_it, input_derivative);
                        matrix_multiply(input_derivative, &chain_rule[i], &chain_rule[i - 1]);
                    } else {
                        try models[i].input_derive(allocator, &inputs[i], param_it, &chain_rule[i - 1]);
                    }
                }
            }
        }
    };
}

fn MeanSquaredError(comptime size: comptime_int) type {
    return struct {
        fn eval(prediction: *const [size]fnn, truth: *const [size]fnn) fnn {
            var mse: fnn = 0;
            for (prediction, truth) |p, t| mse += (p - t) * (p - t);
            mse /= size;
            return mse;
        }

        fn derive(prediction: *const [size]fnn, truth: *const [size]fnn, derivative: *[size]fnn) void {
            for (derivative, prediction, truth) |*d, p, t| d.* = (p - t) * 2 / size;
        }
    };
}

test Chain {
    // The test has been reapropriated from
    // https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
    // the derivatives for b1 are different, because the original uses a single param for all the perceptrons,
    // while this implementation has a different one for each perceptron...

    const Model = Chain(.{
        FullyConnected(3, 2),
        Sigmoid(2),
        FullyConnected(2, 2),
        Sigmoid(2),
    });

    const Error = MeanSquaredError(2);

    const input = [_]fnn{ 1, 4, 5 };
    const truth = [_]fnn{ 0.1, 0.05 };
    //                   b1   w1   w3   w5   b1   w2   w4   w6   b2   w7   w9   b2   w8   w10
    const param = [_]fnn{ 0.5, 0.1, 0.3, 0.5, 0.5, 0.2, 0.4, 0.6, 0.5, 0.7, 0.9, 0.5, 0.8, 0.1 };

    var output: [2]fnn = undefined;

    try Model.eval(std.testing.allocator, &input, &param, &output);

    for ([_]fnn{ 0.8896, 0.8004 }, output) |e, o| {
        try std.testing.expectApproxEqAbs(e, o, 0.0001);
    }

    var param_output_derivative: [Model.param_size][Model.output_size]fnn = undefined;
    try Model.param_derive(std.testing.allocator, &input, &param, &param_output_derivative);

    var output_error_derivative: [Model.output_size]fnn = undefined;
    Error.derive(&output, &truth, &output_error_derivative);

    var param_error_derivative: [Model.param_size]fnn = undefined;
    matrix_vector_multiply(&param_output_derivative, &output_error_derivative, &param_error_derivative);

    //          b1      w1      w3      w5      b1      w2      w4      w6      b2      w7      w9      b2      w8      w10
    for ([_]fnn{ 0.0020, 0.0020, 0.0079, 0.0099, 0.0004, 0.0004, 0.0016, 0.0020, 0.0776, 0.0765, 0.0772, 0.1199, 0.1183, 0.1193 }, param_error_derivative) |ped, it| {
        try std.testing.expectApproxEqAbs(ped, it, 0.0001);
    }
}

/// CITE Adam: A Method for Stochastic Optimization
///
/// Good values are:
/// stepsize = 0.001
/// decay_rate_1 = 0.9
/// decay_rate_2 = 0.999
/// epsilon = 1e-8
fn Adam(comptime param_size: comptime_int, comptime stepsize: fnn, comptime decay_rate_1: fnn, comptime decay_rate_2: fnn) type {
    return struct {
        const epsilon = 1e-8;

        movement_1: [param_size]fnn = [_]fnn{0} ** param_size,
        movement_2: [param_size]fnn = [_]fnn{0} ** param_size,

        bias_correction_1: fnn = 1,
        bias_correction_2: fnn = 1,

        fn optimize(self: *@This(), param: *[param_size]fnn, gradient: *const [param_size]fnn) void {
            self.bias_correction_1 *= decay_rate_1;
            self.bias_correction_2 *= decay_rate_2;

            for (&self.movement_1, gradient) |*m, g| m.* = decay_rate_1 * m.* + (1 - decay_rate_1) * g;
            for (&self.movement_2, gradient) |*v, g| v.* = decay_rate_2 * v.* + (1 - decay_rate_2) * g * g;

            for (param, self.movement_1, self.movement_2) |*p, m, v| {
                const m_hat = m / (1 - self.bias_correction_1);
                const v_hat = v / (1 - self.bias_correction_2);
                p.* = p.* - stepsize * m_hat / (v_hat + epsilon);
            }
        }
    };
}

pub fn main() !void {
    var entire_runtime_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer entire_runtime_arena.deinit();

    const inputs = blk: {
        var file = try std.fs.cwd().openFile("train-images.idx3-ubyte", .{});
        defer file.close();
        var buffered_reader = std.io.bufferedReader(file.reader());
        break :blk try mnist.read_images(fnn, buffered_reader.reader(), entire_runtime_arena.allocator());
    };

    const truths = blk: {
        var file = try std.fs.cwd().openFile("train-labels.idx1-ubyte", .{});
        defer file.close();
        var buffered_reader = std.io.bufferedReader(file.reader());
        break :blk try mnist.read_labels(fnn, buffered_reader.reader(), entire_runtime_arena.allocator());
    };

    var default_prng = std.rand.DefaultPrng.init(1);
    const random = default_prng.random();

    const Model = Chain(.{
        FullyConnected(28 * 28, 28 * 28),
        Tanh(28 * 28),
        FullyConnected(28 * 28, 10),
        Softmax(10),
    });
    const Error = MeanSquaredError(Model.output_size);
    var optimizer = Adam(Model.param_size, 0.001, 0.9, 0.999){};

    const param = try entire_runtime_arena.allocator().create([Model.param_size]fnn);
    const gradient = try entire_runtime_arena.allocator().create([Model.param_size]fnn);

    for (param) |*it| it.* = random.floatNorm(@TypeOf(it.*));

    std.debug.print("Begin training!\n", .{});

    var hot_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer hot_arena.deinit();

    for (0..10_000) |iteration| {
        const eval_err = iteration % 10 == 0;
        var err: fnn = 0;

        var timer = try std.time.Timer.start();

        for (0..1) |_| {
            const idx = random.intRangeLessThan(usize, 0, inputs.len);
            const input = inputs[idx];
            const truth = truths[idx];

            const output = try hot_arena.allocator().create([Model.output_size]fnn);
            try Model.eval(hot_arena.allocator(), &input, param, output);

            const param_output_derivative = try hot_arena.allocator().create([Model.param_size][Model.output_size]fnn);
            try Model.param_derive(hot_arena.allocator(), &input, param, param_output_derivative);

            const output_error_derivative = try hot_arena.allocator().create([Model.output_size]fnn);
            Error.derive(output, &truth, output_error_derivative);

            const param_error_derivative = try hot_arena.allocator().create(MatrixVectorProduct(@TypeOf(param_output_derivative.*), @TypeOf(output_error_derivative.*)));
            matrix_vector_multiply(param_output_derivative, output_error_derivative, param_error_derivative);
            for (gradient, param_error_derivative) |*g, cg| g.* += cg;

            if (eval_err) err += Error.eval(output, &truth);

            _ = hot_arena.reset(.retain_capacity);
        }

        std.debug.print("{}ms\n", .{timer.lap() / std.time.ns_per_ms});

        for (gradient) |*it| it.* /= @floatFromInt(inputs.len);

        err /= @floatFromInt(inputs.len);

        optimizer.optimize(param, gradient);

        if (eval_err) std.debug.print("Iteration: {}  Error: {}\n", .{ iteration, err });
    }

    _ = hot_arena.reset(.free_all);

    {
        const test_inputs = blk: {
            var file = try std.fs.cwd().openFile("t10k-images.idx3-ubyte", .{});
            defer file.close();
            var buffered_reader = std.io.bufferedReader(file.reader());
            break :blk try mnist.read_images(fnn, buffered_reader.reader(), entire_runtime_arena.allocator());
        };

        const test_truths = blk: {
            var file = try std.fs.cwd().openFile("t10k-labels.idx1-ubyte", .{});
            defer file.close();
            var buffered_reader = std.io.bufferedReader(file.reader());
            break :blk try mnist.read_labels(fnn, buffered_reader.reader(), entire_runtime_arena.allocator());
        };

        var error_rate: f32 = 0;

        for (test_inputs, test_truths) |input, truth| {
            const output = try hot_arena.allocator().create([Model.output_size]fnn);
            try Model.eval(hot_arena.allocator(), &input, param, output);

            var output_digit: usize = 0;
            for (output, 0..) |it, i| {
                if (it > output[output_digit]) output_digit = i;
            }

            var truth_digit: usize = 0;
            for (truth, 0..) |it, i| {
                if (it > truth[truth_digit]) truth_digit = i;
            }

            if (output_digit != truth_digit) error_rate += 1;

            _ = hot_arena.reset(.retain_capacity);
        }

        std.debug.print("\n{d} error rate\n", .{error_rate / @as(f32, @floatFromInt(test_inputs.len))});
    }
}
