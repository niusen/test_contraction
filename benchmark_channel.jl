#!/usr/bin/env julia

# Package setup / source update commands for the test environment.
# Uncomment and run once after changing the GitHub source or branch:
#
# import Pkg
# Pkg.activate(@__DIR__)
# Pkg.add(url = "https://github.com/niusen/SUNRepresentations.jl.git", rev = "main")
# Pkg.add(["TensorKit", "TensorKitSectors", "JLD2"])
# Pkg.instantiate()
#
# Normal benchmark run:
#     julia --project=. benchmark_channel.jl
#
# Set SUNREP_PROFILE_CGC=1 for detailed CGC cache/current-stage/lowering logs.

using Logging
using LinearAlgebra

const SCRIPT_DIR = @__DIR__

const CACHE_DIR = get(
    ENV,
    "SUNREP_CGC_CACHE_DIR",
    get(ENV, "SUNREP_TEST_CACHE_DIR", mktempdir(; prefix = "sunrep_cgc_channel_"))
)
ENV["SUNREP_CGC_CACHE_DIR"] = CACHE_DIR
ENV["SUNREP_PROFILE_CGC"] = get(ENV, "SUNREP_PROFILE_CGC", "0")
ENV["SUNREP_PROFILE_CGC_MIN_N"] = get(ENV, "SUNREP_PROFILE_CGC_MIN_N", "5")
ENV["SUNREP_CURRENT_CGC_FILE"] = get(
    ENV, "SUNREP_CURRENT_CGC_FILE", joinpath(SCRIPT_DIR, "current_cgc.txt")
)

# Full CGC highest-weight nullspace method:
#   "off"  = always use the original dense SVD
#   "on"   = always use the matrix-free iterative method
#   "auto" = use matrix-free when the dense matrix estimate is large enough
cgc_matrixfree_mode = "auto"
cgc_matrixfree_tol = 1.0e-13
cgc_matrixfree_maxiter = 1000
cgc_matrixfree_krylovdim = 120
cgc_matrixfree_restarts = 3

ENV["SUNREP_CGC_MATRIXFREE"] = get(ENV, "SUNREP_CGC_MATRIXFREE", cgc_matrixfree_mode)
ENV["SUNREP_CGC_MATRIXFREE_TOL"] = get(ENV, "SUNREP_CGC_MATRIXFREE_TOL", string(cgc_matrixfree_tol))
ENV["SUNREP_CGC_MATRIXFREE_MAXITER"] = get(ENV, "SUNREP_CGC_MATRIXFREE_MAXITER", string(cgc_matrixfree_maxiter))
ENV["SUNREP_CGC_MATRIXFREE_KRYLOVDIM"] = get(ENV, "SUNREP_CGC_MATRIXFREE_KRYLOVDIM", string(cgc_matrixfree_krylovdim))
ENV["SUNREP_CGC_MATRIXFREE_RESTARTS"] = get(ENV, "SUNREP_CGC_MATRIXFREE_RESTARTS", string(cgc_matrixfree_restarts))

global_logger(ConsoleLogger(stderr, Logging.Info))

using SUNRepresentations
import TensorKit: dim

SUNRepresentations.display_mode("dimension")

const T = Float64
const N = 6

# Edit these three lines to benchmark a different channel.
s1 = SUNIrrep{N}("3675")
s2 = SUNIrrep{N}("840")
s3 = SUNIrrep{N}("6")

# Edit these values to trade speed for matrix-free accuracy.
mf_tol = 1.0e-13
mf_maxiter = 1000
mf_krylovdim = 120
mf_restarts = 3

function format_bytes(bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = Float64(bytes)
    for (i, unit) in enumerate(units)
        if value < 1024 || i == length(units)
            return string(round(value; digits = 3), " ", unit)
        end
        value /= 1024
    end
end

function operator_storage_bytes(op)
    return sizeof(eltype(op.val)) * length(op.val) +
           sizeof(Int) * (length(op.src) + length(op.dst)) +
           sizeof(CartesianIndex{2}) * length(op.cols) +
           sizeof(CartesianIndex{3}) * length(op.rows)
end

@info "Using isolated CGC cache directory" CACHE_DIR
@info "SUNRepresentations actual CGC cache path" SUNRepresentations.CGC_CACHE_PATH
@info "Current CGC status file" get(ENV, "SUNREP_CURRENT_CGC_FILE", "")
@info "CGC matrix-free mode" get(ENV, "SUNREP_CGC_MATRIXFREE", "")
@info "Benchmark channel" s1 s2 s3

d1, d2, d3 = dim(s1), dim(s2), dim(s3)
multiplicity = directproduct(s1, s2)[s3]

println("channel: ", s1, " x ", s2, " -> ", s3)
println("dims: ", (d1, d2, d3))
println("multiplicity: ", multiplicity)

println("running dense highest-weight nullspace")
dense_timed = @timed SUNRepresentations.highest_weight_nullspace_dense_uncached(T, s1, s2, s3)
dense = dense_timed.value
println("HW equation: ", dense.op.M, " x ", dense.op.K)
println("dense memory estimate: ", SUNRepresentations._dense_memory_gib(T, dense.op.M, dense.op.K), " GiB")
println("dense matrix storage estimate: ", format_bytes(sizeof(T) * dense.op.M * dense.op.K))
println("highest-weight operator storage estimate: ", format_bytes(operator_storage_bytes(dense.op)))
println("highest-weight operator nonzeros: ", length(dense.op.val))
println("dense highest-weight nullspace time: ", dense_timed.time, " seconds")
println("dense Julia allocated memory: ", format_bytes(dense_timed.bytes))
println("dense GC time: ", dense_timed.gctime, " seconds")
dense_residual = norm(SUNRepresentations.mul_A(dense.op, dense.basis)) /
                 max(norm(dense.basis), eps(float(one(T))))
println("dense residual: ", dense_residual)

println("running matrix-free highest-weight nullspace")
mf_timed = @timed SUNRepresentations.highest_weight_nullspace_matrixfree_uncached(
    T, s1, s2, s3;
    tol = mf_tol,
    maxiter = mf_maxiter,
    krylovdim = mf_krylovdim,
    restarts = mf_restarts,
)
mf = mf_timed.value
println("matrix-free operator storage estimate: ", format_bytes(operator_storage_bytes(mf.op)))
println("matrix-free operator nonzeros: ", length(mf.op.val))
println("matrix-free time: ", mf_timed.time, " seconds")
println("matrix-free Julia allocated memory: ", format_bytes(mf_timed.bytes))
println("matrix-free GC time: ", mf_timed.gctime, " seconds")
println("matrix-free residual: ", mf.residual)
println("matrix-free ortherr: ", mf.ortherr)
println("matrix-free sigmas: ", mf.sigmas)
println("matrix-free discarded sigmas: ", mf.discarded_sigmas)
println("matrix-free raw sigmas: ", mf.raw_sigmas)
println("matrix-free eigenvalues: ", mf.eigenvalues)
println("matrix-free discarded eigenvalues: ", mf.discarded_eigenvalues)
println("matrix-free raw eigenvalues: ", mf.raw_eigenvalues)
println("matrix-free info: ", mf.info)
println("matrix-free selected attempt: ", mf.attempt, " / ", mf.restarts)

overlap = svdvals(dense.basis' * mf.basis)
println("subspace overlap singular values: ", overlap)
if size(dense.basis, 2) == 1 && size(mf.basis, 2) == 1
    phase = sign(dot(dense.basis[:, 1], mf.basis[:, 1]))
    phase = iszero(phase) ? one(T) : phase
    println("abs dot overlap: ", abs(dot(dense.basis[:, 1], mf.basis[:, 1])))
    println("vector difference after sign alignment: ", norm(dense.basis[:, 1] - phase * mf.basis[:, 1]))
end

if lowercase(get(ENV, "SUNREP_BENCH_FULL_CGC", "")) in ("1", "true", "yes", "on")
    println("running full CGC generation")
    full_time = @elapsed begin
        cgc = CGC(T, s1, s2, s3)
    end
    println("full CGC time: ", full_time, " seconds")
else
    println("full CGC generation skipped; set SUNREP_BENCH_FULL_CGC=1 to include lowering/cache write")
end
