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
# Normal profiling run:
#     julia --project=. profile_cgc.jl
#
# Set SUNREP_PROFILE_CGC=1 for detailed CGC cache/current-stage/lowering logs.

using Logging

const SCRIPT_DIR = @__DIR__

const CACHE_DIR = get(
    ENV,
    "SUNREP_CGC_CACHE_DIR",
    get(ENV, "SUNREP_TEST_CACHE_DIR", mktempdir(; prefix = "sunrep_cgc_profile_"))
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

@info "Using isolated CGC cache directory" CACHE_DIR
@info "SUNREP_CGC_CACHE_DIR env" get(ENV, "SUNREP_CGC_CACHE_DIR", "")
@info "Current CGC status file" get(ENV, "SUNREP_CURRENT_CGC_FILE", "")
@info "CGC matrix-free mode" get(ENV, "SUNREP_CGC_MATRIXFREE", "")

using SUNRepresentations

@info "SUNRepresentations actual CGC cache path" SUNRepresentations.CGC_CACHE_PATH

cd(SCRIPT_DIR)

@info "Starting profiled test.jl"
total_time = @elapsed begin
    include(joinpath(SCRIPT_DIR, "test.jl"))
end

@info "Finished profiled test.jl" total_time
