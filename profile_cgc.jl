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

using Logging

const SCRIPT_DIR = @__DIR__

const CACHE_DIR = get(
    ENV,
    "SUNREP_CGC_CACHE_DIR",
    get(ENV, "SUNREP_TEST_CACHE_DIR", mktempdir(; prefix = "sunrep_cgc_profile_"))
)
ENV["SUNREP_CGC_CACHE_DIR"] = CACHE_DIR
ENV["SUNREP_PROFILE_CGC"] = get(ENV, "SUNREP_PROFILE_CGC", "1")
ENV["SUNREP_PROFILE_CGC_MIN_N"] = get(ENV, "SUNREP_PROFILE_CGC_MIN_N", "5")

global_logger(ConsoleLogger(stderr, Logging.Info))

@info "Using isolated CGC cache directory" CACHE_DIR

cd(SCRIPT_DIR)

@info "Starting profiled test.jl"
total_time = @elapsed begin
    include(joinpath(SCRIPT_DIR, "test.jl"))
end

@info "Finished profiled test.jl" total_time
