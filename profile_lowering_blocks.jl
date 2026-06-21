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
# Normal run:
#     julia --project=. profile_lowering_blocks.jl
#
# This file profiles lowering only. It does not modify SUNRepresentations source code,
# does not call CGC(...), and does not write a CGC disk-cache entry.

using LinearAlgebra
using Logging

const SCRIPT_DIR = @__DIR__
const CACHE_DIR = get(
    ENV,
    "SUNREP_CGC_CACHE_DIR",
    get(ENV, "SUNREP_TEST_CACHE_DIR", mktempdir(; prefix = "sunrep_cgc_lowering_"))
)
ENV["SUNREP_CGC_CACHE_DIR"] = CACHE_DIR
ENV["SUNREP_PROFILE_CGC"] = get(ENV, "SUNREP_PROFILE_CGC", "0")
ENV["SUNREP_CGC_MATRIXFREE"] = get(ENV, "SUNREP_CGC_MATRIXFREE", "auto")
ENV["SUNREP_CGC_MATRIXFREE_TOL"] = get(ENV, "SUNREP_CGC_MATRIXFREE_TOL", "1.0e-13")
ENV["SUNREP_CGC_MATRIXFREE_MAXITER"] = get(ENV, "SUNREP_CGC_MATRIXFREE_MAXITER", "1000")
ENV["SUNREP_CGC_MATRIXFREE_KRYLOVDIM"] = get(ENV, "SUNREP_CGC_MATRIXFREE_KRYLOVDIM", "120")
ENV["SUNREP_CGC_MATRIXFREE_RESTARTS"] = get(ENV, "SUNREP_CGC_MATRIXFREE_RESTARTS", "3")

global_logger(ConsoleLogger(stderr, Logging.Info))

using SUNRepresentations
import TensorKit: dim

SUNRepresentations.display_mode("dimension")

const T = Float64
const N = 6

# Edit this list to profile different lowering-heavy channels.
const CHANNELS = [
    ("1701⁺", "3675", "384⁺"),
    ("3675", "4410⁺", "120⁺"),
    ("5040⁺", "3675", "84⁺"),
]

const EMPTY_INDEX_LIST = Int[]

seconds(t0::UInt64) = 1.0e-9 * (time_ns() - t0)

function format_seconds(x)
    return string(round(x; digits = 6), " s")
end

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

function dense_memory_mib(::Type{T}, m, n) where {T}
    return sizeof(T) * Float64(m) * Float64(n) / 1024.0^2
end

function profiled_lower_weight_CGC!(CGC, s1::I, s2::I, s3::I) where {I <: SUNIrrep{N}} where {N}
    N123 = size(CGC, 4)
    T = eltype(CGC)

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)

    map1 = SUNRepresentations.weightmap(basis(s1))
    map2 = SUNRepresentations.weightmap(basis(s2))
    map3 = SUNRepresentations.weightmap(basis(s3))

    w3list = sort(collect(keys(map3)); rev = true)
    wshift = div(sum(weight(s1)) + sum(weight(s2)) - sum(weight(s3)), N)

    rhs_rows = Int[]
    rhs_cols = CartesianIndex{2}[]
    rhs_vals = T[]

    summary = (;
        blocks = Ref(0),
        rhs_triplets = Ref(0),
        output_writes = Ref(0),
        nnz_added = Ref(0),
        max_imax = Ref(0),
        max_jmax = Ref(0),
        max_rhscols = Ref(0),
        total_dense_entries = Ref(0),
        eqs_build_time = Ref(0.0),
        rhs_generation_time = Ref(0.0),
        unique_time = Ref(0.0),
        indexin_time = Ref(0.0),
        rhs_fill_time = Ref(0.0),
        qr_factor_time = Ref(0.0),
        ldiv_time = Ref(0.0),
        sparse_write_time = Ref(0.0),
    )
    block_rows = NamedTuple[]
    total_start = time_ns()

    for α in 1:N123
        for w3 in view(w3list, 2:length(w3list))
            block_start = time_ns()
            m3list = map3[w3]
            jmax = length(m3list)
            imax = sum(1:(N - 1)) do l
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                return length(get(map3, w3′, EMPTY_INDEX_LIST))
            end

            eqs_build_start = time_ns()
            eqs = Array{T}(undef, (imax, jmax))
            eqs_build_time = seconds(eqs_build_start)

            empty!(rhs_rows)
            empty!(rhs_cols)
            empty!(rhs_vals)
            rhs_generation_time = 0.0

            row_index = 0
            for (l, (J⁻₁, J⁻₂, J⁻₃)) in enumerate(zip(Jm_list1, Jm_list2, Jm_list3))
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                for m3′ in get(map3, w3′, EMPTY_INDEX_LIST)
                    row_index += 1

                    eqs_block_start = time_ns()
                    for (j, m3) in enumerate(m3list)
                        eqs[row_index, j] = J⁻₃[m3, m3′]
                    end
                    eqs_build_time += seconds(eqs_block_start)

                    rhs_generation_start = time_ns()
                    for (w1′, m1′list) in map1
                        w2′ = w3′ .- w1′ .+ wshift
                        m2′list = get(map2, w2′, EMPTY_INDEX_LIST)
                        isempty(m2′list) && continue
                        for m2′ in m2′list, m1′ in m1′list
                            CGCcoeff = CGC[m1′, m2′, m3′, α]

                            w1 = Base.setindex(w1′, w1′[l] - 1, l)
                            w1 = Base.setindex(w1, w1′[l + 1] + 1, l + 1)
                            for m1 in get(map1, w1, EMPTY_INDEX_LIST)
                                m2 = m2′
                                Jm1coeff = J⁻₁[m1, m1′]
                                push!(rhs_rows, row_index)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm1coeff * CGCcoeff)
                            end

                            w2 = Base.setindex(w2′, w2′[l] - 1, l)
                            w2 = Base.setindex(w2, w2′[l + 1] + 1, l + 1)
                            for m2 in get(map2, w2, EMPTY_INDEX_LIST)
                                m1 = m1′
                                Jm2coeff = J⁻₂[m2, m2′]
                                push!(rhs_rows, row_index)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm2coeff * CGCcoeff)
                            end
                        end
                    end
                    rhs_generation_time += seconds(rhs_generation_start)
                end
            end

            unique_start = time_ns()
            mask = unique(rhs_cols)
            unique_time = seconds(unique_start)

            indexin_start = time_ns()
            rhs_cols′ = indexin(rhs_cols, mask)
            indexin_time = seconds(indexin_start)

            rhs_fill_start = time_ns()
            rhs = zeros(T, imax, length(mask))
            @inbounds for (row, col, val) in zip(rhs_rows, rhs_cols′, rhs_vals)
                rhs[row, col] += val
            end
            rhs_fill_time = seconds(rhs_fill_start)

            qr_factor_start = time_ns()
            eqs_qr = qr!(eqs)
            qr_factor_time = seconds(qr_factor_start)

            ldiv_start = time_ns()
            sols = ldiv!(eqs_qr, rhs)
            ldiv_time = seconds(ldiv_start)

            sparse_write_start = time_ns()
            nnz_before = length(CGC.data)
            @inbounds for (i, Im1m2) in enumerate(mask)
                for (j, m3) in enumerate(m3list)
                    CGC[Im1m2, m3, α] += sols[j, i]
                end
            end
            sparse_write_time = seconds(sparse_write_start)

            rhscols = length(mask)
            output_writes = rhscols * jmax
            nnz_added = max(length(CGC.data) - nnz_before, 0)
            block_time = seconds(block_start)

            summary.blocks[] += 1
            summary.rhs_triplets[] += length(rhs_vals)
            summary.output_writes[] += output_writes
            summary.nnz_added[] += nnz_added
            summary.max_imax[] = max(summary.max_imax[], imax)
            summary.max_jmax[] = max(summary.max_jmax[], jmax)
            summary.max_rhscols[] = max(summary.max_rhscols[], rhscols)
            summary.total_dense_entries[] += imax * jmax
            summary.eqs_build_time[] += eqs_build_time
            summary.rhs_generation_time[] += rhs_generation_time
            summary.unique_time[] += unique_time
            summary.indexin_time[] += indexin_time
            summary.rhs_fill_time[] += rhs_fill_time
            summary.qr_factor_time[] += qr_factor_time
            summary.ldiv_time[] += ldiv_time
            summary.sparse_write_time[] += sparse_write_time

            push!(
                block_rows,
                (;
                    α,
                    w3,
                    imax,
                    jmax,
                    rhscols,
                    rhs_triplets = length(rhs_vals),
                    output_writes,
                    nnz_added,
                    total_time = block_time,
                    eqs_build_time,
                    rhs_generation_time,
                    unique_time,
                    indexin_time,
                    rhs_fill_time,
                    qr_factor_time,
                    ldiv_time,
                    sparse_write_time,
                ),
            )
        end
    end

    return (; summary, block_rows, total_time = seconds(total_start), nnz = length(CGC.data))
end

function print_timing_summary(result, s1, s2, s3, N123)
    summary = result.summary
    total = result.total_time
    println()
    println("lower_weight_CGC! profiling summary")
    println("channel: ", s1, " x ", s2, " -> ", s3)
    println("N123: ", N123)
    println("total lowering time: ", format_seconds(total))
    println("number of lowering blocks: ", summary.blocks[])
    println("total rhs triplets: ", summary.rhs_triplets[])
    println("total output writes: ", summary.output_writes[])
    println("total nnz added to CGC: ", summary.nnz_added[])
    println("final CGC nnz before purge: ", result.nnz)
    println("max imax: ", summary.max_imax[])
    println("max jmax: ", summary.max_jmax[])
    println("max rhscols: ", summary.max_rhscols[])
    println("max dense block memory: ", round(dense_memory_mib(T, summary.max_imax[], summary.max_jmax[]); digits = 3), " MiB")
    println("total dense entries: ", summary.total_dense_entries[])
    println()
    println("timing categories:")
    categories = [
        ("eqs build", summary.eqs_build_time[]),
        ("rhs contribution generation", summary.rhs_generation_time[]),
        ("unique(rhs_cols)", summary.unique_time[]),
        ("indexin(rhs_cols, mask)", summary.indexin_time[]),
        ("rhs allocation/fill", summary.rhs_fill_time[]),
        ("qr!(eqs)", summary.qr_factor_time[]),
        ("ldiv solve", summary.ldiv_time[]),
        ("CGC sparse write", summary.sparse_write_time[]),
    ]
    for (name, time) in categories
        pct = total > 0 ? 100 * time / total : 0.0
        println(rpad(name, 30), format_seconds(time), "  (", round(pct; digits = 2), "%)")
    end
end

function print_top_blocks(result; n = 12)
    rows = sort(result.block_rows; by = row -> row.total_time, rev = true)
    println()
    println("slowest lowering blocks:")
    for row in Iterators.take(rows, n)
        println(
            "alpha=", row.α,
            " imax=", row.imax,
            " jmax=", row.jmax,
            " rhscols=", row.rhscols,
            " rhs_triplets=", row.rhs_triplets,
            " writes=", row.output_writes,
            " nnz_added=", row.nnz_added,
            " total=", format_seconds(row.total_time),
            " rhs_gen=", format_seconds(row.rhs_generation_time),
            " unique=", format_seconds(row.unique_time),
            " indexin=", format_seconds(row.indexin_time),
            " rhs_fill=", format_seconds(row.rhs_fill_time),
            " qr=", format_seconds(row.qr_factor_time),
            " ldiv=", format_seconds(row.ldiv_time),
            " write=", format_seconds(row.sparse_write_time),
            " w3=", row.w3,
        )
    end
end

@info "Using isolated CGC cache directory" CACHE_DIR
@info "SUNRepresentations actual CGC cache path" SUNRepresentations.CGC_CACHE_PATH
@info "CGC matrix-free mode" get(ENV, "SUNREP_CGC_MATRIXFREE", "")

for (channel_index, (s1_name, s2_name, s3_name)) in enumerate(CHANNELS)
    s1 = SUNIrrep{N}(s1_name)
    s2 = SUNIrrep{N}(s2_name)
    s3 = SUNIrrep{N}(s3_name)

    println()
    println("=" ^ 80)
    println("channel ", channel_index, " / ", length(CHANNELS))
    println("channel: ", s1, " x ", s2, " -> ", s3)
    println("dims: ", (dim(s1), dim(s2), dim(s3)))
    N123 = directproduct(s1, s2)[s3]
    println("multiplicity: ", N123)

    println()
    println("building highest-weight CGC")
    hw_timed = @timed SUNRepresentations.highest_weight_CGC(T, s1, s2, s3)
    CGC = hw_timed.value
    println("highest-weight CGC time: ", format_seconds(hw_timed.time))
    println("highest-weight allocated memory: ", format_bytes(hw_timed.bytes))
    println("highest-weight nnz: ", length(CGC.data))

    println()
    println("running profiled lowering")
    lowering_timed = @timed profiled_lower_weight_CGC!(CGC, s1, s2, s3)
    result = lowering_timed.value
    println("profiled lowering wrapper time: ", format_seconds(lowering_timed.time))
    println("profiled lowering allocated memory: ", format_bytes(lowering_timed.bytes))
    println("profiled lowering GC time: ", format_seconds(lowering_timed.gctime))

    print_timing_summary(result, s1, s2, s3, N123)
    print_top_blocks(result)
    GC.gc()
end
