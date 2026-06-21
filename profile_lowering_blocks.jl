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
const VALIDATION_RTOL = parse(Float64, get(ENV, "SUNREP_LOWERING_COLUMNMAP_RTOL", "1.0e-10"))
const VALIDATION_ATOL = parse(Float64, get(ENV, "SUNREP_LOWERING_COLUMNMAP_ATOL", "1.0e-10"))

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

function column_key(m1::Int, m2::Int, d1::Int)
    return m1 + (m2 - 1) * d1
end

function column_index(key::Int, d1::Int)
    m2 = div(key - 1, d1) + 1
    m1 = key - (m2 - 1) * d1
    return CartesianIndex(m1, m2)
end

function get_rhs_column!(
        rhs_columns::Vector{Vector{T}},
        mask_keys::Vector{Int},
        colmap::Dict{Int, Int},
        key::Int,
        imax::Int,
        ::Type{T}
    ) where {T}
    return get!(colmap, key) do
        push!(mask_keys, key)
        push!(rhs_columns, zeros(T, imax))
        return length(mask_keys)
    end
end

function get_rhs_column_vector!(
        rhs_columns::Vector{Vector{T}},
        mask_keys::Vector{Int},
        col_of_key::Vector{Int},
        key::Int,
        imax::Int,
        ::Type{T}
    ) where {T}
    col = @inbounds col_of_key[key]
    if col == 0
        push!(mask_keys, key)
        push!(rhs_columns, zeros(T, imax))
        col = length(mask_keys)
        @inbounds col_of_key[key] = col
    end
    return col
end

function profiled_lower_weight_CGC_columnmap!(CGC, s1::I, s2::I, s3::I) where {I <: SUNIrrep{N}} where {N}
    N123 = size(CGC, 4)
    T = eltype(CGC)
    d1 = dim(s1)

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)

    map1 = SUNRepresentations.weightmap(basis(s1))
    map2 = SUNRepresentations.weightmap(basis(s2))
    map3 = SUNRepresentations.weightmap(basis(s3))

    w3list = sort(collect(keys(map3)); rev = true)
    wshift = div(sum(weight(s1)) + sum(weight(s2)) - sum(weight(s3)), N)

    rhs_columns = Vector{T}[]
    mask_keys = Int[]
    colmap = Dict{Int, Int}()

    summary = (;
        blocks = Ref(0),
        rhs_contributions = Ref(0),
        output_writes = Ref(0),
        nnz_added = Ref(0),
        max_imax = Ref(0),
        max_jmax = Ref(0),
        max_rhscols = Ref(0),
        total_dense_entries = Ref(0),
        eqs_build_time = Ref(0.0),
        rhs_generation_time = Ref(0.0),
        rhs_materialize_time = Ref(0.0),
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

            empty!(rhs_columns)
            empty!(mask_keys)
            empty!(colmap)
            rhs_generation_time = 0.0
            rhs_contributions = 0

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
                                col = get_rhs_column!(
                                    rhs_columns, mask_keys, colmap,
                                    column_key(m1, m2, d1), imax, T
                                )
                                rhs_columns[col][row_index] += Jm1coeff * CGCcoeff
                                rhs_contributions += 1
                            end

                            w2 = Base.setindex(w2′, w2′[l] - 1, l)
                            w2 = Base.setindex(w2, w2′[l + 1] + 1, l + 1)
                            for m2 in get(map2, w2, EMPTY_INDEX_LIST)
                                m1 = m1′
                                Jm2coeff = J⁻₂[m2, m2′]
                                col = get_rhs_column!(
                                    rhs_columns, mask_keys, colmap,
                                    column_key(m1, m2, d1), imax, T
                                )
                                rhs_columns[col][row_index] += Jm2coeff * CGCcoeff
                                rhs_contributions += 1
                            end
                        end
                    end
                    rhs_generation_time += seconds(rhs_generation_start)
                end
            end

            rhs_materialize_start = time_ns()
            rhs = Matrix{T}(undef, imax, length(rhs_columns))
            @inbounds for col in eachindex(rhs_columns)
                rhs[:, col] .= rhs_columns[col]
            end
            rhs_materialize_time = seconds(rhs_materialize_start)

            qr_factor_start = time_ns()
            eqs_qr = qr!(eqs)
            qr_factor_time = seconds(qr_factor_start)

            ldiv_start = time_ns()
            sols = ldiv!(eqs_qr, rhs)
            ldiv_time = seconds(ldiv_start)

            sparse_write_start = time_ns()
            nnz_before = length(CGC.data)
            @inbounds for (i, key) in enumerate(mask_keys)
                Im1m2 = column_index(key, d1)
                for (j, m3) in enumerate(m3list)
                    CGC[Im1m2, m3, α] += sols[j, i]
                end
            end
            sparse_write_time = seconds(sparse_write_start)

            rhscols = length(mask_keys)
            output_writes = rhscols * jmax
            nnz_added = max(length(CGC.data) - nnz_before, 0)
            block_time = seconds(block_start)

            summary.blocks[] += 1
            summary.rhs_contributions[] += rhs_contributions
            summary.output_writes[] += output_writes
            summary.nnz_added[] += nnz_added
            summary.max_imax[] = max(summary.max_imax[], imax)
            summary.max_jmax[] = max(summary.max_jmax[], jmax)
            summary.max_rhscols[] = max(summary.max_rhscols[], rhscols)
            summary.total_dense_entries[] += imax * jmax
            summary.eqs_build_time[] += eqs_build_time
            summary.rhs_generation_time[] += rhs_generation_time
            summary.rhs_materialize_time[] += rhs_materialize_time
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
                    rhs_contributions,
                    output_writes,
                    nnz_added,
                    total_time = block_time,
                    eqs_build_time,
                    rhs_generation_time,
                    rhs_materialize_time,
                    qr_factor_time,
                    ldiv_time,
                    sparse_write_time,
                ),
            )
        end
    end

    return (; summary, block_rows, total_time = seconds(total_start), nnz = length(CGC.data))
end

function profiled_lower_weight_CGC_vectormap!(CGC, s1::I, s2::I, s3::I; skipzero::Bool = false) where {I <: SUNIrrep{N}} where {N}
    N123 = size(CGC, 4)
    T = eltype(CGC)
    d1 = dim(s1)
    d2 = dim(s2)

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)

    map1 = SUNRepresentations.weightmap(basis(s1))
    map2 = SUNRepresentations.weightmap(basis(s2))
    map3 = SUNRepresentations.weightmap(basis(s3))

    w3list = sort(collect(keys(map3)); rev = true)
    wshift = div(sum(weight(s1)) + sum(weight(s2)) - sum(weight(s3)), N)

    rhs_columns = Vector{T}[]
    mask_keys = Int[]
    col_of_key = zeros(Int, d1 * d2)

    summary = (;
        blocks = Ref(0),
        candidate_cgccoeffs = Ref(0),
        skipped_zero_cgccoeffs = Ref(0),
        rhs_contributions = Ref(0),
        output_writes = Ref(0),
        nnz_added = Ref(0),
        max_imax = Ref(0),
        max_jmax = Ref(0),
        max_rhscols = Ref(0),
        total_dense_entries = Ref(0),
        eqs_build_time = Ref(0.0),
        rhs_generation_time = Ref(0.0),
        rhs_materialize_time = Ref(0.0),
        qr_factor_time = Ref(0.0),
        ldiv_time = Ref(0.0),
        sparse_write_time = Ref(0.0),
    )
    block_rows = NamedTuple[]
    total_start = time_ns()

    for α in 1:N123
        for w3 in view(w3list, 2:length(w3list))
            @inbounds for key in mask_keys
                col_of_key[key] = 0
            end
            empty!(rhs_columns)
            empty!(mask_keys)

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
            rhs_generation_time = 0.0
            candidate_cgccoeffs = 0
            skipped_zero_cgccoeffs = 0
            rhs_contributions = 0

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
                            candidate_cgccoeffs += 1
                            if skipzero && iszero(CGCcoeff)
                                skipped_zero_cgccoeffs += 1
                                continue
                            end

                            w1 = Base.setindex(w1′, w1′[l] - 1, l)
                            w1 = Base.setindex(w1, w1′[l + 1] + 1, l + 1)
                            for m1 in get(map1, w1, EMPTY_INDEX_LIST)
                                m2 = m2′
                                Jm1coeff = J⁻₁[m1, m1′]
                                col = get_rhs_column_vector!(
                                    rhs_columns, mask_keys, col_of_key,
                                    column_key(m1, m2, d1), imax, T
                                )
                                rhs_columns[col][row_index] += Jm1coeff * CGCcoeff
                                rhs_contributions += 1
                            end

                            w2 = Base.setindex(w2′, w2′[l] - 1, l)
                            w2 = Base.setindex(w2, w2′[l + 1] + 1, l + 1)
                            for m2 in get(map2, w2, EMPTY_INDEX_LIST)
                                m1 = m1′
                                Jm2coeff = J⁻₂[m2, m2′]
                                col = get_rhs_column_vector!(
                                    rhs_columns, mask_keys, col_of_key,
                                    column_key(m1, m2, d1), imax, T
                                )
                                rhs_columns[col][row_index] += Jm2coeff * CGCcoeff
                                rhs_contributions += 1
                            end
                        end
                    end
                    rhs_generation_time += seconds(rhs_generation_start)
                end
            end

            rhs_materialize_start = time_ns()
            rhs = Matrix{T}(undef, imax, length(rhs_columns))
            @inbounds for col in eachindex(rhs_columns)
                rhs[:, col] .= rhs_columns[col]
            end
            rhs_materialize_time = seconds(rhs_materialize_start)

            qr_factor_start = time_ns()
            eqs_qr = qr!(eqs)
            qr_factor_time = seconds(qr_factor_start)

            ldiv_start = time_ns()
            sols = ldiv!(eqs_qr, rhs)
            ldiv_time = seconds(ldiv_start)

            sparse_write_start = time_ns()
            nnz_before = length(CGC.data)
            @inbounds for (i, key) in enumerate(mask_keys)
                Im1m2 = column_index(key, d1)
                for (j, m3) in enumerate(m3list)
                    CGC[Im1m2, m3, α] += sols[j, i]
                end
            end
            sparse_write_time = seconds(sparse_write_start)

            rhscols = length(mask_keys)
            output_writes = rhscols * jmax
            nnz_added = max(length(CGC.data) - nnz_before, 0)
            block_time = seconds(block_start)

            summary.blocks[] += 1
            summary.candidate_cgccoeffs[] += candidate_cgccoeffs
            summary.skipped_zero_cgccoeffs[] += skipped_zero_cgccoeffs
            summary.rhs_contributions[] += rhs_contributions
            summary.output_writes[] += output_writes
            summary.nnz_added[] += nnz_added
            summary.max_imax[] = max(summary.max_imax[], imax)
            summary.max_jmax[] = max(summary.max_jmax[], jmax)
            summary.max_rhscols[] = max(summary.max_rhscols[], rhscols)
            summary.total_dense_entries[] += imax * jmax
            summary.eqs_build_time[] += eqs_build_time
            summary.rhs_generation_time[] += rhs_generation_time
            summary.rhs_materialize_time[] += rhs_materialize_time
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
                    candidate_cgccoeffs,
                    skipped_zero_cgccoeffs,
                    rhs_contributions,
                    output_writes,
                    nnz_added,
                    total_time = block_time,
                    eqs_build_time,
                    rhs_generation_time,
                    rhs_materialize_time,
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

function print_columnmap_timing_summary(result, s1, s2, s3, N123; label = "column-map")
    summary = result.summary
    total = result.total_time
    println()
    println(label, " lower_weight_CGC! profiling summary")
    println("channel: ", s1, " x ", s2, " -> ", s3)
    println("N123: ", N123)
    println("total lowering time: ", format_seconds(total))
    println("number of lowering blocks: ", summary.blocks[])
    if hasproperty(summary, :candidate_cgccoeffs)
        skipped = summary.skipped_zero_cgccoeffs[]
        candidates = summary.candidate_cgccoeffs[]
        skipped_pct = candidates > 0 ? 100 * skipped / candidates : 0.0
        println("candidate CGC coeffs: ", candidates)
        println("skipped zero CGC coeffs: ", skipped, "  (", round(skipped_pct; digits = 2), "%)")
    end
    println("total rhs contributions: ", summary.rhs_contributions[])
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
        ("rhs generation incl map", summary.rhs_generation_time[]),
        ("rhs materialize", summary.rhs_materialize_time[]),
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

function print_columnmap_top_blocks(result; n = 12, label = "column-map")
    rows = sort(result.block_rows; by = row -> row.total_time, rev = true)
    println()
    println("slowest ", label, " lowering blocks:")
    for row in Iterators.take(rows, n)
        println(
            "alpha=", row.α,
            " imax=", row.imax,
            " jmax=", row.jmax,
            " rhscols=", row.rhscols,
            hasproperty(row, :candidate_cgccoeffs) ? string(" candidates=", row.candidate_cgccoeffs) : "",
            hasproperty(row, :skipped_zero_cgccoeffs) ? string(" skipped_zero=", row.skipped_zero_cgccoeffs) : "",
            " rhs_contributions=", row.rhs_contributions,
            " writes=", row.output_writes,
            " nnz_added=", row.nnz_added,
            " total=", format_seconds(row.total_time),
            " rhs_gen=", format_seconds(row.rhs_generation_time),
            " rhs_mat=", format_seconds(row.rhs_materialize_time),
            " qr=", format_seconds(row.qr_factor_time),
            " ldiv=", format_seconds(row.ldiv_time),
            " write=", format_seconds(row.sparse_write_time),
            " w3=", row.w3,
        )
    end
end

function compare_sparse_arrays(ref, test)
    ref_norm2 = 0.0
    diff_norm2 = 0.0
    max_abs_error = 0.0
    only_ref = 0
    only_test = 0

    for (idx, ref_value) in ref.data
        test_value = get(test.data, idx, zero(eltype(test)))
        diff = ref_value - test_value
        ref_norm2 += abs2(ref_value)
        diff_norm2 += abs2(diff)
        max_abs_error = max(max_abs_error, abs(diff))
        haskey(test.data, idx) || (only_ref += 1)
    end
    for idx in keys(test.data)
        if !haskey(ref.data, idx)
            test_value = test.data[idx]
            diff_norm2 += abs2(test_value)
            max_abs_error = max(max_abs_error, abs(test_value))
            only_test += 1
        end
    end

    ref_norm = sqrt(ref_norm2)
    diff_norm = sqrt(diff_norm2)
    relative_error = diff_norm / max(ref_norm, eps(float(one(eltype(ref)))))
    return (;
        ref_nnz = length(ref.data),
        test_nnz = length(test.data),
        only_ref,
        only_test,
        max_abs_error,
        diff_norm,
        ref_norm,
        relative_error,
    )
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
    CGC_highest = hw_timed.value
    println("highest-weight CGC time: ", format_seconds(hw_timed.time))
    println("highest-weight allocated memory: ", format_bytes(hw_timed.bytes))
    println("highest-weight nnz: ", length(CGC_highest.data))

    CGC_ref = deepcopy(CGC_highest)
    CGC_columnmap = deepcopy(CGC_highest)
    CGC_vectormap = deepcopy(CGC_highest)
    CGC_skipzero = deepcopy(CGC_highest)
    CGC_highest = nothing
    GC.gc()

    println()
    println("running reference profiled lowering")
    ref_timed = @timed profiled_lower_weight_CGC!(CGC_ref, s1, s2, s3)
    ref_result = ref_timed.value
    println("reference lowering wrapper time: ", format_seconds(ref_timed.time))
    println("reference lowering allocated memory: ", format_bytes(ref_timed.bytes))
    println("reference lowering GC time: ", format_seconds(ref_timed.gctime))
    print_timing_summary(ref_result, s1, s2, s3, N123)
    print_top_blocks(ref_result)

    println()
    println("running column-map profiled lowering")
    columnmap_timed = @timed profiled_lower_weight_CGC_columnmap!(CGC_columnmap, s1, s2, s3)
    columnmap_result = columnmap_timed.value
    println("column-map lowering wrapper time: ", format_seconds(columnmap_timed.time))
    println("column-map lowering allocated memory: ", format_bytes(columnmap_timed.bytes))
    println("column-map lowering GC time: ", format_seconds(columnmap_timed.gctime))
    print_columnmap_timing_summary(columnmap_result, s1, s2, s3, N123)
    print_columnmap_top_blocks(columnmap_result)

    println()
    println("validating reference vs column-map lowering")
    comparison = compare_sparse_arrays(CGC_ref, CGC_columnmap)
    println("ref nnz: ", comparison.ref_nnz)
    println("column-map nnz: ", comparison.test_nnz)
    println("only in ref: ", comparison.only_ref)
    println("only in column-map: ", comparison.only_test)
    println("max_abs_error: ", comparison.max_abs_error)
    println("diff_norm: ", comparison.diff_norm)
    println("ref_norm: ", comparison.ref_norm)
    println("relative_error: ", comparison.relative_error)
    println("validation rtol: ", VALIDATION_RTOL)
    println("validation atol: ", VALIDATION_ATOL)
    validation_passed =
        comparison.relative_error <= VALIDATION_RTOL &&
        comparison.max_abs_error <= VALIDATION_ATOL
    println("validation passed: ", validation_passed)
    validation_passed || error("column-map lowering validation failed")
    println("speedup vs reference total lowering: ", ref_result.total_time / columnmap_result.total_time)
    columnmap_total_time = columnmap_result.total_time

    CGC_columnmap = nothing
    columnmap_result = nothing
    GC.gc()

    println()
    println("running vector-map profiled lowering")
    println("vector-map lookup array memory: ", format_bytes(sizeof(Int) * dim(s1) * dim(s2)))
    vectormap_timed = @timed profiled_lower_weight_CGC_vectormap!(CGC_vectormap, s1, s2, s3)
    vectormap_result = vectormap_timed.value
    println("vector-map lowering wrapper time: ", format_seconds(vectormap_timed.time))
    println("vector-map lowering allocated memory: ", format_bytes(vectormap_timed.bytes))
    println("vector-map lowering GC time: ", format_seconds(vectormap_timed.gctime))
    print_columnmap_timing_summary(vectormap_result, s1, s2, s3, N123; label = "vector-map")
    print_columnmap_top_blocks(vectormap_result; label = "vector-map")

    println()
    println("validating reference vs vector-map lowering")
    comparison = compare_sparse_arrays(CGC_ref, CGC_vectormap)
    println("ref nnz: ", comparison.ref_nnz)
    println("vector-map nnz: ", comparison.test_nnz)
    println("only in ref: ", comparison.only_ref)
    println("only in vector-map: ", comparison.only_test)
    println("max_abs_error: ", comparison.max_abs_error)
    println("diff_norm: ", comparison.diff_norm)
    println("ref_norm: ", comparison.ref_norm)
    println("relative_error: ", comparison.relative_error)
    println("validation rtol: ", VALIDATION_RTOL)
    println("validation atol: ", VALIDATION_ATOL)
    validation_passed =
        comparison.relative_error <= VALIDATION_RTOL &&
        comparison.max_abs_error <= VALIDATION_ATOL
    println("validation passed: ", validation_passed)
    validation_passed || error("vector-map lowering validation failed")
    println("speedup vs reference total lowering: ", ref_result.total_time / vectormap_result.total_time)
    println("speedup vs column-map total lowering: ", columnmap_total_time / vectormap_result.total_time)
    vectormap_total_time = vectormap_result.total_time

    CGC_vectormap = nothing
    vectormap_result = nothing
    GC.gc()

    println()
    println("running vector-map skipzero profiled lowering")
    println("vector-map skipzero lookup array memory: ", format_bytes(sizeof(Int) * dim(s1) * dim(s2)))
    skipzero_timed = @timed profiled_lower_weight_CGC_vectormap!(CGC_skipzero, s1, s2, s3; skipzero = true)
    skipzero_result = skipzero_timed.value
    println("vector-map skipzero lowering wrapper time: ", format_seconds(skipzero_timed.time))
    println("vector-map skipzero lowering allocated memory: ", format_bytes(skipzero_timed.bytes))
    println("vector-map skipzero lowering GC time: ", format_seconds(skipzero_timed.gctime))
    print_columnmap_timing_summary(skipzero_result, s1, s2, s3, N123; label = "vector-map skipzero")
    print_columnmap_top_blocks(skipzero_result; label = "vector-map skipzero")

    println()
    println("validating reference vs vector-map skipzero lowering")
    comparison = compare_sparse_arrays(CGC_ref, CGC_skipzero)
    println("ref nnz: ", comparison.ref_nnz)
    println("vector-map skipzero nnz: ", comparison.test_nnz)
    println("only in ref: ", comparison.only_ref)
    println("only in vector-map skipzero: ", comparison.only_test)
    println("max_abs_error: ", comparison.max_abs_error)
    println("diff_norm: ", comparison.diff_norm)
    println("ref_norm: ", comparison.ref_norm)
    println("relative_error: ", comparison.relative_error)
    println("validation rtol: ", VALIDATION_RTOL)
    println("validation atol: ", VALIDATION_ATOL)
    validation_passed =
        comparison.relative_error <= VALIDATION_RTOL &&
        comparison.max_abs_error <= VALIDATION_ATOL
    println("validation passed: ", validation_passed)
    validation_passed || error("vector-map skipzero lowering validation failed")
    println("speedup vs reference total lowering: ", ref_result.total_time / skipzero_result.total_time)
    println("speedup vs column-map total lowering: ", columnmap_total_time / skipzero_result.total_time)
    println("speedup vs vector-map total lowering: ", vectormap_total_time / skipzero_result.total_time)

    CGC_ref = nothing
    CGC_skipzero = nothing
    ref_result = nothing
    skipzero_result = nothing
    GC.gc()
    flush(stdout)
end
