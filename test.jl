using TensorKit
import TensorKit: dim
using TensorKitSectors
using SUNRepresentations

cd(@__DIR__)

SUNRepresentations.display_mode("dimension")

const T = Float64

# These spaces are transcribed from the printed space(envL), space(envR),
# space(mpo1), space(mpo2), and space(AB).  The tensors below are random
# TensorMaps with the same leg structure, so this test does not depend on
# an old test.jld2 serialization format.
Vleft = Rep[U₁ × SU{6}](
    (62, "15") => 1,
    (63, "20") => 2,
    (62, "105′") => 1,
    (62, "105") => 1,
    (64, "15⁺") => 3,
    (63, "70") => 2,
    (65, "6⁺") => 2,
    (64, "21⁺") => 1,
    (66, "1") => 1,
)

Vright = Rep[U₁ × SU{6}](
    (63, "20") => 1,
    (64, "15⁺") => 4,
    (63, "70") => 1,
    (65, "6⁺") => 8,
    (64, "21⁺") => 2,
    (66, "1") => 7,
    (67, "6") => 8,
    (64, "105⁺") => 1,
    (65, "210⁺") => 1,
    (68, "15") => 4,
    (65, "84⁺") => 5,
    (66, "189") => 3,
    (67, "210") => 1,
    (69, "20") => 1,
    (66, "35") => 8,
    (67, "84") => 5,
    (68, "105") => 1,
    (65, "840⁺") => 1,
    (66, "280") => 1,
    (64, "384⁺") => 1,
    (65, "120⁺") => 2,
)

Vbond7 = Rep[U₁ × SU{6}](
    (0, "1") => 2,
    (1, "6") => 7,
)

Vbond6 = Rep[U₁ × SU{6}](
    (0, "1") => 2,
    (1, "6") => 6,
)

Vphys = Rep[U₁ × SU{6}](
    (0, "1") => 1,
    (1, "6") => 1,
    (2, "15") => 1,
)

envL = randn(T, Vleft ⊗ Vbond7' ⊗ Vleft' ← one(Vleft))
envR = randn(T, Vright' ⊗ Vbond6 ⊗ Vright ← one(Vright))
mpo1 = randn(T, Vbond7 ⊗ Vbond7' ← Vphys' ⊗ Vphys)
mpo2 = randn(T, Vbond7 ⊗ Vbond6' ← Vphys' ⊗ Vphys)
AB = randn(T, Vleft ⊗ Vphys ⊗ Vphys ⊗ Vright' ← one(Vleft))

println(space(AB, 2))
println("dims of envL: ", (dim(space(envL, 1)), dim(space(envL, 2)), dim(space(envL, 3))))
println("dims of envR: ", (dim(space(envR, 1)), dim(space(envR, 2)), dim(space(envR, 3))))
println(
    "dims of mpo1: ",
    (dim(space(mpo1, 1)), dim(space(mpo1, 2)), dim(space(mpo1, 3)), dim(space(mpo1, 4)))
)
println(
    "dims of mpo2: ",
    (dim(space(mpo2, 1)), dim(space(mpo2, 2)), dim(space(mpo2, 3)), dim(space(mpo2, 4)))
)
println(
    "dims of AB: ",
    (dim(space(AB, 1)), dim(space(AB, 2)), dim(space(AB, 3)), dim(space(AB, 4)))
)

@tensor tt[:] := envL[-1, 2, 1] *
                 mpo1[2, 5, -2, 3] *
                 mpo2[5, 6, -3, 4] *
                 AB[1, 3, 4, 7] *
                 envR[-4, 6, 7]
