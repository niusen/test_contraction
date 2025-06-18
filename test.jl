
using TensorKit
using JLD2
using TensorKit
import TensorKit:dim
using TensorKitSectors
using SUNRepresentations


cd(@__DIR__)





######################
#load data
data=load("test.jld2");
envL=data["envL"][1].data;
envR=data["envR"][1].data;
H=data["H"][1];
pos=data["pos"];
L=data["L"];
AB=data["AB"];

mpo1=H.tensors[pos];
mpo2=H.tensors[pos+1];
#######################

#show information of tensors

println(space(AB,2))
println("dims of envL: ", (dim(space(envL,1)),dim(space(envL,2)),dim(space(envL,3))))
println("dims of envR: ", (dim(space(envR,1)),dim(space(envR,2)),dim(space(envR,3))))
println("dims of mpo1: ", (dim(space(mpo1,1)),dim(space(mpo1,2)),dim(space(mpo1,3)),dim(space(mpo1,4))))
println("dims of mpo2: ", (dim(space(mpo2,1)),dim(space(mpo2,2)),dim(space(mpo2,3)),dim(space(mpo2,4))))
println("dims of AB: ", (dim(space(AB,1)),dim(space(AB,2)),dim(space(AB,3)),dim(space(AB,4))))


#contract

@tensor tt[:]:=envL[-1,2,1]*mpo1[2,5,-2,3]*mpo2[5,6,-3,4]*AB[1,3,4,7]*envR[-4,6,7];


