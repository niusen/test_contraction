
using TensorKit
using JLD2
using TensorKit
using TensorKitSectors
using SUNRepresentations


cd(@__DIR__)





#####################
data=load("test.jld2");
envL=data["envL"][1].data;
envR=data["envR"][1].data;
H=data["H"][1];
pos=data["pos"];
L=data["L"];
AB=data["AB"];

mpo1=H.tensors[pos];
mpo2=H.tensors[pos+1];

@tensor tt[:]:=envL[-1,2,1]*mpo1[2,5,-2,3]*mpo2[5,6,-3,4]*AB[1,3,4,7]*envR[-4,6,7];


