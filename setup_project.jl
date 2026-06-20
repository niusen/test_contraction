#!/usr/bin/env julia

import Pkg

const SCRIPT_DIR = @__DIR__
const DEFAULT_SUNREP_PATH = normpath(
    joinpath(SCRIPT_DIR, "..", "..", "Tensor network", "SUNrepresentations", "SUNRepresentations.jl")
)
const SUNREP_PATH = get(ENV, "SUNREP_DEV_PATH", DEFAULT_SUNREP_PATH)
const SUNREP_URL = get(ENV, "SUNREP_DEV_URL", "")
const SUNREP_REV = get(ENV, "SUNREP_DEV_REV", "")

Pkg.activate(SCRIPT_DIR)
if !isempty(SUNREP_URL)
    spec = isempty(SUNREP_REV) ?
        Pkg.PackageSpec(url = SUNREP_URL) :
        Pkg.PackageSpec(url = SUNREP_URL, rev = SUNREP_REV)
    Pkg.add(spec)
elseif isdir(SUNREP_PATH)
    Pkg.develop(path = SUNREP_PATH)
else
    error(
        "Cannot find SUNRepresentations.jl at SUNREP_DEV_PATH=$(SUNREP_PATH). " *
        "Set SUNREP_DEV_URL to your GitHub fork URL on the server."
    )
end
Pkg.instantiate()
Pkg.status()
