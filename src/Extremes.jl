module Extremes

using Distributions
using JuMP, Ipopt
using ForwardDiff

include("functions.jl")

export gevfit, gevhessian

end # module
