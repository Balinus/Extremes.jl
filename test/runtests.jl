using Extremes
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

include("script.jl")

# write your own tests here
@test 1 == 2
