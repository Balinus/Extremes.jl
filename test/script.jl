
using Extremes, Distributions

pd = GeneralizedExtremeValue(0,1,.1)

y = rand(pd,300)

f = gevfit(y)

θ̂ = params(f)

H = gevhessian(y,θ̂...)

#= Location non-stationary =#

using Extremes, Distributions, DataFrames
using JuMP, Ipopt, ForwardDiff

t = linspace(0,99,100)
(μ₀, μ₁, σ, ξ) = (0.0, 0.025, 1.0, 0.1 )
μ = μ₀ + μ₁*t

pd = GeneralizedExtremeValue.(μ,σ,ξ)

y = rand.(pd)

D = DataFrame(Maxima=y,Year=t)

fobj(μ₀, μ₁, ϕ, ξ) = sum(gevloglike.(y,μ₀+μ₁*t,exp(ϕ),ξ))

# https://discourse.julialang.org/t/the-function-to-optimize-in-jump/4964/13
mle = Model(solver=IpoptSolver(print_level=0))
JuMP.register(mle,:fobj,4,fobj,autodiff=true)
@variable(mle, μ₀, start=0)
@variable(mle, μ₁, start=0)
@variable(mle, ϕ, start=0)
@variable(mle, ξ, start=.1)
@NLobjective(mle, Max, fobj(μ₀, μ₁, ϕ, ξ) )

solution  = JuMP.solve(mle)

θ̂ = [getvalue(μ₀), getvalue(μ₁), exp(getvalue(ϕ)), getvalue(ξ)]

logl(θ) = sum(gevloglike.(y,θ[1]+θ[2]*t,θ[3],θ[4]))

H = ForwardDiff.hessian(logl, θ̂)


#= Location and scale non-stationary =#

using Extremes, Distributions, DataFrames
using JuMP, Ipopt, ForwardDiff

t = linspace(0,99,1000)
(μ₀, μ₁, ϕ₀, ϕ₁, ξ) = (0.0, 0.025, 0.0, .005, 0.1 )
μ = μ₀ + μ₁*t
σ = exp.( ϕ₀ + ϕ₁*t )

pd = GeneralizedExtremeValue.(μ,σ,ξ)

y = rand.(pd)

D = DataFrame(Maxima=y,Year=t)

fobj(μ₀, μ₁, ϕ₀, ϕ₁ , ξ) = sum(gevloglike.(y,μ₀+μ₁*t,exp(ϕ₀+ϕ₁*t),ξ))

# https://discourse.julialang.org/t/the-function-to-optimize-in-jump/4964/13
mle = Model(solver=IpoptSolver(print_level=0))
JuMP.register(mle,:fobj,5,fobj,autodiff=true)
@variable(mle, μ₀, start=0)
@variable(mle, μ₁, start=0)
@variable(mle, ϕ₀, start=0)
@variable(mle, ϕ₁, start=0)
@variable(mle, ξ, start=.1)
@NLobjective(mle, Max, fobj(μ₀, μ₁, ϕ₀, ϕ₁ , ξ) )

solution  = JuMP.solve(mle)

θ̂ = [getvalue(μ₀), getvalue(μ₁), getvalue(ϕ₀), getvalue(ϕ₁), getvalue(ξ)]

logl(θ) = sum(gevloglike.(y,θ[1]+θ[2]*t,exp.(θ[3]+θ[4]*t),θ[5]))

H = ForwardDiff.hessian(logl, θ̂)
