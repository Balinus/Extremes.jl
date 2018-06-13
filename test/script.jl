
using Distributions
using Extremes

pd = GeneralizedExtremeValue(0,1,.1)
y = rand(pd,50)
θ̂ = gevfit(y)


μ = linspace(0,3,300)
pd = GeneralizedExtremeValue.(μ,1,.1)
y = rand.(pd)
θ̂ = gevfit(y,method="ml", location_covariate = μ)


ϕ = linspace(0,1,300)
pd = GeneralizedExtremeValue.(0,exp.(ϕ),.1)
y = rand.(pd)
θ̂ = gevfit(y,method="ml", initialvalues = [0;0;0;.1], logscale_covariate = ϕ)


μ = linspace(0,3,300)
ϕ = linspace(0,1,300)
pd = GeneralizedExtremeValue.(μ,exp.(ϕ),.1)
y = rand.(pd)
θ̂ = gevfit(y,method="ml", location_covariate = μ, logscale_covariate = ϕ)

# θ̂ = params(f)

# H = gevhessian(y,θ̂...)
