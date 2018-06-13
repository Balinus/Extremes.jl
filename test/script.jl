
using Distributions
using Extremes

x = linspace(0,3,1000)

pd = GeneralizedExtremeValue.(x,1,.1)
pd = GeneralizedExtremeValue.(0,exp.(log1p.(x/2)),.1)
pd = GeneralizedExtremeValue.(x,exp.(log1p.(x/2)),.1)

y = rand.(pd)

# f = gevfitns(y,method="ml",location_covariate=collect(x))
g = gevfit(y,method="ml")
g = gevfit(y,method="ml", location_covariate=collect(x))
g = gevfit(y,method="ml", logscale_covariate = collect(x))
g = gevfit(y,method="ml", location_covariate=collect(x), logscale_covariate = collect(x))

# θ̂ = params(f)

# H = gevhessian(y,θ̂...)
