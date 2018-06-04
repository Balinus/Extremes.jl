
function gevloglike(y::Real,μ::Real,σ::Real,ξ::Real)

    @assert σ > 0 "The scale parameter must be positive"

    z = (y-μ)/σ

        if abs(ξ) > eps()
            if (1+ξ*z) <= 0
                loglike = -Inf
            else
                loglike = -log(σ) - (1/ξ+1)*log1p(ξ*z) - (1+ξ*z).^(-1/ξ)
            end
        else
            loglike = -log(σ) - z - exp(-z)
        end

    return loglike

end

function gevfitlmom(y::Array{N,1} where N)

    n = length(y)
    sort!(y)
    r = 1:n

#     L-Moments estimations (Cunnane, 1989)
    b₀ = mean(y)
    b₁ = sum( (r-1).*y )/n/(n-1)
    b₂ = sum( (r-1).*(r-2).*y ) /n/(n-1)/(n-2)

    # GEV parameters estimations
    c = (2b₁ - b₀)/(3b₂ - b₀) - log(2)/log(3)
    k = 7.859c + 2.9554c^2
    σ = k *( 2b₁-b₀ ) /(1-2^(-k))/gamma(1+k)
    μ = b₀ - σ/k*( 1-gamma(1+k) )

    ξ = -k

    return GeneralizedExtremeValue(μ, σ, ξ)
end

function gevfit(y::Array{N,1} where N)

    # only MLE for the moment

    # Get initial values with the L-moments method
    pd₀ = gevfitlmom(y)

    (μ₀, σ₀, ξ₀) = params(pd₀)

  fobj(μ, ϕ, ξ) = sum(gevloglike.(y,μ,exp(ϕ),ξ))

  # https://discourse.julialang.org/t/the-function-to-optimize-in-jump/4964/13
  mle = Model(solver=IpoptSolver(print_level=0))
  JuMP.register(mle,:fobj,3,fobj,autodiff=true)
  @variable(mle, μ, start=μ₀)
  @variable(mle, ϕ, start=log(σ₀))
  @variable(mle, ξ, start=ξ₀)
  @NLobjective(mle, Max, fobj(μ, ϕ, ξ) )

  solution  = JuMP.solve(mle)

  θ̂ = [getvalue(μ), exp(getvalue(ϕ)), getvalue(ξ)]

  logl(θ) = sum(gevloglike.(y,θ...))

  H = ForwardDiff.hessian(logl, θ̂)

    if solution == :Optimal
        return GeneralizedExtremeValue(θ̂...)
        #return H
    else
      error("The algorithm did not find a solution.")
    end
end
