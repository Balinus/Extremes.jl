
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

function gevfitlmom(y::Array{Float64,1})

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

function gevfit(y::Array{Float64,1}; method="ml", initialvalues=[])

    if method == "ml"

        if isempty(initialvalues)
            initialvalues = params(gevfitlmom(y))
        end

        # Log-likelihood function to maximize
        # In optimization, the function to maximize is refered to the "obective function"
        fobj(μ, ϕ, ξ) = sum(gevloglike.(y,μ,exp(ϕ),ξ))

        # https://discourse.julialang.org/t/the-function-to-optimize-in-jump/4964/13
        mle = Model(solver=IpoptSolver(print_level=0))
        JuMP.register(mle,:fobj,3,fobj,autodiff=true)
        @variable(mle, μ, start = initialvalues[1])
        @variable(mle, ϕ, start = log(initialvalues[2]))
        @variable(mle, ξ, start = initialvalues[3])
        @NLobjective(mle, Max, fobj(μ, ϕ, ξ) )
        solution  = JuMP.solve(mle)

        if solution == :Optimal
            return GeneralizedExtremeValue(getvalue(μ), exp(getvalue(ϕ)), getvalue(ξ))
        else
            error("The algorithm did not find a solution.")
        end

    elseif method == "lmom"
        return gevfitlmom(y)
    end
end


function gevhessian(y::Array{Float64,1},μ::Real,σ::Real,ξ::Real)

    #= Estimate the hessian matrix evaluated at (μ, σ, ξ) for the iid gev random sample y =#

    logl(θ) = sum(gevloglike.(y,θ[1],θ[2],θ[3]))

    H = ForwardDiff.hessian(logl, [μ σ ξ])

end
