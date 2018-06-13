
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

    return [μ, σ, ξ]
end

function gevfit(y::Array{Float64,1}; method="ml", initialvalues=Float64[], location_covariate=Float64[], logscale_covariate =Float64[])

    if method == "ml"

        if isempty(location_covariate) & isempty(logscale_covariate)
            if isempty(initialvalues)
                initialvalues = gevfitlmom(y)
            end

            fobj(μ, ϕ, ξ) = sum(gevloglike.(y,μ,exp(ϕ),ξ))
            mle = Model(solver=IpoptSolver(print_level=0,sb="yes"))
            JuMP.register(mle,:fobj,3,fobj,autodiff=true)
            @variable(mle, μ, start = initialvalues[1])
            @variable(mle, ϕ, start = log(initialvalues[2]))
            @variable(mle, ξ, start = initialvalues[3])
            @NLobjective(mle, Max, fobj(μ, ϕ, ξ) )
            solution  = JuMP.solve(mle)

            if solution == :Optimal
                θ = [getvalue(μ) exp(getvalue(ϕ)) getvalue(ξ)]
            else
                error("The algorithm did not find a solution.")
            end

        elseif !isempty(location_covariate) & isempty(logscale_covariate)

            if isempty(initialvalues)
                initialvalues = zeros(4)
                if length(y)>20
                    θ₀ = gevfitlmom(y[1:20])
                else
                    θ₀ = gevfitlmom(y)
                end
                initialvalues[1] = θ₀[1]
                initialvalues[3] = log(θ₀[2])
                initialvalues[4] = θ₀[3]
            end

            fobj_location(μ₀, μ₁, ϕ, ξ) = sum(gevloglike.(y,μ₀+μ₁*location_covariate,exp(ϕ),ξ))
            #mle = Model(solver=IpoptSolver(print_level=0, sb="yes"))
            mle = Model(solver=IpoptSolver())
            JuMP.register(mle,:fobj_location,4,fobj_location,autodiff=true)
            @variable(mle, μ₀, start=  initialvalues[1])
            @variable(mle, μ₁, start = initialvalues[2])
            @variable(mle, ϕ, start = initialvalues[3])
            @variable(mle, ξ, start= initialvalues[4])
            @NLobjective(mle, Max, fobj_location(μ₀, μ₁, ϕ, ξ) )
            solution  = JuMP.solve(mle)

            if solution == :Optimal
                θ = [getvalue(μ₀) getvalue(μ₁) exp(getvalue(ϕ)) getvalue(ξ)]
            else
                error("The algorithm did not find a solution.")
            end

        elseif isempty(location_covariate) & !isempty(logscale_covariate)

            if isempty(initialvalues)
                initialvalues = zeros(4)
                if length(y)>20
                    θ₀ = gevfitlmom(y[1:20])
                else
                    θ₀ = gevfitlmom(y)
                end
                initialvalues[1] = θ₀[1]
                initialvalues[2] = log(θ₀[2])
                initialvalues[4] = θ₀[3]
            end

            fobj_logscale(μ, ϕ₀, ϕ₁, ξ) = sum(gevloglike.(y,μ,exp.(ϕ₀+ϕ₁*logscale_covariate),ξ))
            mle = Model(solver=IpoptSolver(print_level=0, sb="yes"))
            JuMP.register(mle,:fobj_logscale,4,fobj_logscale,autodiff=true)
            @variable(mle, μ, start=  initialvalues[1])
            @variable(mle, ϕ₀, start = initialvalues[2])
            @variable(mle, ϕ₁, start = initialvalues[3])
            @variable(mle, ξ, start= initialvalues[4])
            @NLobjective(mle, Max, fobj_logscale(μ, ϕ₀, ϕ₁, ξ) )
            solution  = JuMP.solve(mle)

            if solution == :Optimal
                θ = [getvalue(μ) getvalue(ϕ₀) getvalue(ϕ₁) getvalue(ξ)]
            else
                error("The algorithm did not find a solution.")
            end

        elseif !isempty(location_covariate) & !isempty(logscale_covariate)

            if isempty(initialvalues)
                initialvalues = zeros(5)
                if length(y)>20
                    θ₀ = gevfitlmom(y[1:20])
                else
                    θ₀ = gevfitlmom(y)
                end
                initialvalues[1] = θ₀[1]
                initialvalues[3] = log(θ₀[2])
                initialvalues[5] = θ₀[3]
            end

            fobj_locationscale(μ₀, μ₁, ϕ₀, ϕ₁, ξ) = sum(gevloglike.(y,μ₀+μ₁*location_covariate,exp.(ϕ₀+ϕ₁*logscale_covariate),ξ))
            mle = Model(solver=IpoptSolver(print_level=0, sb="yes"))
            JuMP.register(mle,:fobj_locationscale,5,fobj_locationscale,autodiff=true)
            @variable(mle, μ₀, start=  initialvalues[1])
            @variable(mle, μ₁, start = initialvalues[2])
            @variable(mle, ϕ₀, start = initialvalues[3])
            @variable(mle, ϕ₁, start = initialvalues[4])
            @variable(mle, ξ, start= initialvalues[5])
            @NLobjective(mle, Max, fobj_locationscale(μ₀, μ₁, ϕ₀, ϕ₁, ξ) )
            solution  = JuMP.solve(mle)

            if solution == :Optimal
                θ = [getvalue(μ₀) getvalue(μ₁) getvalue(ϕ₀) getvalue(ϕ₁) getvalue(ξ)]
            else
                error("The algorithm did not find a solution.")
            end

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
