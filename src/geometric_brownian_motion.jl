#-------------------------------------------------------------------------------------------#
#                              Analytical Geometric Brownian Motion                         #
#-------------------------------------------------------------------------------------------#

struct AnalyticalGeometricBrownianMotionModel{T} <: AnalyticalAbstractModel
    S0::T
    μ::T
    σ::T
    ts::Array{T}
end

function simulate(S0, μ, σ, T, M, N, gen)
    dt = T / M
    
    exp_μ_term = exp((μ - 0.5 * σ ^ 2) *  dt)
    σ_term = (σ * √dt)

    S = zeros(eltype(S0), N, M + 1)
    @inbounds for i = 1:N
        S[i, 1] = S0
        @inbounds for j = 2:M+1
            z = randn(gen, eltype(S0))
            S[i, j] = S[i, j - 1] * exp_μ_term * exp(σ_term * z)
        end
    end
    return S
end

function simulate!(S, S0, μ, σ, T, M, N, gen)
    dt = T / M
    
    exp_μ_term = exp((μ - 0.5 * σ ^ 2) *  dt)
    σ_term = (σ * √dt)

    @inbounds for i = 1:N
        S[i, 1] = S0
        @inbounds for j = 2:M+1
            z = randn(gen, eltype(S0))
            S[i, j] = S[i, j - 1] * exp_μ_term * exp(σ_term * z)
        end
    end
    nothing
end

function simulate(sde, N, gen)
    @unpack S0, μ, σ, ts = sde

    M = length(ts)

    S = simulate(S0, μ, σ, ts, M, N, gen)

    return S
end

#-------------------------------------------------------------------------------------------#
#                              1-D Geometric Brownian Motion                                #
#-------------------------------------------------------------------------------------------#

function GeometricBrownianMotionModel(
    u0::R,
    params::NT;
    tspan::TS = (0.0, 1.0),
    kwargs...,
) where {R<:Real,TS<:Tuple{<:Real,<:Real},NT<:NamedTuple{(:μ, :σ),<:Tuple}}

    function f(u, p, t)
        @unpack μ = p

        μ * u
    end

    function g(u, p, t)
        @unpack σ = p

        σ * u
    end

    return SDEProblem{false}(f, g, u0, tspan, params; kwargs...)
end


#-------------------------------------------------------------------------------------------#
#                    N-D uncorrelated Geometric Brownian Motion                             #
#-------------------------------------------------------------------------------------------#


function GeometricBrownianMotionModel(
    u0::Array{<:Real},
    params::NT;
    tspan = (0.0, 1.0),
    kwargs...,
) where {NT<:NamedTuple{(:μ, :σ),<:Tuple}}

    N = length(u0)

    function f!(du, u, p, t)
        @unpack μ = p

        @. du = μ * u
        return nothing
    end

    if typeof(params.σ) <: Matrix

        function g_non_diag!(du, u, p, t)
            @unpack σ = p

            du .= σ * Diagonal(u)
            return nothing
        end

        nrp = zeros(eltype(u0), N, N)

        return SDEProblem{true}(
            f!,
            g_non_diag!,
            u0,
            tspan,
            params;
            noise_rate_prototype = nrp,
            kwargs...,
        )

    elseif typeof(params.σ) <: Vector

        function g_diag!(du, u, p, t)
            @unpack σ = p

            @. du = σ * u
            return nothing
        end

        return SDEProblem{true}(f!, g_diag!, u0, tspan, params; kwargs...)
    else
        error("σ must be a Matrix or a Vector")
    end
end

#-------------------------------------------------------------------------------------------#
#                      N-D correlated Geometric Brownian Motion                             #
#-------------------------------------------------------------------------------------------#

function GeometricBrownianMotionModel(
    u0::Array{<:Real},
    params::NT,
    tspan = (0.0, 1.0);
    kwargs...,
) where {NT<:NamedTuple{(:μ, :σ, :Γ),<:Tuple}}

    function f!(du, u, p, t)
        @unpack μ = p

        @. du = μ * u
        return nothing
    end

    N = length(u0)
    if typeof(params.σ) <: Matrix

        function g_non_diag!(du, u, p, t)
            @unpack σ = p

            du .= σ * Diagonal(u)
            return nothing
        end

        nrp = zeros(eltype(u0), N, N)

        N = length(u0)
        # TODO: Z0 could be nothing if we use EM() because low order algorithms don´t need it
        gbm_noise = CorrelatedWienerProcess!(
            params.Γ,
            tspan[1],
            zeros(eltype(u0), N),
            zeros(eltype(u0), N),
        )

        return SDEProblem{true}(
            f!,
            g_non_diag!,
            u0,
            tspan,
            params;
            noise_rate_prototype = nrp,
            noise = gbm_noise,
            kwargs...,
        )

    elseif typeof(params.σ) <: Vector

        function g_diag!(du, u, p, t)
            @unpack σ = p

            @. du = σ * u
            return nothing
        end

        N = length(u0)
        # TODO: Z0 could be nothing if we use EM() because low order algorithms don´t need it
        gbm_noise = CorrelatedWienerProcess!(
            params.Γ,
            tspan[1],
            zeros(eltype(u0), N),
            zeros(eltype(u0), N),
        )

        return SDEProblem{true}(
            f!,
            g_diag!,
            u0,
            tspan,
            params;
            noise = gbm_noise,
            kwargs...,
        )
    else
        error("σ must be a Matrix or a Vector")
    end
end
