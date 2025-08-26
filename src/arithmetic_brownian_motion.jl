
#-------------------------------------------------------------------------------------------#
#                              1-D Arithmetic Brownian Motion                                #
#-------------------------------------------------------------------------------------------#

function ArithmeticBrownianMotionModel(
    u0::R,
    params::NT;
    tspan::TS = (0.0, 1.0),
    kwargs...,
) where {R<:Real,TS<:Tuple{<:Real,<:Real},NT<:NamedTuple{(:μ, :σ),<:Tuple}}

    function f(u, p, t)
        @unpack μ = p

        μ
    end

    function g(u, p, t)
        @unpack σ = p

        σ
    end

    return SDEProblem{false}(f, g, u0, tspan, params; kwargs...)
end


#-------------------------------------------------------------------------------------------#
#                    N-D uncorrelated Arithmetic Brownian Motion                             #
#-------------------------------------------------------------------------------------------#


function ArithmeticBrownianMotionModel(
    u0::Array{<:Real},
    params::NT;
    tspan = (0.0, 1.0),
    kwargs...,
) where {NT<:NamedTuple{(:μ, :σ),<:Tuple}}

    N = length(u0)

    function f!(du, u, p, t)
        @unpack μ = p

        @. du = μ
        return nothing
    end

    if typeof(params.σ) <: Matrix

        function g_non_diag!(du, u, p, t)
            @unpack σ = p

            du .= σ
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

            @. du = σ
            return nothing
        end

        return SDEProblem{true}(f!, g_diag!, u0, tspan, params; kwargs...)
    else
        error("σ must be a Matrix or a Vector")
    end
end

#-------------------------------------------------------------------------------------------#
#                      N-D correlated Arithmetic Brownian Motion                             #
#-------------------------------------------------------------------------------------------#

function ArithmeticBrownianMotionModel(
    u0::Array{<:Real},
    params::NT,
    tspan = (0.0, 1.0);
    kwargs...,
) where {NT<:NamedTuple{(:μ, :σ, :Γ),<:Tuple}}

    function f!(du, u, p, t)
        @unpack μ = p

        @. du = μ
        return nothing
    end

    N = length(u0)
    if typeof(params.σ) <: Matrix

        function g_non_diag!(du, u, p, t)
            @unpack σ = p

            du .= σ
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

            @. du = σ 
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
