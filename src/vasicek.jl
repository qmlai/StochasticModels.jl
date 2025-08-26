
function VasicekModel(u0::R, params, tspan = (0.0, 1.0); kwargs...) where {R<:Real}

    function f(u, p, t)
        @unpack κ, θ = p

        κ * (θ - u)
    end

    function g(u, p, t)
        @unpack σ = p

        σ
    end

    SDEProblem{false}(f, g, u0, tspan, params, kwargs...)

end

function VasicekModel(u0::Vector{<:Real}, params, tspan = (0.0, 1.0); kwargs...)

    function f!(du, u, p, t)
        @unpack κ, θ = p

        du .= κ * (θ - u)
        return nothing
    end

    function g!(du, u, p, t)
        @unpack σ = p

        du .= σ
        return nothing
    end

    if typeof(params.σ) <: Matrix

        N = length(u0)
        nrp = zeros(eltype(u0), N, N)

        return SDEProblem{true}(
            f!,
            g!,
            u0,
            tspan,
            params,
            noise_rate_prototype = nrp,
            kwargs...,
        )

    elseif typeof(params.σ) <: Vector

        return SDEProblem{true}(f!, g!, u0, tspan, params, kwargs...)
    else
        error("σ must be either a Matrix or a Vector")
    end
end
