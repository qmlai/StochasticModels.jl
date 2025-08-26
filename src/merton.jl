function MertonModel(u0::R, params, tspan = (0.0, 1.0); kwargs...) where {R<:Real}
    function f(u, p, t)
        @unpack μ = p

        μ * u
    end

    function g(u, p, t)
        @unpack σ = p

        σ * u
    end

    sde = SDEProblem{false}(f, g, u0, tspan, params, kwargs...)

    JumpProblem(sde)
end

function MertonModel(
    u0::Array{<:Real},
    params::NT,
    tspan = (0.0, 1.0);
    kwargs...,
) where {NT<:NamedTuple{(:μ, :σ),<:Tuple}}

    function f!(du, u, p, t)
        @unpack μ = p

        @. du = μ * u
        return nothing
    end

    function g!(du, u, p, t)
        @unpack σ = p

        du .= σ * u
        return nothing
    end

    sde = SDEProblem{true}(f!, g!, u0, tspan, params, kwargs...)

    JumpProblem(sde)
end
