
function HullWhiteModel(u0::R, params, tspan = (0.0, 1.0); kwargs...) where {R<:Real}

    function f(u, p, t)
        @unpack κ, θ = p

        κ(t) * (θ(t) - u)
    end

    function g(u, p, t)
        @unpack σ = p

        σ(t)
    end

    SDEProblem{false}(f, g, u0, tspan, params, kwargs...)

end

function HullWhiteModel(
    u0::R,
    params,
    noise,
    tspan = (0.0, 1.0);
    kwargs...,
) where {R<:Vector{<:Real}}

    function f!(du, u, p, t)
        @unpack κ, θ, κ_cache, θ_cache = p

        κ(κ_cache, t)
        θ(θ_cache, t)

        du .= κ_cache * (θ_cache - u)

        return nothing
    end

    function g!(du, u, p, t)
        @unpack σ, σ_cache = p

        σ(σ_cache, t)

        du .= σ_cache

        return nothing
    end

    if noise == :Diagonal
        N = length(u0)

        params = (;
            params...,
            κ_cache = zeros(eltype(u0), N, N),
            θ_cache = zeros(eltype(u0), N),
            σ_cache = zeros(eltype(u0), N),
        )

        return SDEProblem{true}(f!, g!, u0, tspan, params; kwargs...)

    elseif noise == :NonDiagonal
        N = length(u0)

        params = (;
            params...,
            κ_cache = zeros(eltype(u0), N, N),
            θ_cache = zeros(eltype(u0), N),
            σ_cache = zeros(eltype(u0), N, N),
        )

        nrp = zeros(eltype(u0), N, N)

        return SDEProblem{true}(
            f!,
            g!,
            u0,
            tspan,
            params;
            noise_rate_prototype = nrp,
            kwargs...,
        )

    else
        error("Noise must be either Diagonal or Nondiagonal")
    end
end
