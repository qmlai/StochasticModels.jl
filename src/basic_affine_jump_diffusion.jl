
function BasicAffineJumpModel(
    u0::R,
    params,
    tspan::Tuple{Float64,Float64} = (0.0, 1.0);
    kwargs...,
) where {R<:Real}

    function f(u, p, t)
        @unpack κ, θ = p

        κ(t) * (θ(t) - u)
    end

    function g(u, p, t)
        @unpack Σ, α, β = p

        Σ(t) * √u
    end

    affine = SDEProblem{false}(f, g, u0, tspan, params; kwargs...)

    JumpProblem(affine)
end

function BasicAffineJumpModel(
    u0::Array{<:Real},
    params,
    tspan::Tuple{Float64,Float64} = (0.0, 1.0);
    kwargs...,
)

    N = length(u0)
    params = (;
        params...,
        κ_cache = zeros(eltype(u0), N, N),
        θ_cache = zeros(eltype(u0), N),
        Σ_cache = zeros(eltype(u0), N, N),
        α_cache = zeros(eltype(u0), N),
        β_cache = zeros(eltype(u0), N, N),
    )

    function f!(du, u, p, t)
        @unpack κ, θ, κ_cache, θ_cache = p

        κ(κ_cache, t)
        θ(θ_cache, t)

        mul!(du, -κ_cache, u)
        du .+= κ_cache * θ_cache

        return nothing
    end

    function g!(du, u, p, t)
        @unpack Σ, α, β, Σ_cache, α_cache, β_cache = p

        Σ(Σ_cache, t)
        α(α_cache, t)
        β(β_cache, t)

        du .= Σ_cache * √Diagonal(α_cache + β_cache * u)

        return nothing
    end

    nrp = zeros(N, N)

    sde = SDEProblem{true}(f!, g!, u0, tspan, params, noise_rate_prototype = nrp; kwargs...)

    JumpProblem(sde)
end
