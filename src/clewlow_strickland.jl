
function ClewlowStricklandModel(
    u0::R,
    parameters::NT,
    tspan  = (0.0, 1.0);
) where {R<:Real,NT<:NamedTuple{(:α, :σ, :τ),<:Tuple}}
    function f(u, p, t)
        return 0.0
    end

    function g(u, p, t)
        @unpack α, σ, τ = p

        return σ * exp(-α * τ)
    end

    SDEProblem{false}(f, g, u0, tspan, parameters)
end

function ClewlowStricklandModel(
    u0::Array{<:Real},
    parameters::NT,
    tspan  = (0.0, 1.0);
) where {NT<:NamedTuple{(:α, :σ, :τ),<:Tuple}}
    @unpack τ = parameters

    N = length(τ)

    params = (; parameters..., N = N, z = zeros(eltype(τ), N))

    function f!(du, u, p, t)
        @unpack z = p
        du .= z
        return nothing
    end

    function g!(du, u, p, t)
        @unpack α, σ, τ, N = p

        for i = 1:N
            du[i, 1] = σ[1] * exp(-α * τ[i])
            du[i, 2] = σ[2]
        end

        return nothing
    end

    SDEProblem{true}(
        f!,
        g!,
        u0,
        tspan,
        params,
        noise_rate_prototype = zeros(eltype(τ), N, 2),
    )
end

function ClewlowStricklandModel(
    u0::Array{<:Real},
    parameters::NT,
    tspan  = (0.0, 1.0);
) where {NT<:NamedTuple{(:α, :σ, :τ, :Γ),<:Tuple}}
    @unpack τ = parameters

    N = length(τ)

    params = (; parameters..., N = N, z = zeros(eltype(τ), N))

    function f!(du, u, p, t)
        @unpack z = p
        du .= z
        return nothing
    end

    function g!(du, u, p, t)
        @unpack α, σ, τ, N = p

        for i = 1:N
            du[i, 1] = σ[1] * exp(-α * τ[i])
            du[i, 2] = σ[2]
        end

        return nothing
    end

    cs_noise = CorrelatedWienerProcess!(params.Γ, tspan[1], zeros(N), zeros(2))

    SDEProblem{true}(
        f!,
        g!,
        u0,
        tspan,
        params,
        noise = cs_noise,
        noise_rate_prototype = zeros(eltype(τ), N, 2),
    )
end

