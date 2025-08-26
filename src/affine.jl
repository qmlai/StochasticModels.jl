
#-------------------------------------------------------------------------------------------#
#                                       1-D Affine SDE                                      #
#-------------------------------------------------------------------------------------------#

function AffineModel(
    u0::R,
    params,
    tspan::Tuple{Float64,Float64} = (0.0, 1.0);
    kwargs...,
) where {R<:Real}

    function f(u, p, t)
        @unpack κ, θ = p

        return κ(t) * (θ(t) - u)
    end

    function g(u, p, t)
        @unpack σ, α, β = p

        return σ(t) * √(α(t) + β(t) * u)
    end

    return SDEProblem{false}(f, g, u0, tspan, params; kwargs...)
end


#-------------------------------------------------------------------------------------------#
#                                       N-D Affine SDE                                      #
#-------------------------------------------------------------------------------------------#

function AffineModel(
    u0::Array{<:Real},
    params,
    noise,
    tspan::Tuple{Float64,Float64} = (0.0, 1.0);
    kwargs...,
)

    N = length(u0)

    function f!(du, u, p, t)
        @unpack κ, θ, κ_cache, θ_cache = p

        κ(κ_cache, t)
        θ(θ_cache, t)

        mul!(du, -κ_cache, u)
        du .+= κ_cache * θ_cache

        return nothing
    end

    if noise == :Diagonal

        function g_diag!(du, u, p, t)
            @unpack σ, α, β, σ_cache, α_cache, β_cache, cache = p

            σ(σ_cache, t)
            α(α_cache, t)
            β(β_cache, t)

            cache .= .√(α_cache + β_cache * u)

            du .= σ_cache .* cache

            return nothing
        end

        params = (;
            params...,
            κ_cache = zeros(eltype(u0), N, N),
            θ_cache = zeros(eltype(u0), N),
            σ_cache = zeros(eltype(u0), N),
            α_cache = zeros(eltype(u0), N),
            β_cache = zeros(eltype(u0), N, N),
            cache = zeros(eltype(u0), N),
        )

        return SDEProblem{true}(f!, g_diag!, u0, tspan, params; kwargs...)

    elseif noise == :NonDiagonal

        function g_non_diag!(du, u, p, t)
            @unpack σ, α, β, σ_cache, α_cache, β_cache, cache = p

            σ(σ_cache, t)
            α(α_cache, t)
            β(β_cache, t)

            cache .= Diagonal(.√(α_cache + β_cache * u))

            du .= σ_cache * cache

            return nothing
        end

        params = (;
            params...,
            κ_cache = zeros(eltype(u0), N, N),
            θ_cache = zeros(eltype(u0), N),
            σ_cache = zeros(eltype(u0), N, N),
            α_cache = zeros(eltype(u0), N),
            β_cache = zeros(eltype(u0), N, N),
            cache = zeros(eltype(u0), N, N),
        )

        nrp = zeros(N, N)

        return SDEProblem{true}(
            f!,
            g_non_diag!,
            u0,
            tspan,
            params,
            noise_rate_prototype = nrp;
            kwargs...,
        )

    else
        error("Noise type must be :Diagonal or :NonDiagonal")
    end
end
