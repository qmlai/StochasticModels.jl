
function CoxIngersollRossModel(
    u0::R,
    params::NT,
    tspan::Tuple{Float64,Float64} = (0.0, 1.0);
    kwargs...,
) where {R<:Real,NT<:NamedTuple{(:κ, :θ, :σ),<:Tuple}}

    function f(u, p, t)
        @unpack κ, θ = p

        κ * (θ - u)
    end

    function g(u, p, t)
        @unpack σ = p

        σ * √u
    end

    SDEProblem{false}(f, g, u0, tspan, params; kwargs...)

end

function CoxIngersollRossModel(
    u0::Vector{<:Real},
    params::NT,
    tspan::Tuple{Float64,Float64} = (0.0, 1.0);
    kwargs...,
) where {NT<:NamedTuple{(:κ, :θ, :σ),<:Tuple}}

    function f!(du, u, p, t)
        @unpack κ, θ = p

        du .= κ * (θ - u)
        return nothing
    end


    if typeof(params.σ) <: Matrix

        N = length(u0)
        nrp = zeros(eltype(u0), N, N)

        function g_non_diag!(du, u, p, t)
            @unpack σ = p
            du .= Diagonal(.√u)
            du .*= σ
            return nothing
        end

        return SDEProblem{true}(
            f!,
            g_non_diag!,
            u0,
            tspan,
            params,
            noise_rate_prototype = nrp,
            kwargs...,
        )

    elseif typeof(params.σ) <: Vector

        function g_diag!(du, u, p, t)
            @unpack σ = p
            du .= .√u
            du .*= σ
            return nothing
        end

        return SDEProblem{true}(f!, g_diag!, u0, tspan, params, kwargs...)
    else
        error("σ must be a Matrix or a Vector")
    end
end

function CoxIngersollRossModel(
    u0::Vector{<:Real},
    params::NT,
    tspan::Tuple{Float64,Float64} = (0.0, 1.0);
    kwargs...,
) where {NT<:NamedTuple{(:κ, :θ, :σ, :Γ),<:Tuple}}

    function f!(du, u, p, t)
        @unpack κ, θ = p

        du .= κ * (θ - u)
        return nothing
    end

    if typeof(params.σ) <: Matrix

        N = length(u0)
        nrp = zeros(eltype(u0), N, N)

        function g_non_diag!(du, u, p, t)
            @unpack σ = p
            du .= Diagonal(.√u)
            du .*= σ
            return nothing
        end

        N = length(params.θ)

        cir_noise = CorrelatedWienerProcess!(params.Γ, tspan[1], zeros(N), zeros(N))

        return SDEProblem{true}(
            f!,
            g_non_diag!,
            u0,
            tspan,
            params,
            noise_rate_prototype = nrp,
            noise = cir_noise,
            kwargs...,
        )

    elseif typeof(params.σ) <: Vector

        function g_diag!(du, u, p, t)
            @unpack σ = p
            du .= .√u
            du .*= σ
            return nothing
        end

        N = length(params.θ)

        cir_noise = CorrelatedWienerProcess!(params.Γ, tspan[1], zeros(N), zeros(N))

        return SDEProblem{true}(
            f!,
            g_non_diag!,
            u0,
            tspan,
            params,
            noise = cir_noise,
            kwargs...,
        )

    else
        error("σ must be a Matrix or a Vector")
    end
end
