function SABRModel(u0::Vector{<:Real}, params, tspan = (0.0, 1.0); kwargs...)

    function f!(du, u, p, t)
        du[1] = 0.0
        du[2] = 0.0
        return nothing
    end

    function g!(du, u, p, t)
        @unpack β, α, Γ = p

        du[1] = u[2] * u[1]^β
        du[2] = α * u[2]
        return nothing
    end

    sabr_noise = CorrelatedWienerProcess!(params.Γ, tspan[1], zeros(2), zeros(2))

    SDEProblem{true}(f!, g!, u0, tspan, params, noise = sabr_noise, kwargs...)
end

function SABRModel(u0::SVector{2,<:Real}, params, tspan = (0.0, 1.0); kwargs...)

    function f(u, p, t)

        @SVector zeros(eltype(u), 2)
    end

    function g(u, p, t)
        @unpack β, α = p

        du1 = u[2] * u[1]^β
        du2 = α * u[2]

        @SVector [du1, du2]
    end

    w0 = @SVector zeros(2)
    z0 = @SVector zeros(2)

    sabr_noise = CorrelatedWienerProcess!(params.Γ, tspan[1], w0, z0)

    SDEProblem{false}(f, g, u0, tspan, params, noise = sabr_noise, kwargs...)
end
