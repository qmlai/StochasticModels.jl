function GibsonSchwartzModel(u0::Array{<:Real}, parameters, tspan = (0.0, 1.0); kwargs...)

    function f!(du, u, p, t)
        @unpack r, κ, α, λ, σ₂ = p

        du[1] = u[1] * (r - u[2])
        du[2] = κ * (α - u[2]) - λ * σ₂
        return nothing
    end

    function g!(du, u, p, t)
        @unpack σ₁, σ₂ = p

        du[1] = σ₁ * u[1]
        du[2] = σ₂
        return nothing
    end

    SDEProblem{true}(f!, g!, u0, tspan, parameters)
end

function GibsonSchwartzModel(
    u0::SVector{2,<:Real},
    parameters,
    tspan = (0.0, 1.0);
    kwargs...,
)

    function f(u, p, t)
        @unpack r, κ, α, λ, σ₂ = p

        du₁ = u[1] * (r - u[2])
        du₂ = κ * (α - u[2]) - λ * σ₂
        return @SVector [du₁, du₂]
    end

    function g(u, p, t)
        @unpack σ₁, σ₂ = p

        du₁ = σ₁ * u[1]
        du₂ = σ₂
        return @SVector [du₁, du₂]
    end

    SDEProblem{false}(f, g, u0, tspan, parameters)
end

