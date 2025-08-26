
function LongstaffSchwartzModel(u0::Vector{<:Real}, params, tspan = (0.0, 1.0); kwargs...)

    function f!(du, u, p, t)
        @unpack κ₁, κ₂, θ₁, θ₂ = p

        du[1] = κ₁ * (θ₁ - u[1])
        du[2] = κ₂ * (θ₂ - u[2])
        return nothing
    end

    function g!(du, u, p, t)
        @unpack σ₁, σ₂ = p

        du[1] = σ₁ * √u[1]
        du[2] = σ₂ * √u[2]
        return nothing
    end

    SDEProblem{true}(f!, g!, u0, tspan, params, kwargs...)
end

function LongstaffSchwartzModel(
    u0::SVector{2,<:Real},
    params,
    tspan = (0.0, 1.0);
    kwargs...,
)

    function f(u, p, t)
        @unpack κ₁, κ₂, θ₁, θ₂ = p

        du1 = κ₁ * (θ₁ - u[1])
        du2 = κ₂ * (θ₂ - u[2])

        @SVector [du1, du2]
    end

    function g(u, p, t)
        @unpack σ₁, σ₂ = p

        du1 = σ₁ * √u[1]
        du2 = σ₂ * √u[2]

        @SVector [du1, du2]
    end

    SDEProblem{false}(f, g, u0, tspan, params, kwargs...)
end
