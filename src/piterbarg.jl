function PiterbargModel(u0::Array{<:Real}, parameters, tspan = (0.0, 1.0); kwargs...)

    params = (; parameters..., S₀ = u0[1], z₀ = u0[2])

    function f!(du, u, p, t)
        @unpack z₀, θ = p

        du[1] = 0.0
        du[2] = θ(t) * (z₀ - u[2])
        return nothing
    end

    function g!(du, u, p, t)
        @unpack S₀, σ, β, θ, γ = p

        du[1] = σ(t) * (β(t) * u[1] + (1.0 - β(t) * S₀)) * √u[2]
        du[2] = γ(t) * √u[2]
        return nothing
    end

    SDEProblem{true}(f!, g!, u0, tspan, params)
end

function PiterbargModel(u0::SVector{2,<:Real}, parameters, tspan = (0.0, 1.0); kwargs...)

    params = (; parameters..., S₀ = u0[1], z₀ = u0[2])

    function f(u, p, t)
        @unpack z₀, θ = p

        du₁ = 0.0
        du₂ = θ(t) * (z₀ - u[2])
        return @SVector [du₁, du₂]
    end

    function g(u, p, t)
        @unpack S₀, σ, β, θ, γ = p

        du₁ = σ(t) * (β(t) * u[1] + (1.0 - β(t) * S₀)) * √u[2]
        du₂ = γ(t) * √u[2]
        return @SVector [du₁, du₂]
    end

    SDEProblem{false}(f, g, u0, tspan, params)
end