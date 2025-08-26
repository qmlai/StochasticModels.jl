function FongVasicekModel(u0::Vector{<:Real}, params, tspan = (0.0, 1.0); kwargs...)

    function f!(du, u, p, t)
        @unpack κ₁, κ₂, θ₁, θ₂ = p

        du[1] = κ₁ * (θ₁ - u[1])
        du[2] = κ₂ * (θ₂ - u[2])
        return nothing
    end

    function g!(du, u, p, t)
        @unpack η, ρ = p

        du[1, 1] = √u[2]
        du[1, 2] = 0.0
        du[2, 1] = η * ρ * √u[2]
        du[2, 2] = η * ρ * √u[2] * √(1 - ρ^2)
        return nothing
    end

    SDEProblem{true}(
        f!,
        g!,
        u0,
        tspan,
        params,
        noise_rate_prototype = zeros(2, 2),
        kwargs...,
    )
end

function FongVasicekModel(u0::SVector{2,<:Real}, params, tspan = (0.0, 1.0); kwargs...)

    function f(u, p, t)
        @unpack κ₁, κ₂, θ₁, θ₂ = p

        du1 = κ₁ * (θ₁ - u[1])
        du2 = κ₂ * (θ₂ - u[2])

        @SVector [du1, du2]
    end

    function g(u, p, t)
        @unpack η, ρ = p

        du11 = √u[2]
        du12 = 0.0
        du21 = η * ρ * √u[2]
        du22 = η * ρ * √u[2] * √(1 - ρ^2)

        @SMatrix [du11 du12; du21 du22]
    end

    noise_prototype = @SMatrix zeros(2, 2)

    SDEProblem{false}(
        f,
        g,
        u0,
        tspan,
        params,
        noise_rate_prototype = noise_prototype,
        kwargs...,
    )
end
