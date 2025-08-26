function ChenModel(u0::Vector{<:Real}, params, tspan = (0.0, 1.0); kwargs...)
    
    function f!(du, u, p, t)
        @unpack κ, ν, ζ, μ, β, α, η = p

        du[1] = κ * (u[2] - u[1])
        du[2] = ν * (ζ - u[2])
        du[3] = μ * (β - u[3])
        return nothing
    end

    function g!(du, u, p, t)
        @unpack κ, ν, ζ, μ, β, α, η = p
        
        du[1] = √(u[1] * u[3])
        du[2] = α * √u[2]
        du[3] = η * √u[3]
        return nothing
    end

    SDEProblem{true}(f!, g!, u0, tspan, params, kwargs...)

end

function ChenModel(u0::SVector{3, <:Real}, params, tspan = (0.0, 1.0); kwargs...)
    
    function f(u, p, t)
        @unpack κ, ν, ζ, μ, β, α, η = p

        du1 = κ * (u[2] - u[1])
        du2 = ν * (ζ - u[2])
        du3 = μ * (β - u[3])

        @SVector [du1, du2, du3]

    end

    function g(u, p, t)
        @unpack κ, ν, ζ, μ, β, α, η = p
        
        du1 = √(u[1] * u[3])
        du2 = α * √u[2]
        du3 = η * √u[3]

        @SVector [du1, du2, du3]

    end

    SDEProblem{false}(f, g, u0, tspan, params, kwargs...)
    
end
