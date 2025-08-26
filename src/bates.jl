
function BatesModel(u0::Vector{<:Real}, params, tspan::Tuple{Float64,Float64} = (0.0, 1.0); kwargs...)

    function f!(du, u, p, t)
        @unpack μ, κ, θ = p

        du[1] = μ * u[1]
        du[2] = κ * (θ - u[2])
        return nothing
    end

    function g!(du, u, p, t)
        @unpack θ = p

        du[1] = √u[2] * u[1]
        du[2] = θ * √u[2]
        return nothing
    end

    bates_noise = CorrelatedWienerProcess!(params.Γ, tspan[1], zeros(2), zeros(2))

    bates = SDEProblem{true}(f!, g!, u0, tspan, params, noise = bates_noise, kwargs...)

    JumpProblem(bates)
end

function BatesModel(u0::SVector{2,<:Real}, params, tspan::Tuple{Float64,Float64} = (0.0, 1.0); kwargs...)

    function f(u, p, t)
        @unpack μ, κ, θ = p

        du1 = μ * u[1]
        du2 = κ * (θ - u[2])

        @SVector [du1, du2]
    end

    function g(u, p, t)
        @unpack θ = p

        du1 = √u[2] * u[1]
        du2 = θ * √u[2]

        @SVector [du1, du2]
    end

    z0 = @SVector zeros(2)
    w0 = @SVector zeros(2)

    bates_noise = CorrelatedWienerProcess!(params.Γ, tspan[1], z0, w0)

    bates = SDEProblem{false}(f, g, u0, tspan, params, noise = bates_noise, kwargs...)

    JumpProblem(bates)
end
