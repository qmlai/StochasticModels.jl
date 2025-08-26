function ThreeTwoModel(u0, params, tspan = (0.0, 1.0); kwargs...)
    function f(u, p, t)
        @unpack ω, θ = p

        u * (ω - θ * u)
    end

    function g(u, p, t)
        @unpack ξ = p

        ξ * u^(3 / 2)
    end

    SDEProblem{false}(f, g, u0, tspan, params)
end
