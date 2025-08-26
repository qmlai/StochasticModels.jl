
function GARCHModel(u0, params, tspan = (0.0, 1.0); kwargs...)
    function f(u, p, t)
        @unpack θ, ω = p

        θ * (ω - u)
    end

    function g(u, p, t)
        @unpack ξ = p

        ξ * u
    end
    
    SDEProblem{false}(f, g, u0, tspan, params)
end

