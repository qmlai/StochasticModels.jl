function BlackKarasinskiModel(u0::R, params, tspan = (0.0, 1.0); kwargs...) where {R<:Real}

    function f(u, p, t)
        @unpack θ, ϕ = p

        θ(t) - ϕ(t) * log(u)
    end

    function g(u, p, t)
        @unpack σ = p

        σ(t)
    end

    SDEProblem{false}(f, g, u0, tspan, params, kwargs...)
end

