function BlackDermanToyModel(u0::R, params, tspan = (0.0, 1.0); kwargs...) where {R<:Real}
    
    σ = params.σ
    dσdt(x) = ForwardDiff.derivative(σ, x)
    params = (; params..., δσδt = dσdt)

    function f(u, p, t)
        @unpack θ, σ, δσδt = p

        θ(t) - δσδt(t) / σ(t) * log(u)
    end

    function g(u, p, t)
        @unpack σ = p

        σ(t)
    end

    SDEProblem{false}(f, g, u0, tspan, params, kwargs...)
end

