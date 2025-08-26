function ConstantElasticityVarianceModel(u0, params, tspan = (0.0, 1.0); kwargs...)
    function f(u, p, t)
        @unpack μ = p
        μ * u
    end

    function g(u, p, t)
        @unpack σ, γ = p

        σ * u^γ
    end

    SDEProblem{false}(f, g, u0, tspan, params, kwargs...)
end
