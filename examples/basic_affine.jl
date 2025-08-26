using StochasticModels

#-------------------------------------------------------------------------------------------#
#                                  1-D Affine Model                                         #
#-------------------------------------------------------------------------------------------#

x0 = 1.5
tspan = (0.0, 1.0)

κ(t) = 1.
θ(t) = 1.
Σ(t) = 0.5
α(t) = 1.
β(t) = 0.0

params = (κ = κ, 
          θ = θ, 
          Σ = Σ, 
          α = α, 
          β = β)

aff = BasicAffineJumpModel(x0, params, tspan)
sol = solve(EnsembleProblem(aff), SRIW1(), trajectories=10)

plot(sol)

#-------------------------------------------------------------------------------------------#
#                                  N-D Affine Model                                         #
#-------------------------------------------------------------------------------------------#

x0    = [2., 5.]
tspan = (0.0, 1.0)

κ(u, t) = u .= abs.(randn(2, 2))
θ(u, t) = u .= randn(2)

Σ(u, t) = u .= 0.1abs.(randn(2, 2))
α(u, t) = u .= ones(2)
β(u, t) = u .= zeros(2, 2)

params = (κ = κ, 
          θ = θ, 
          Σ = Σ, 
          α = α, 
          β = β)

aff = BasicAffineJumpModel(x0, params, tspan)
sol = solve(EnsembleProblem(aff), SRA(), trajectories=10)
sum = EnsembleSummary(sol)

plot(sum)
