using StochasticModels

#-------------------------------------------------------------------------------------------#
#                                  1-D Affine Model                                         #
#-------------------------------------------------------------------------------------------#

x0 = 1.
tspan = (0.0, 1.0)

κ(t) = 1.
θ(t) = 1.
σ(t) = 1.
α(t) = 1.
β(t) = 0.0

params = (κ = κ, 
          θ = θ, 
          σ = σ, 
          α = α, 
          β = β)

aff = AffineModel(x0, params, tspan)
solve(aff; alg=EM(), dt=1/252)
s = solve(EnsembleProblem(aff), alg=SRIW1(), trajectories=100)

using Plots

plot(s)

#-------------------------------------------------------------------------------------------#
#                               N-D Diagonal Affine Model                                   #
#-------------------------------------------------------------------------------------------#

x0    = [2., 20.]
tspan = (0.0, 1.0)

κ(u, t) = u .= ones(2, 2)
θ(u, t) = u .= ones(2)

σ(u, t) = u .= ones(2)
α(u, t) = u .= ones(2)
β(u, t) = u .= zeros(2, 2)

params = (κ = κ, 
          θ = θ, 
          σ = σ, 
          α = α, 
          β = β)

aff = AffineModel(x0, params, :Diagonal, tspan)

solve(aff; alg=EM(), dt=1/252)

#-------------------------------------------------------------------------------------------#
#                             N-D Non Diagonal Affine Model                                 #
#-------------------------------------------------------------------------------------------#

x0    = [2., 20.]
tspan = (0.0, 1.0)

κ(u, t) = u .= ones(2, 2)
θ(u, t) = u .= ones(2)

σ(u, t) = u .= ones(2, 2)
α(u, t) = u .= ones(2)
β(u, t) = u .= zeros(2, 2)

params = (κ = κ, 
          θ = θ, 
          σ = σ, 
          α = α, 
          β = β)

aff = AffineModel(x0, params, :NonDiagonal, tspan)

solve(aff; alg=EM(), dt=1/252)
