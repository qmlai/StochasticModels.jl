using StochasticModels 

#-------------------------------------------------------------------------------------------#
#                                     IIP Heston Model                                      #
#-------------------------------------------------------------------------------------------#

u0    = [1., 0.3]
tspan = (0.0, 1.0)
p     = (μ=1., κ=0.2, θ=0.2, Γ=[1. 0.2; 0.2 1.])

hs = HestonModel(u0, p, tspan)

solve(hs, SRA1())

#-------------------------------------------------------------------------------------------#
#                                     OOP Heston Model                                      #
#-------------------------------------------------------------------------------------------#

u0    = @SVector [1., 0.3]
tspan = (0.0, 1.0)
p     = (μ=1., κ=0.2, θ=0.2, Γ=[1. 0.2; 0.2 1.])

hs  = HestonModel(u0, p, tspan)
gbm = GeometricBrownianMotionModel(u0, p; tspan)

sol = solve(EnsembleProblem(hs), SRA1(), trajectories=1000, saveat=0.0:0.1:1.0)
arr = Array(sol)

arrgbm = Array(solve(EnsembleProblem(gbm), SRA1(), trajectories=1000, saveat=0.0:0.01:1.0))
plot(arrgbm[end, :], seriestype=:scatterhist, bins=50)


using Plots

plot(sol)
plot(arr[2, end, :], seriestype=:scatterhist, bins=50)


u0    = 1.0
p     = (μ=1., σ=1.)
tspan = (0.0, 1.0)

