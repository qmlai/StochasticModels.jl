using StochasticModels

#-------------------------------------------------------------------------------------------#
#                       1-D Constant Elasticity Variance Model                              #
#-------------------------------------------------------------------------------------------#

u0     = 1.2
tspan  = (0.0, 1.0)
params = (μ=1.0, σ=0.2, γ=0.2) 

cev = ConstantElasticityVarianceModel(u0, params, tspan)

solve(cev, SRIW1())