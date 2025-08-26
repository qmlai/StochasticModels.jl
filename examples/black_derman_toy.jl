using StochasticModels

#-------------------------------------------------------------------------------------------#
#                                 1-D Black Derman Toy Model                                #
#-------------------------------------------------------------------------------------------#

u0     = 1.2
tspan  = (0.0, 1.0)
params = (θ=t->0.20, σ=t->exp(-t)*0.2) 

bdt = BlackDermanToyModel(u0, params, tspan)

solve(bdt, SRIW1())