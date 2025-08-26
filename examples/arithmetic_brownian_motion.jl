using StochasticModels, BenchmarkTools

#-------------------------------------------------------------------------------------------#
#                        Scalar Arithmetic Brownian Motion Model                             #
#-------------------------------------------------------------------------------------------#

u0    = 1.0
p     = (μ=1., σ=1.)
tspan = (0.0, 1.0)

gbm = ArithmeticBrownianMotionModel(u0, p; tspan=tspan)

@btime mean(solve(gbm; alg=SRIW1(), trajectories=1E5, saveat=tspan[end]).u)

#-------------------------------------------------------------------------------------------#
#                        Diagonal Arithmetic Brownian Motion Model                           #
#-------------------------------------------------------------------------------------------#

u01   = [1.0, 1.2]
tspan = (0.0, 11.0)
p1    = (μ=[1., 1.], σ=[1., 1.])

gbm1 = ArithmeticBrownianMotionModel(u01, p1; tspan)

solve(gbm1; alg=SRA1())

#-------------------------------------------------------------------------------------------#
#                        Non-Diagonal Arithmetic Brownian Motion Model                       #
#-------------------------------------------------------------------------------------------#

u02   = [1.0, 12.0]
tspan = (0.0, 1.20)
p2    = (μ=[1., 1.], 
         σ=[1. 1.; 1. 1.])

gbm2 = ArithmeticBrownianMotionModel(u02, p2; tspan)

solve(gbm2; alg=SRA1())

#-------------------------------------------------------------------------------------------#
#                 Non-Diagonal correlated Arithmetic Brownian Motion Model                   #
#-------------------------------------------------------------------------------------------#

u03   = [1.0, 1.0]
tspan = (0.0, 1.30)
p3    = (μ=[1., 1.], 
         σ=[1. 1.; 1. 1.], 
         Γ=[1. 0.9; 0.9 1.])

gbm3 = ArithmeticBrownianMotionModel(u03, p3; tspan)

solve(gbm3; alg=SRA1())


