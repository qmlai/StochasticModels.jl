using StochasticModels, Test

#-------------------------------------------------------------------------------------------#
# Limiting cases of the 1-D and N-D affine models which are collapsed into simpler StochasticModels #
#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#                               1-D Affine Model vs Hull-White Model                        #
#-------------------------------------------------------------------------------------------#

let 
    x0    = 1.
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
    hwm = HullWhiteModel(x0, params, tspan)
    
    seed = rand(1:1000000)
    
    as = solve(aff, SRIW1(); seed=seed)
    hs = solve(hwm, SRIW1(); seed=seed)

    @test as.u ≈ hs.u
end

#-------------------------------------------------------------------------------------------#
#                         1-D Affine Model vs Cox Ingersoll Model                           #
#-------------------------------------------------------------------------------------------#

let 
    x0    = 1.
    tspan = (0.0, 1.0)

    κ(t) = 1.
    θ(t) = 1.
    σ(t) = 1.
    α(t) = 0.0
    β(t) = 1.0

    params = (κ = κ, 
              θ = θ, 
              σ = σ, 
              α = α, 
              β = β)

    params1 = (κ = 1., 
               θ = 1., 
               σ = 1.)

    aff = AffineModel(x0, params, tspan)
    cir = CoxIngersollRossModel(x0, params1, tspan)

    seed = rand(1:1000000)

    as = solve(aff, SRIW1(); seed=seed)
    cs = solve(cir, SRIW1(); seed=seed)

    @test as.u ≈ cs.u
end

#-------------------------------------------------------------------------------------------#
#                            1-D Affine Model vs Vasicek Model                              #
#-------------------------------------------------------------------------------------------#

let 
    x0    = 1.
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

    params1 = (κ = 1., 
               θ = 1., 
               σ = 1.)

    aff = AffineModel(x0, params, tspan)
    vsk = VasicekModel(x0, params1, tspan)

    seed = rand(1:1000000)

    as = solve(aff, SRIW1(); seed=seed)
    vs = solve(vsk, SRIW1(); seed=seed)

    @test as.u ≈ vs.u
end

#-------------------------------------------------------------------------------------------#
#                         N-D Affine Model vs Hull White Model                              #
#-------------------------------------------------------------------------------------------#

let 
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

    params1 = (κ = κ, 
               θ = θ, 
               σ = σ)

    aff = AffineModel(x0, params, :NonDiagonal, tspan)
    hwm = HullWhiteModel(x0, params1, :NonDiagonal, tspan)
    
    seed = rand(1:1000000)
    
    as = solve(aff, SRA1(); seed=seed)
    hs = solve(hwm, SRA1(); seed=seed)

    @test as.u ≈ hs.u
end

#-------------------------------------------------------------------------------------------#
#                       N-D Affine Model vs Cox Ingersoll Ross Model                        #
#-------------------------------------------------------------------------------------------#

let 
    x0    = [1.2, 1.]
    tspan = (0.0, 1.0)

    κ(u, t) = u .= 0.1 * ones(2, 2)
    θ(u, t) = u .= 0.1 * ones(2)

    σ(u, t) = u .= 0.1 * ones(2)
    α(u, t) = u .= zeros(2)
    β(u, t) = u .= Diagonal(ones(2, 2))

    params = (κ = κ, 
              θ = θ, 
              σ = σ, 
              α = α, 
              β = β)

    params1 = (κ = 0.1 * ones(2, 2), 
               θ = 0.1 * ones(2), 
               σ = 0.1 * ones(2))

    aff = AffineModel(x0, params, :Diagonal, tspan)
    cir = CoxIngersollRossModel(x0, params1, tspan)
    
    seed = rand(1:1000000)
    
    as = solve(aff, SRA1(); seed=seed)
    cs = solve(cir, SRA1(); seed=seed)

    @test as.u ≈ cs.u
end

#-------------------------------------------------------------------------------------------#
#                           N-D Affine Model vs Vasicek Model                               #
#-------------------------------------------------------------------------------------------#

let 
    x0    = [1., 1.]
    tspan = (0.0, 1.0)

    κ(u, t) = u .= 0.1 * ones(2, 2)
    θ(u, t) = u .= 0.1 * ones(2)

    σ(u, t) = u .= 0.1 * ones(2, 2)
    α(u, t) = u .= ones(2)
    β(u, t) = u .= zeros(2, 2)

    params = (κ = κ, 
              θ = θ, 
              σ = σ, 
              α = α, 
              β = β)

    params1 = (κ = 0.1 * ones(2, 2), 
               θ = 0.1 * ones(2), 
               σ = 0.1 * ones(2, 2))

    aff = AffineModel(x0, params, :NonDiagonal, tspan)
    vsk = VasicekModel(x0, params1, tspan)
    
    seed = rand(1:1000000)
    
    as = solve(aff, SRA1(); seed=seed)
    vs = solve(vsk, SRA1(); seed=seed)

    @test as.u ≈ vs.u
end

#-------------------------------------------------------------------------------------------#
#                       N-D Affine Model vs Longstaff-Schwartz Model                        #
#-------------------------------------------------------------------------------------------#

let 
    x0    = [1., 1.]
    tspan = (0.0, 1.0)

    κ(u, t) = u .= 0.1 * Diagonal(ones(2, 2))
    θ(u, t) = u .= 0.1 * ones(2)

    σ(u, t) = u .= 0.1 * ones(2)
    α(u, t) = u .= zeros(2)
    β(u, t) = u .= Diagonal(ones(2, 2))

    params = (κ = κ, 
              θ = θ, 
              σ = σ, 
              α = α, 
              β = β)

    params1 = (κ₁ = 0.1,
               κ₂ = 0.1, 
               θ₁ = 0.1, 
               θ₂ = 0.1,
               σ₁ = 0.1, 
               σ₂ = 0.1)

    aff = AffineModel(x0, params, :Diagonal, tspan)
    lst = LongstaffSchwartzModel(x0, params1, tspan)
   
    seed = rand(1:1000000)
    
    as = solve(aff; alg=SRIW1(), seed=seed)
    ls = solve(lst; alg=SRIW1(), seed=seed)

    @test as.u ≈ ls.u
end

