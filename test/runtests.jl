using Test

@time begin
    @time @testset "Affine SDE limiting cases" begin include("affine_sde.jl") end
end