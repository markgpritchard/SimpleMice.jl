
module SimpleMice

using DataFrames, Econometrics, GLM, Random, StaticArrays, StatsBase
using CategoricalArrays: CategoricalPool, CategoricalValue, CategoricalVector 
using HypothesisTests: confint, OneSampleTTest, pvalue
using StatsModels: TableRegressionModel
import Base: ==, /, *, +, -
import StatsBase: var

include("types.jl")
include("basefunctions.jl")
include("mice.jl")
include("statsfunctions.jl")

export 
    # types.jl
    #AbstractImputedData, ImputedData, ImputedMissingBoolData, ImputedMissingData, 
    #ImputedNonMissingData, ImputedRegressionResult,
    # basefunctions.jl
    ==, /, *, +, -,
    # mice.jl 
    mice, mice!,
    # statsfunctions.jl
    imputedlm, rubinsmean, rubinsvar, var




module TempEconometricsVariant 

# The function `fit!` from Econometrics needs bkr to be changed to qrr. Pull request 85 
# at Econometrics.jl should solve this. In the meantime, corrected code is included 
# here.

# Distributions, ForwardDiff, LinearAlgebra, Optim, Parameters and StatsFuns can 
# be removed from Project.toml once this code is removed

import Econometrics: fit!, Dual, EconometricModel, Hermitian, NominalResponse, obtain_Ω, 
    OrdinalResponse, TwiceDifferentiable

using CategoricalArrays, Distributions, ForwardDiff, LinearAlgebra, Optim, Parameters, 
    StatsBase, StatsFuns

export fit!

@views function fit!(obj::EconometricModel{<:NominalResponse})
	@unpack estimator, X, y, z, Z, wts = obj
	@unpack categories = estimator
    @assert isempty(z) && isempty(Z) "Nominal response models can only contain exogenous features"
    b = mapreduce(elem -> (eachindex(categories) .== elem)', vcat, y)
    F = qr(X, ColumnNorm())
    qrr = count(x -> abs(x) ≥ √eps(), diag(F.R))
    if qrr < size(F, 2)
        lin_ind = sort!(F.p[sortperm(F.p)[1:qrr]])
        X = convert(Matrix{Float64}, X[:,lin_ind])
        F = qr(X, ColumnNorm())
    else
        lin_ind = collect(1:size(F, 2))
    end
    Q = Matrix(F.Q)
    Q⊤ = transpose(Q)
    m, p = size(F)
    k = size(b, 2)
    η = zeros(m, k)
    μ, μ′, w = zero(η), zero(η), zero(η)
    β = zeros(p, k)
    ℓℓ = [Inf, Inf, Inf]
    max_iteration = 250
    tol = 1e-9
    converged = false
    for iteration ∈ 1:max_iteration
      converged && break
      for row ∈ 1:m
        μ[row,:] .= softmax(η[row,:])
      end
      μ′ .= max.(μ .* (one(Float64) .- μ), √eps())
      w .= wts .* μ′
      ℓℓ[1] = ℓℓ[2]
      ℓℓ[2] = sum(wts[idx[1]] * logpdf(Categorical(collect(μ[idx[1],:])), idx[2]) for idx in findall(b))
      if isone(iteration)
          ℓℓ₀ = ℓℓ[2]
      end
      η .+= (b .- μ) ./ μ′
      for idx ∈ 2:size(b, 2)
        C = cholesky!(Hermitian(Q⊤ * Diagonal(w[:,idx]) * Q)).factors
        β[:,idx] = LowerTriangular(transpose(C)) \ (Q⊤ * Diagonal(w[:,idx]) * η[:,idx])
        β[:,idx] = UpperTriangular(C) \ β[:,idx]
      end
      η .= Q * β
      ℓℓ[3] = abs(ℓℓ[2] - ℓℓ[1])
      converged = ℓℓ[3] < 1e-9 || iszero(ℓℓ[1])
    end
    converged || throw(ConvergenceException(max_iteration, ℓℓ[3], tol))
    β .= F \ η
    Ψ = Hermitian(inv(bunchkaufman!(obtain_Ω(X, μ, wts))))
    ŷ = X * β
    β = collect(vec(β)[size(X, 2) + 1:end])
	@pack! obj = X, y, β, Ψ, ŷ
	obj.vars = (obj.vars[1], obj.vars[2][lin_ind])
	obj
end
@views function fit!(obj::EconometricModel{<:OrdinalResponse})
	@unpack estimator, X, y, z, Z, wts = obj
	@unpack categories = estimator
    @assert isempty(z) && isempty(Z) "Ordinal response models can only contain exogenous features"
    F = qr(X, ColumnNorm())
    qrr = count(x -> abs(x) ≥ √eps(), diag(F.R))
    if qrr < size(F, 2)
        lin_ind = sort!(F.p[sortperm(F.p)[1:qrr]])
        X = convert(Matrix{Float64}, X[:,lin_ind])
    else
        lin_ind = collect(1:size(F, 2))
    end
    m, n = size(X)
    xs = 1:n
    ks = length(categories) - 1
    ζs = n + 1:(n + ks)
    δ₀ = mapreduce(x -> (1:ks .== x - 1)', vcat, y)
    δ₁ = mapreduce(x -> (1:ks .== x)', vcat, y)
    η = Vector{Union{Float64,Dual}}(undef, m)
    ζ = vcat(-Inf, similar(η, ks), Inf)
    y₁ = y .+ 1
    Y₀ = similar(η)
    Y₁ = similar(η)
    F₀ = similar(η)
    F₁ = similar(η)
    f₀ = similar(η)
    f₁ = similar(η)
    pr = similar(η)
    θ₁ = ones(ks)
    function f(β)
      η .= X * β[xs]
      ζ[2:end - 1] .= cumsum(vcat(β[n + 1], exp.(β[n + 2:end])))
      Y₀ .= max.(-100, ζ[y] - η)
      Y₁ .= min.(100, ζ[y₁] - η)
      F₀ .= cdf.(Logistic(), Y₀)
      F₁ .= cdf.(Logistic(), Y₁)
      pr .= F₁ .- F₀
      -sum(w * log(p) for (w, p) ∈ zip(wts, pr))
    end
    function g!(G, β)
      ζ[2:end - 1] .= cumsum(vcat(β[n + 1], exp.(β[n + 2:end])))
      η .= X * β[xs]
      Y₀ .= max.(-100, ζ[y] - η)
      Y₁ .= min.(100, ζ[y₁] - η)
      F₀ .= cdf.(Logistic(), Y₀)
      F₁ .= cdf.(Logistic(), Y₁)
      pr .= F₁ .- F₀
      f₀ .= pdf.(Logistic(), Y₀)
      f₁ .= pdf.(Logistic(), Y₁)
      G[xs] .= X' * (wts .* (f₁ .- f₀) ./ pr)
      g2 = - (δ₁ .* f₁ .- δ₀ .* f₀)' * (wts ./ pr)
      θ₁[2:end] .= exp.(β[ζs[2:end]])
      G[ζs] .= diagm((d => θ₁[1:ks - d] for d ∈ 0:ks - 1)...) * g2
    end
    β₀ = vcat(zero(xs), log.(1:ks))
    td = TwiceDifferentiable(f, g!, β₀, autodiff = :forwarddiff)
    β = optimize(td, β₀) |> minimizer
    H = hessian!(td, β)
    Ψ = inv(bunchkaufman!(Hermitian(H)))
    A = Matrix(1.0I, size(Ψ)...)
    A[ζs, ζs] .= diagm((d => θ₁[1:ks + d] for d ∈ 0:-1:1 - ks)...)
    Ψ = Hermitian(A * Ψ * A')
    ŷ = X * β[1:size(X, 2)]
    β[n + 1:end] .= cumsum(vcat(β[n + 1], exp.(β[n + 2:end])))
	@pack! obj = X, y, β, Ψ, ŷ
	obj.vars = (obj.vars[1], obj.vars[2][lin_ind .+ 1])
	obj
end

end # module TempEconometricsVariant

using .TempEconometricsVariant

end # module SimpleMice
