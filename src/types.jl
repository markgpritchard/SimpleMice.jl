
abstract type AbstractImputedData end 

struct ImputedNonMissingData{N, T} <: AbstractImputedData 
    v           :: T
end

struct ImputedMissingBinData{N, T} <: AbstractImputedData
    v           :: MVector{N, T}
    truev       :: T
    falsev      :: T
end

struct ImputedMissingContData{N, T} <: AbstractImputedData
    v           :: MVector{N, T}
end

struct ImputedMissingOrderedData{N, T} <: AbstractImputedData
    v           :: SizedVector{N}
    levels      :: Vector{T}
end

struct ImputedMissingUnOrderedData{N, T} <: AbstractImputedData
    v           :: SizedVector{N}
    levels      :: Vector{T}
end

ImputedData{N, T} = Union{
    ImputedNonMissingData{N, T}, 
    ImputedMissingBinData{N, T},
    ImputedMissingContData{N, T},
    ImputedMissingOrderedData{N, T},
    ImputedMissingUnOrderedData{N, T}
}

ImputedMissingData{N, T} = Union{
    ImputedMissingBinData{N, T},
    ImputedMissingContData{N, T},
    ImputedMissingOrderedData{N, T},
    ImputedMissingUnOrderedData{N, T}
}

struct ImputedRegressionResult{T}
    cn          :: Vector{String}
    allcoefs    :: T
    allsterrors :: T 
    coefs       :: Vector{Float64}
    vtotals     :: Vector{Float64}
    t           :: Vector{Float64}
    p           :: Vector{Float64}
    ci          :: Vector{Tuple{Float64, Float64}}
    lmdf        :: DataFrame 
end
