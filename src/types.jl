
abstract type AbstractImputedData end 

struct ImputedNonMissingData{N, T} <: AbstractImputedData 
    v       :: T
end

struct ImputedMissingData{N, T} <: AbstractImputedData
    v       :: MVector{N, T}
end

struct ImputedMissingBoolData{N, T} <: AbstractImputedData
    v       :: MVector{N, Bool}
    pr      :: MVector{N, Float64} 
    truev   :: T
    falsev  :: T
end

ImputedData{N, T} = Union{ImputedNonMissingData{N, T}, ImputedMissingData{N, T}}

struct ImputedRegressionResult{T}
    cn              :: Vector{String}
    allcoefs        :: T
    allsterrors     :: T 
    coefs           :: Vector{Float64}
    vtotals         :: Vector{Float64}
    t               :: Vector{Float64}
    p               :: Vector{Float64}
    ci              :: Vector{Tuple{Float64, Float64}}
    lmdf            :: DataFrame 
end
