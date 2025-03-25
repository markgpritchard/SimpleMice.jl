# Functions to combine imputed values. Soure for Rubin's rules:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2727536/

function sum(v::AbstractImputedVector{Ni, S, T}) where {Ni, S, T}
    Z = typeof(one(S) + one(T))
    return SVector{Ni, Z}([ sum(imputedvectorview(v, i)) for i ∈ 1:Ni ])
end

function elementmean(v::AbstractImputedVector{Ni, S, T}) where {Ni, S, T}
    Z = typeof(one(S) + one(T) / 1)
    return SVector{Ni, Z}([ mean(imputedvectorview(v, i)) for i ∈ 1:Ni ])
end

function mean(v::AbstractImputedVector{Ni, S, T}) where {Ni, S, T}
    return sum(elementmean(v)) / Ni
end

function elementvar(v::AbstractImputedVector{Ni, S, T}; corrected=true) where {Ni, S, T}
    Z = typeof(one(S) + one(T) / 1)
    return SVector{Ni, Z}([ var(imputedvectorview(v, i); corrected) for i ∈ 1:Ni ])
end

function var(v::AbstractImputedVector; corrected=true, mean=nothing)
    return _var(v, corrected, mean)
end

function _var(v::AbstractImputedVector{Ni, S, T}, corrected, ::Nothing) where {Ni, S, T}
    means = elementmean(v)
    vars = elementvar(v; corrected)
    return rubinsvar(Ni, means, vars)
end

function _var(v::AbstractImputedVector, corrected, ::Any) 
    _varwarnonce()
    return _var(v, corrected, nothing)
end

_varwarncounter::Int = 0 

function _varwarnonce()
    if _varwarncounter == 0 
        @warn """
        Function `var(v::AbstractImputedVector)` does not use the keyword `mean`.
        This message is displayed only once.
        """
    end 
    global _varwarncounter += 1 
end

function rubinsvar(Ni, means, vars)
    ubar = mean(vars)
    qbar = mean(means) 
    qsquarediff = @. (means - qbar)^2
    b = sum(qsquarediff) / (Ni - 1)
    v = ubar + (1 + 1 / Ni) * b
    return v
end
