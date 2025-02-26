# `MiceValue` is used to hold a series of imputed values that can be manipulated 
# arithmetically or combined according to Rubin's rules

abstract type AbstractMiceValue{Ni} end  # not exported

@auto_hash_equals struct MiceValue{Ni, T} <: AbstractMiceValue{Ni}  # not exported
    x                   :: MVector{Ni, T} 
end

function MiceValue(v::AbstractVector) 
    M = length(v)
    return MiceValue(M, v) 
end

function MiceValue(M::Integer, v::AbstractVector) 
    return MiceValue(MVector{M}(v)) 
end

function isapprox(a::AbstractMiceValue, b::AbstractMiceValue; kwargs...)
    return isapprox(a.x, b.x; kwargs...)
end

for f ∈ [ :^, :/, :*, :+, :- ]
    eval(quote
        function $f(a::AbstractMiceValue, b::AbstractMiceValue)
            v = @. $f(a.x, b.x)
            return MiceValue(v)
        end

        function $f(a::AbstractMiceValue, b::Number)
            v = @. $f(a.x, b)
            return MiceValue(v)
        end
        
        function $f(a::Number, b::AbstractMiceValue)
            v = @. $f(a, b.x)
            return MiceValue(v)
        end
    end)
end

for f ∈ [ :sin, :cos, :tan, :log, :exp ]
    eval(quote
        function $f(a::AbstractMiceValue)
            v = @. $f(a.x)
            return MiceValue(v)
        end
    end)
end
