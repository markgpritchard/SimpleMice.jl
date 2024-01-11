
for f ∈ [ :/, :*, :+, :- ]
    @eval begin
        function $f(a::ImputedNonMissingData{N, T}, b::ImputedNonMissingData{N, S}) where {N, T, S} 
            val = $f(a.v, b.v)
            return ImputedNonMissingData{N, typeof(val)}(val)
        end
  
        function $f(a::ImputedNonMissingData{N, T}, b::ImputedMissingContData{N, S}) where {N, T, S} 
            val = MVector{N}([ $f.(a.v, b.v[i]) for i ∈ 1:N ]) 
            return ImputedMissingContData(val)
        end
        
        function $f(a::ImputedMissingContData{N, T}, b::ImputedNonMissingData{N, S}) where {N, T, S} 
            val = MVector{N}([ $f.(a.v[i], b.v) for i ∈ 1:N ]) 
            return ImputedMissingContData(val)
        end
        
        function $f(a::ImputedMissingContData{N, T}, b::ImputedMissingContData{N, S}) where {N, T, S} 
            val = MVector{N}([ $f.(a.v[i], b.v[i]) for i ∈ 1:N ]) 
            return ImputedMissingContData(val)
        end

        function $f(a::ImputedNonMissingData{N, T}, b::Number) where {N, T} 
            val = $f(a.v, b)
            return ImputedNonMissingData{N, typeof(val)}(val)
        end
        function $f(a::ImputedMissingContData{N, T}, b::Number) where {N, T} 
            val = MVector{N}([ $f.(a.v[i], b) for i ∈ 1:N ]) 
            return ImputedMissingContData(val)
        end
        function $f(a::Number, b::ImputedNonMissingData{N, T}) where {N, T} 
            val = MVector{N}([ $f.(a, b.v) for i ∈ 1:N ]) 
            return ImputedMissingContData(val)
        end
        function $f(a::Number, b::ImputedMissingContData{N, T}) where {N, T} 
            val = MVector{N}([ $f.(a, b.v[i]) for i ∈ 1:N ]) 
            return ImputedMissingContData(val)
        end
    end 
end 

==(a::ImputedMissingContData{N, S}, b::ImputedMissingContData{N, T}) where {N, S, T} = a.v == b.v
