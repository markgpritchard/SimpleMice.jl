
for f ∈ [ :/, :*, :+, :- ]
    @eval begin
        function $f(a::ImputedNonMissingData{N, T}, b::ImputedNonMissingData{N, S}) where {N, T, S} 
            val = $f(a.v, b.v)
            return ImputedNonMissingData{N, typeof(val)}(val)
        end
  
        function $f(a::ImputedNonMissingData{N, T}, b::ImputedMissingData{N, S}) where {N, T, S} 
            val = MVector{N}([ $f.(a.v, b.v[i]) for i ∈ 1:N ]) 
            return ImputedMissingData(val)
        end
        
        function $f(a::ImputedMissingData{N, T}, b::ImputedNonMissingData{N, S}) where {N, T, S} 
            val = MVector{N}([ $f.(a.v[i], b.v) for i ∈ 1:N ]) 
            return ImputedMissingData(val)
        end
        
        function $f(a::ImputedMissingData{N, T}, b::ImputedMissingData{N, S}) where {N, T, S} 
            val = MVector{N}([ $f.(a.v[i], b.v[i]) for i ∈ 1:N ]) 
            return ImputedMissingData(val)
        end

        function $f(a::ImputedNonMissingData{N, T}, b::Number) where {N, T} 
            val = $f(a.v, b)
            return ImputedNonMissingData{N, typeof(val)}(val)
        end
        function $f(a::ImputedMissingData{N, T}, b::Number) where {N, T} 
            val = MVector{N}([ $f.(a.v[i], b) for i ∈ 1:N ]) 
            return ImputedMissingData(val)
        end
        function $f(a::Number, b::ImputedNonMissingData{N, T}) where {N, T} 
            val = MVector{N}([ $f.(a, b.v) for i ∈ 1:N ]) 
            return ImputedMissingData(val)
        end
        function $f(a::Number, b::ImputedMissingData{N, T}) where {N, T} 
            val = MVector{N}([ $f.(a, b.v[i]) for i ∈ 1:N ]) 
            return ImputedMissingData(val)
        end
    end 
end 

==(a::ImputedMissingData{N, T}, b::ImputedMissingData{N, T}) where {N, T} = a.v == b.v
