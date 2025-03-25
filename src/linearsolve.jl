#

function linearsolveimputedvectorelements(
    A, b::AbstractImputedVector{Ni, S, T}; 
    alg=LinearSolve.KrylovJL_LSMR()
) where {Ni, S, T}
    output = zeros(size(A[1], 2), Ni)
    for i ∈ A.indexes
        prob = LinearProblem(A[i], imputedvectorview(b, i))
        sol = solve(prob, alg)
        output[:, i] .= sol
    end
    return output
end

function linearsolveimputedvector(A, b; kwargs...)
    elementsol = solveimputedvectorelements(A, b; kwargs...) 
    return [ mean(elementsol[i, :]) for i ∈ axes(elementsol, 1) ]
end
