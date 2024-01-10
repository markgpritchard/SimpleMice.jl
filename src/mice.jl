
function mice(df, args...; kwargs...) 
    newdf = deepcopy(df)
    mice!(newdf, vars, N, M)
    return newdf
end 

function mice!(df, vars, N::Int, M::Int; imputevars = vars, imputeboolvars = Symbol[ ])
    # find all vars that are in imputevars or imputeboolvars
    iv = vars[findall(x -> x ∈ imputevars || x ∈ imputeboolvars, vars)]
    selectinitialvalues!(df, N; imputeboolvars, iv)
    imputevalues!(df, vars, N, M; imputevars = iv, imputeboolvars, iv)
end

mice!(df, N::Int, M::Int; kwargs...) = mice!(df, Symbol.(names(df)), N::Int, M::Int; kwargs...)

function selectinitialvalues!(df::DataFrame, N; imputeboolvars, iv)
    for v ∈ iv 
        originalv = getproperty(df, v)
        select!(df, Not(v))
        if v ∈ imputeboolvars insertcols!(df, v => selectinitialvaluesbool(originalv, N))
        else                  insertcols!(df, v => selectinitialvalues(originalv, N))
        end
    end
end

function selectinitialvalues(v::Vector{<:Union{Missing, T}}, N) where T
    nonmissingv = collect(skipmissing(v))
    return ImputedData{N, T}[ ismissing(a) ? selectinitialvalue(nonmissingv, N) : ImputedNonMissingData{N, T}(a) 
        for a ∈ v ]
end

selectinitialvalue(nonmissingv, N) = ImputedMissingData(MVector{N}([ sample(nonmissingv) for _ ∈ 1:N ]))

function selectinitialvaluesbool(v::Vector{<:Union{Missing, T}}, N) where T 
    nonmissingv = collect(skipmissing(v))
    truev = maximum(nonmissingv)
    falsev = minimum(nonmissingv)
    return ImputedData{N, T}[ 
        ismissing(a) ? selectinitialvaluebool(nonmissingv, N, truev, falsev) : ImputedNonMissingData{N, T}(a) 
        for a ∈ v ]
end

function selectinitialvaluebool(nonmissingv, N, truev, falsev) 
    initialvs = MVector{N}([ sample(nonmissingv) for _ ∈ 1:N ])
    initialbools = MVector{N}([ v == truev for v ∈ initialvs ])
    initialprs = MVector{N}([ v ? 1. : .0 for v ∈ initialbools ])
    return ImputedMissingBoolData(initialbools, initialprs, truev, falsev)
end

function imputevalues!(df::DataFrame, vars, N, M; imputeboolvars, iv)
    for i ∈ 1:N 
        for _ ∈ 1:M
            for v ∈ vars 
                y = imputationvector(df, v, i) 
                X = imputationmatrix(df, vars, v, i) 
                newvalues = predict(lm(X, y))
                imputedvalues!(getproperty(df, v), newvalues, i)
            end
        end
    end
end

function imputedvalues!(dfvector, newvalues, i)
    for j ∈ eachindex(dfvector) 
        imputedvalue!(dfvector, dfvector[j], newvalues[j], i, j)
    end
end

imputedvalue!(dfvector, dfvalue::ImputedNonMissingData, newvalue, i, j) = nothing 

imputedvalue!(dfvector, dfvalue::ImputedMissingData, newvalue, i, j) = dfvector[j].v[i] = newvalue

imputationvector(df, v, i) = [ getvalue(a, i) for a ∈ getproperty(df, v) ]

function imputationmatrix(df, vars, v, i) 
    # may need to relax type of mat later
    mat = Matrix{Float64}(undef, size(df, 1), length(vars) - 1)
    imputationmatrix!(mat, df, vars, v, i) 
    return mat
end

function imputationmatrix!(mat, df, vars, v, i) 
    col = 1
    for var ∈ vars 
        var == v && continue 
        mat[:, col] = [ getvalue(a, i) for a ∈ getproperty(df, var) ]
        col += 1
    end
end
