
function mice(df, args...; kwargs...) 
    newdf = deepcopy(df)
    mice!(newdf, args...; kwargs...)
    return newdf
end 

function mice!(df, N::Int, M::Int; 
        binaryvars = Symbol[ ], continuousvars = Symbol[ ], nonmissingvars = Symbol[ ], 
        orderedcatvars = Symbol[ ], unorderedcatvars = Symbol[ ]
    )
    allvars = [ binaryvars; continuousvars; nonmissingvars; orderedcatvars; unorderedcatvars ]
    mice!(df, allvars, N, M; 
        binaryvars, continuousvars, nonmissingvars, orderedcatvars, unorderedcatvars)
end

function mice!(df, vars::Vector{<:Symbol}, N::Int, M::Int; 
        binaryvars = Symbol[ ], continuousvars = Symbol[ ], nonmissingvars = Symbol[ ], 
        orderedcatvars = Symbol[ ], unorderedcatvars = Symbol[ ]
    )
    selectinitialvalues!(df, N; binaryvars, continuousvars, orderedcatvars, unorderedcatvars)
    imputevalues!(df, vars, N, M; binaryvars, continuousvars, orderedcatvars, unorderedcatvars)
end

function selectinitialvalues!(df, N::Int; binaryvars, continuousvars, orderedcatvars, unorderedcatvars)
    for v ∈ binaryvars       selectinitialvalue!(df, selectinitialbinvals, v, N)          end
    for v ∈ continuousvars   selectinitialvalue!(df, selectinitialcontvals, v, N)         end
    for v ∈ orderedcatvars   selectinitialvalue!(df, selectinitialorderedcatvals, v, N)   end
    for v ∈ unorderedcatvars selectinitialvalue!(df, selectinitialunorderedcatvals, v, N) end    
end

function selectinitialvalue!(df, func, v, N)
    originalv = getproperty(df, v)
    select!(df, Not(v))
    insertcols!(df, v => func(originalv, N))
end

function selectinitialbinvals(originalv::Vector{<:Union{Missing, T}}, N) where T
    nonmissingv = collect(skipmissing(originalv))
    truev = maximum(nonmissingv)
    falsev = minimum(nonmissingv)
    return ImputedData{N, T}[ 
        ismissing(a) ? selectinitialbinval(nonmissingv, N, truev, falsev) : ImputedNonMissingData{N, T}(a) 
        for a ∈ originalv ]
end

function selectinitialbinval(nonmissingv, N, truev, falsev) 
    initialvs = MVector{N}([ sample(nonmissingv) for _ ∈ 1:N ])
    return ImputedMissingBinData(initialvs, truev, falsev)
end

function selectinitialcontvals(originalv::Vector{<:Union{Missing, T}}, N) where T
    nonmissingv = collect(skipmissing(originalv))
    return ImputedData{N, T}[ 
        ismissing(a) ? selectinitialcontval(nonmissingv, N) : ImputedNonMissingData{N, T}(a) 
        for a ∈ originalv ]
end

selectinitialcontval(nonmissingv, N) = ImputedMissingContData(MVector{N}([ sample(nonmissingv) for _ ∈ 1:N ]))

function selectinitialorderedcatvals(originalv::CategoricalVector{<:Union{Missing, T}}, N) where T
    nonmissingv = collect(skipmissing(originalv))
    lv = levels(originalv)
    return ImputedData{N, T}[ 
        ismissing(a) ? selectinitialorderedcatval(nonmissingv, N, lv) : ImputedNonMissingData{N, T}(a) 
        for a ∈ originalv ]
end

function selectinitialorderedcatvals(originalv::Vector{<:Union{Missing, T}}, N) where T
    catv = categorical(originalv; ordered = true)
    return selectinitialorderedcatvals(catv, N)
end

function selectinitialorderedcatval(nonmissingv, N, lv) 
    v = SizedVector{N}([ sample(nonmissingv) for _ ∈ 1:N ])
    return ImputedMissingOrderedData(v, lv)
end

function selectinitialunorderedcatvals(originalv::CategoricalVector{<:Union{Missing, T}}, N) where T
    nonmissingv = collect(skipmissing(originalv))
    lv = levels(originalv)
    return ImputedData{N, T}[ 
        ismissing(a) ? selectinitialunorderedcatval(nonmissingv, N, lv) : ImputedNonMissingData{N, T}(a) 
        for a ∈ originalv ]
end

function selectinitialunorderedcatvals(originalv::Vector{<:Union{Missing, T}}, N) where T
    catv = categorical(originalv; ordered = false)
    return selectinitialunorderedcatvals(catv, N)
end

function selectinitialunorderedcatval(nonmissingv, N, lv) 
    v = SizedVector{N}([ sample(nonmissingv) for _ ∈ 1:N ])
    return ImputedMissingUnOrderedData(v, lv)
end

function imputevalues!(data::DataFrame, vars, N, M; binaryvars, continuousvars, orderedcatvars, unorderedcatvars, imputeformulas = Dict{Symbol, FormulaTerm}())
    tdf = createimputeddf(data, vars; orderedcatvars, unorderedcatvars)
    imputevaluesformulae!(imputeformulas, vars)
    for i ∈ 1:N 
        for j ∈ 1:M
            for v ∈ vars 
                mutateimputeddf!(tdf, data, vars, i)
                fla = imputeformulas[v]
                if v ∈ binaryvars
                    try
                        newvalues = predict(glm(fla, tdf, Binomial(), LogitLink()))
                    catch e 
                        @warn "glm error $e on value $i and iteration $j of $v" 
                        continue
                    end
                elseif v ∈ continuousvars
                    try
                        newvalues = predict(lm(fla, tdf))
                    catch e 
                        @warn "lm error $eon value $i and iteration $j of $v"
                        continue
                    end
                elseif v ∈ orderedcatvars || v ∈ unorderedcatvars
                    try
                        newvalues = predict(fit(EconometricModel, fla, tdf))
                    catch e 
                        @warn "EconometricModel fit error $eon value $i and iteration $j of $v"
                        continue
                    end
                else 
                    newvalues = nothing
                end
                imputedvalues!(getproperty(data, v), newvalues, i)
            end
        end
    end
end

function imputevaluesformulae!(imputeformulas, vars)
    for v ∈ vars 
        if v ∉ keys(imputeformulas) 
            push!(imputeformulas, v => imputevaluesformula(v, vars))
        end
    end
end

function imputevaluesformula(v, vars)
    lhs = Term(v) 
    rhs = ( [ x == v ? ConstantTerm(1) : Term(x) for x ∈ vars ]..., )
    return FormulaTerm(lhs, rhs)
end

imputedvalues!(dfvector, newvalues::Nothing, i) = nothing

function imputedvalues!(dfvector, newvalues::Vector, i)
    for j ∈ eachindex(dfvector) 
        imputedvalue!(dfvector, dfvector[j], newvalues[j], i, j)
    end
end

function imputedvalues!(dfvector, newvalues::Matrix, i)
    for j ∈ eachindex(dfvector) 
        imputedvalue!(dfvector, dfvector[j], newvalues[j, :], i, j)
    end
end

imputedvalue!(dfvector, dfvalue::ImputedNonMissingData, newvalue, i, j) = nothing 

imputedvalue!(dfvector, dfvalue::ImputedMissingBinData, newvalue, i, j) = 
    dfvector[j].v[i] = rand() < newvalue ? dfvector[j].truev : dfvector[j].falsev

imputedvalue!(dfvector, dfvalue::ImputedMissingContData, newvalue, i, j) = dfvector[j].v[i] = newvalue

function imputedvalue!(dfvector, dfvalue::T, newvalue::Vector, i, j
    ) where T <: Union{ImputedMissingOrderedData, ImputedMissingUnOrderedData}
    threshs = cumsum(newvalue)
    k = findfirst(x -> x >= rand(), threshs)
    newval = CategoricalPool(dfvector[j].levels)[k]
    dfvector[j].v[i] = newval
end
