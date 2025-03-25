# contains functions to create imputed data set

function initializemice(Ni::Integer, args...; kwargs...)
    return initializemice(default_rng(), Ni, args...; kwargs...)
end

function initializemice(rng::AbstractRNG, Ni::Integer, args...; kwargs...)
    return initializemice(rng, Float64, Ni, args...; kwargs...)
end

function initializemice(T::DataType, Ni::Integer, args...; kwargs...)
    return initializemice(default_rng(), T, Ni, args...; kwargs...)
end

function initializemice(
    rng::AbstractRNG, T::DataType, Ni::Integer, v::AbstractVector; 
    kwargs...
)
    return _initializemicevector(rng, T, Ni, v; kwargs...)
end

function initializemice(
    rng::AbstractRNG, T::DataType, Ni::Integer, df::DataFrame; 
    kwargs...
)
    return _initializemicedataframe(rng, T, Ni, df; kwargs...)
end

function initializemice(
    rng::AbstractRNG, T::DataType, Ni::Integer, df::DataFrame, columns::Vector{Symbol}; 
    kwargs...
)
    return _initializemicedataframe(rng, T, Ni, df, columns; kwargs...)
end

function initializemice(
    rng::AbstractRNG, T::DataType, Ni::Integer, df::DataFrame, columns; 
    kwargs...
)
    return _initializemicedataframe(
        rng, T, Ni, df, _inputsymbolvector(df, columns); 
        kwargs...
    )
end

function _initializemicevector(rng, T, Ni, v; staticthresh=100)
    nmvector = collect(skipmissing(v))
    @assert length(nmvector) >= 1 "Must have at least 1 non-missing value"
    S = typeof(nmvector[1])
    Nm = sum(ismissing.(v))
    Np = Ni * Nm 
    if Np == 0 
        return v
    elseif Np < staticthresh 
        return _initializemicestatic(rng, Ni, Nm, Np, S, T, v, nmvector)
    else 
        return _initializemicenotstatic(rng, Ni, Nm, Np, S, T, v, nmvector)
    end
end

function _initializemicedataframe(rng, T, Ni, df; kwargs...)
    columns = Symbol.(names(df))
    return _initializemicedataframe(rng, T, Ni, df, columns; kwargs...)
end

function _initializemicedataframe(rng, T, Ni, df, columns; kwargs...)
    columnnames = Symbol.(names(df))
    columntypes = Vector{Type}(undef, length(columnnames))                 
    unchangedvectors::Dict{Symbol, Vector} = Dict()
    imputedmstaticvectors::Dict{Symbol, ImputedVectorMStatic} = Dict()
    imputedvectors::Dict{Symbol, ImputedVector} = Dict()
    size1 = size(df, 1)
    for (j, name) ∈ enumerate(columnnames) 
        if name ∈ columns
            vnew = _initializemicevector(rng, T, Ni, getproperty(df, name); kwargs...)
            _pushinitializemice!(
                unchangedvectors, imputedmstaticvectors, imputedvectors, name, vnew
            )
            columntypes[j] = typeof(vnew)
        else
            push!(unchangedvectors, name => getproperty(df, name))
            columntypes[j] = typeof(getproperty(df, name))
        end
    end
    return ImputedTable{Ni}(
        columnnames, 
        columntypes, 
        unchangedvectors, 
        imputedmstaticvectors, 
        imputedvectors, 
        size1
    )
end

function _pushinitializemice!(unchangedvectors, ::Any, ::Any, name, vnew::Vector{<:Number})
    push!(unchangedvectors, name => vnew)
end

function _pushinitializemice!(
    ::Any, imputedmstaticvectors, ::Any, name, vnew::ImputedVectorMStatic
)
    push!(imputedmstaticvectors, name => vnew)
end

function _pushinitializemice!(::Any, ::Any, imputedvectors, name, vnew::ImputedVector)
    push!(imputedvectors, name => vnew)
end

function _initializemicestatic(rng, Ni, Nm, Np, S, T, v, nmvector)
    return ImputedVectorMStatic{Ni, Nm, Np, S, T}(
        v,
        SVector{Nm, Int}(findall(ismissing, v)),
        MMatrix{Nm, Ni, T, Np}([ T(sample(rng, nmvector)) for _ ∈ 1:Nm, _ ∈ 1:Ni ])
    )
end

function _initializemicenotstatic(rng, Ni, Nm, Np, S, T, v, nmvector)
    return ImputedVector{Ni, Nm, Np, S, T}(
        v,
        findall(ismissing, v),
        [ T(sample(rng, nmvector)) for _ ∈ 1:Nm, _ ∈ 1:Ni ]
    )
end

function linearupdatemicevalues!(table::ImputedTableView, y::Symbol, x::Vector{Symbol})
    _linearupdatemicevalues_tableview!(table, y, x)
end

function linearupdatemicevalues!(table::ImputedTableView, y, x)
    _linearupdatemicevalues_tableview!(
        table, _inputsymbol(table, y), _inputsymbolvector(table, x)
    )
end

function linearupdatemicevalues!(
    table::ImputedTable, 
    iteratevars::Vector{Symbol}, 
    includevars::Vector{Symbol}, 
    iterations;
    kwargs...
) 
    _linearupdatemicevalues_wholetable!(
        table, iteratevars, includevars, iterations; 
        kwargs...
    )
end

function linearupdatemicevalues!(
    table::ImputedTable, iteratevars, includevars, iterations;
    kwargs...
) 
    _linearupdatemicevalues_wholetable!(
        table, 
        _inputsymbolvector(table, iteratevars), 
        _inputsymbolvector(table, includevars),
        iterations;
        kwargs...
    )
end

function _linearupdatemicevalues_tableview!(tableview, y, x)
    A = imputedtableviewmatrix(tableview, x)
    b = getcolumn(tableview, y)
    prob = LinearProblem(A, b)
    sol = solve(prob, LinearSolve.KrylovJL_LSMR())
    preds = A * sol
    for (i, j) ∈ enumerate(getproperty(tableview.originaltable, y).missingindex)
        getproperty(tableview.originaltable, y).imputedvalues[i, tableview.index] = preds[j]
    end
end

function _linearupdatemicevalues_wholetable!(
    table, iteratevars, includevars, iterations;
    multithread=true
) 
    if multithread 
        _linearupdatemicevalues_wholetable_multithreads!(
            table, iteratevars, includevars, iterations
        )
    else
        _linearupdatemicevalues_wholetable_nomultithreads!(
            table, iteratevars, includevars, iterations
        )
    end
end

function _linearupdatemicevalues_wholetable_nomultithreads!(
    table::ImputedTable{Ni}, iteratevars, includevars, iterations
) where Ni
    for i ∈ 1:Ni 
        _linearupdatemicevaluesimputeone!(table, iteratevars, includevars, iterations, i) 
    end
end

function _linearupdatemicevalues_wholetable_multithreads!(
    table::ImputedTable{Ni}, iteratevars, includevars, iterations
) where Ni
    @threads for i ∈ 1:Ni 
        _linearupdatemicevaluesimputeone!(table, iteratevars, includevars, iterations, i) 
    end
end

function _linearupdatemicevaluesimputeone!(table, iteratevars, includevars, iterations, i) 
    tableview = imputedtableview(table, i)
    for _ ∈ 1:iterations
        _linearupdatemicevaluesiterate!(tableview, iteratevars, includevars) 
    end
end

function _linearupdatemicevaluesiterate!(tableview, iteratevars, includevars) 
    for v ∈ iteratevars 
        _linearupdatemicevaluesiteration!(tableview, iteratevars, includevars, v)
    end
end

function _linearupdatemicevaluesiteration!(tableview, iteratevars, includevars, v)
    x = [ iteratevars[findall(z -> z != v, iteratevars)]; includevars ]
    _linearupdatemicevalues_tableview!(tableview, v, x)
end

function impute(Ni::Integer, df::DataFrame, args...; kwargs...)
    return impute(default_rng(), Ni, df, args...; kwargs...)
end

function impute(rng::AbstractRNG, Ni::Integer, df::DataFrame, args...; kwargs...)
    return impute(rng, Float64, Ni, df, args...; kwargs...)
end

function impute(T::DataType, Ni::Integer, df::DataFrame, args...; kwargs...)
    return impute(default_rng(), T, Ni, df, args...; kwargs...)
end

function impute(
    rng::AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df::DataFrame, 
    iterations::Integer;
    kwargs...
)
    iteratevars = Symbol.(names(df))
    return impute(rng, T, Ni, df, iteratevars, iterations; kwargs...)
end

function impute(
    rng::AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df::DataFrame, 
    iteratevars,
    iterations::Integer;
    kwargs...
)
    includevars = Symbol[ ]
    return impute(rng, T, Ni, df, iteratevars, includevars, iterations; kwargs...)
end

function impute(
    rng::AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df::DataFrame, 
    iteratevars,
    includevars,
    iterations::Integer;
    kwargs...
)
    return _impute(
        rng, 
        T, 
        Ni, 
        df, 
        _inputsymbolvector(df, iteratevars),
        _inputsymbolvector(df, includevars),
        iterations;
        kwargs...
    )
end

function impute(
    rng::AbstractRNG, 
    T::DataType, 
    Ni::Integer, 
    df::DataFrame, 
    iteratevars::Vector{Symbol}, 
    includevars::Vector{Symbol}, 
    iterations::Integer; 
    kwargs...
)
    return _impute(rng, T, Ni, df, iteratevars, includevars, iterations; kwargs...) 
end

function _impute(
    rng, T, Ni, df, iteratevars, includevars, iterations; 
    multithread=true, staticthresh=100
)
    table = _initializemicedataframe(
        rng, T, Ni, df, [ iteratevars; includevars ]; 
        staticthresh
    )
    _linearupdatemicevalues_wholetable!(
        table, iteratevars, includevars, iterations; 
        multithread
    )
    return table 
end

ktypeof(::AbstractImputedVectorView{S}) where S = Vector{S}
