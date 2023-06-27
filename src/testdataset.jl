
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulate missing completely at random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function mcar(df, args...)
    ndf = deepcopy(df)
    mcar!(ndf, args...)
    return ndf 
end 

function mcar!(df::DataFrame, vars::Vector, p::Number)
    ps = [ p for _ ∈ eachindex(vars) ]
    mcar!(df, vars, ps) 
end 

function mcar!(df::DataFrame, vars::Vector, p::Vector{<:Number})
    @assert length(vars) == length(p)
    for (v, prob) ∈ zip(vars, p) mcar!(df, v, prob) end 
end 

function mcar!(df::DataFrame, var, p::Number)
    for i in axes(df, 1)
        if rand() < p df[i, var] = missing end 
    end 
end 
