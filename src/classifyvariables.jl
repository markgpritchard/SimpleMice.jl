
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Turn lists of strings into lists of symbols 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

symbollist(list::Vector{Symbol}) = list 
symbollist(list::Vector{String}) = [ Symbol(v) for v ∈ list ]
symbollist(list::Nothing) = nothing


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Divide lists of variables according to type 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function classifyvars!(vars, binvars, contvars, noimputevars, df; printdropped)
    # turn any lists of strings into lists of symbols 
    bv = symbollist(binvars) 
    cv = symbollist(contvars) 
    niv = symbollist(noimputevars)
    # any variables that are already classified should not be reclassified by this function 
    initialclassifyvars!(vars, bv, cv, niv)
    # now for missing lists, classify variables 
    newnoimputevars = choosenoimputevars(df, vars, niv) 
    initialclassifyvars!(vars, newnoimputevars, niv)
    newbinvars = choosebinvars(df, vars, bv) 
    initialclassifyvars!(vars, newbinvars, bv)
    newcontvars = choosecontvars(df, vars, cv) 
    initialclassifyvars!(vars, newcontvars, cv)
    if printdropped
        if length(vars) > 0
            @warn """
            The following variables are not not included in imputation: $vars.
            If you wish these to be included, check the use of keywords binvars, 
            contvars and noimputevars.
            """
        end 
    end 
    return ( newbinvars, newcontvars, newnoimputevars )
end 

# remove those that are already placed into a category from the main vars list
function initialclassifyvars!(vars, bv, cv, niv)
    for v ∈ [ bv, cv, niv ] initialclassifyvars!(vars, v) end 
end 

initialclassifyvars!(vars, v::Nothing) = nothing
initialclassifyvars!(vars, v::Vector) = filter!(x -> x ∉ v, vars)

# version of the above function that only needs to run if the `vector` was previously 
# nothing. If variables already in a vector then on need to re-run initialclassifyvars
initialclassifyvars!(vars, v, oldv::Nothing) = initialclassifyvars!(vars, v)
initialclassifyvars!(vars, v, oldv::Vector) = nothing

function choosebinvars(df, vars, binvars::Nothing)
    newbinvars = Symbol[] 
    choosebinvars!(newbinvars, df, vars)
    return newbinvars 
end 

function choosebinvars(df, vars, binvars::Vector{T}) where T <: Symbol
    return binvars 
end 

function choosebinvars!(newbinvars, df, vars)
    for var ∈ vars choosebinvar!(newbinvars, df, var) end 
end 

function choosebinvar!(newbinvars, df, var)
    varvector = getproperty(df, var)
    choosebinvar!(newbinvars, df, var, varvector)
end 

function choosebinvar!(newbinvars, df, var, varvector::Vector{MissBool}) 
    # if the input variable is a Bool then this will always be treated as a binary 
    # variable 
    push!(newbinvars, var)
end 

function choosebinvar!(newbinvars, df, var, varvector::Vector{MissInt}) 
    # integer variables are treated as binary if their only values are 0 and 1 
    nmvector = varvector[identifynonmissings(varvector)]
    if maximum(nmvector) == 1 && minimum(nmvector) == 0 
        push!(newbinvars, var)
    else 
        nothing 
    end 
end 

function choosebinvar!(newbinvars, df, var, varvector::Vector{MissString}) 
    # string variables are treated as binary if they only have two non-missing values 
    nmvector = varvector[identifynonmissings(varvector)]
    if size(unique(nmvector), 1) == 2 
        push!(newbinvars, var)
    else 
        nothing 
    end 
end 

choosebinvar!(newbinvars, df, var, varvector::Vector) = nothing # for any other vector

function choosenoimputevars(df, vars, noimputevars::Nothing)
    newnoimputevars = Symbol[] 
    choosenoimputevars!(newnoimputevars, df, vars)
    return newnoimputevars 
end 

choosenoimputevars(df, vars, noimputevars::Vector{}) = noimputevars 

function choosenoimputevars!(newnoimputevars, df, vars)
    for var ∈ vars choosenoimputevar!(newnoimputevars, df, var) end 
end 

function choosenoimputevar!(newnoimputevars, df, var)
    varvector = getproperty(df, var)
    if maximum([ ismissing(x) for x ∈ varvector ]) == 0 push!(newnoimputevars, var) end
end 

function choosecontvars(df, vars, contvars::Nothing)
    return deepcopy(vars) # deepcopy, otherwise lose these variables when vars is subsequently filtered 
end 

function choosecontvars(df, vars, contvars::Vector{T}) where T <: Symbol
    return contvars 
end 
