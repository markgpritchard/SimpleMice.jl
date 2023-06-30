
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulate missing completely at random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
    function mcar(df::DataFrame, vars, p)

Create a copy of `df` and set a proportion to missing in the parameters `vars`.

See mutating version `mcar!`
"""
function mcar(df, args...)
    ndf = deepcopy(df)
    mcar!(ndf, args...)
    return ndf 
end 

"""
    function mcar!(df::DataFrame, vars, p)

Set a proportion `p` of `DataFrame` values to missing in the parameters `vars`.

`vars` can be a vector of variables or a single variable, and `p` can be a single 
    number applied to all variables or a vector of values, one for each variable.

Values are set to missing if `rand() < p` so the proportion of values set to missing 
    will not necessarily exactly equal `p`.
"""
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Draft test dataset 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function testdataset()
    ages::Vector{Union{Float64, Missing}} = round.([ i / 11 + rand() * 5 for i ∈ 1:1000 ], digits = 2)
    sexes::Vector{Union{String, Missing}} = [ rand() < .5 ? "F" : "M" for _ ∈ 1:1000 ]
    vara::Vector{Union{Bool, Missing}} = [ rand() < age * .0025 for age ∈ ages ]
    varb::Vector{Union{Int, Missing}} = [ rand() < (.1 + age * .0025) for age ∈ ages ]
    varc::Vector{Union{Bool, Missing}} = [ rand() < .01 for _ ∈ 1:1000 ]
    vardm = [ age < 15 ? age / 10 : 1.5 + ifelse(sex == "F", 0, .2) for (age, sex) ∈ zip(ages, sexes) ]
    vard::Vector{Union{Float64, Missing}} = [ rand(Normal(v, .015 * v)) for v ∈ vardm ]
    vare::Vector{Union{Float64, Missing}} = [ rand(Normal(25, 5)) * v^2 for v ∈ vard ]
    varf::Vector{Union{String, Missing}} = [ rand() < .8 ? "Y" : "N" for _ ∈ 1:1000 ]
    varg::Vector{Union{Bool, Missing}} = [ ages[i] > 18 && varf[i] == "Y" ? rand() < .8 : false for i ∈ 1:1000 ]
    op = [ .025 + .00025 * ages[i] + .05 * (vara[i]  + varb[i] + varc[i]) + .005 * (vare[i] / vard[i]^2) / 30 
        for i ∈ 1:1000 ]
    oq = [ varf[i] == "Y" ? 2 * op[i] : op[i] for i ∈ 1:1000 ]
    os = [ varg[i] ? oq[i] / 4 : oq[i] for i ∈ 1:1000 ]
    outcome = [ rand() < osv / (osv + 1) for osv ∈ os ]
    vars = [ ages, sexes, vara, varb, varc, vard, vare, varf, varg, outcome ]
    varnames = [ :Ages, :Sexes, :Vara, :Varb, :Varc, :Vard, :Vare, :Varf, :Varg, :Outcome ]
    df = DataFrame([ name => v for (name, v) ∈ zip(varnames, vars) ])
    return df
end 
