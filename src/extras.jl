
"""
    function desentinelize!(df::DataFrame)
        
Remove `SentinelArrays` from `DataFrame`.

Based on code shared at 
    https://discourse.julialang.org/t/disable-sentinelarrays-for-csv-read/54843/3
"""
function desentinelize!(df::DataFrame)
    for (i, col) ∈ enumerate(eachcol(df)) df[!, i] = collect(col) end
end
