
using SimpleMice
using Test
using DataFrames, Distributions, Random

function testdataset()
    ages::Vector{Union{Float64, Missing}} = round.([ i / 11 + rand() * 5 for i ∈ 1:1000 ], digits = 2)
    sexes::Vector{Union{String, Missing}} = [ rand() < .5 ? "F" : "M" for _ ∈ 1:1000 ]
    vara::Vector{Union{Bool, Missing}} = [ rand() < age * .0025 for age ∈ ages ]
    varb::Vector{Union{Int, Missing}} = [ rand() < (.1 + age * .0025) for age ∈ ages ]
    varc::Vector{Union{Bool, Missing}} = [ rand() < .005 for _ ∈ 1:1000 ]
    vard::Vector{Union{Float64, Missing}} = rand(Normal(25, 5), 1000)
    vare::Vector{Union{String, Missing}} = [ rand() < .8 ? "Y" : "N" for _ ∈ 1:1000 ]
    varf::Vector{Union{Bool, Missing}} = [ ages[i] > 18 && vare[i] == "Y" ? rand() < .8 : false for i ∈ 1:1000 ]
    op = [ .025 + .00025 * ages[i] + .05 * (vara[i]  + varb[i] + varc[i]) + .005 * vard[i] / 30 for i ∈ 1:1000 ]
    oq = [ vare[i] == "Y" ? 2 * op[i] : op[i] for i ∈ 1:1000 ]
    os = [ varf[i] ? oq[i] / 4 : oq[i] for i ∈ 1:1000 ]
    outcome = [ rand() < osv / (osv + 1) for osv ∈ os ]
    vars = [ ages, sexes, vara, varb, varc, vard, vare, varf, outcome ]
    varnames = [ :Ages, :Sexes, :Vara, :Varb, :Varc, :Vard, :Vare, :Varf, :Outcome ]
    df = DataFrame([ name => v for (name, v) ∈ zip(varnames, vars) ])
    return df
end 

Random.seed!(1729)
const NOMISSINGDATA = testdataset()
const MCAR1 = mcar(
    NOMISSINGDATA, 
    [ :Ages, :Sexes, :Vara, :Varb, :Varc, :Vard, :Vare, :Varf ],
    .01
)
const MCAR50 = mcar(
    NOMISSINGDATA, 
    [ :Ages, :Sexes, :Vara, :Varb, :Varc, :Vard, :Vare, :Varf ],
    .5
)

@testset "SimpleMice.jl" begin
    @testset "MCAR function" begin 
        @test 3 <= minimum(describe(MCAR1).nmissing[1:8]) <= 10
        @test 10 <= maximum(describe(MCAR1).nmissing[1:8]) <= 30
        @test 200 <= minimum(describe(MCAR50).nmissing[1:8]) <= 500
        @test 500 <= maximum(describe(MCAR50).nmissing[1:8]) <= 800
    end # @testset "MCAR function"
    @testset "Imputation tests" begin
        @testset "List of symbols" begin 
            @test SimpleMice.symbollist([ :a, :b, :c ]) == [ :a, :b, :c ] 
            @test SimpleMice.symbollist([ "a", "b", "c" ]) == [ :a, :b, :c ] 
            @test SimpleMice.symbollist(nothing) == nothing
            @test SimpleMice.symbollist(Float64[]) == Symbol[]
            @test_throws AssertionError SimpleMice.symbollist([ 2 ])
        end # @testset "List of symbols"
        
        @testset "Classification of variables" begin
            @testset "None missing" begin 
                vars = Symbol.(names(NOMISSINGDATA))
                bv, cv, niv = SimpleMice.classifyvars!(vars, nothing, nothing, nothing, NOMISSINGDATA; 
                    printdropped = false)
                @test bv == Symbol[]
                @test cv == Symbol[]
                @test niv == [ :Ages, :Sexes, :Vara, :Varb, :Varc, :Vard, :Vare, :Varf, :Outcome ]
                @test vars == Symbol[]
            end # @testset "None missing"
            @testset "Auto - some missing" for df ∈ [ MCAR1, MCAR50 ]
                vars = Symbol.(names(df))
                bv, cv, niv = SimpleMice.classifyvars!(vars, nothing, nothing, nothing, df; 
                    printdropped = false)
                @test bv == [ :Sexes, :Vara, :Varb, :Varc, :Vare, :Varf ]
                @test cv == [ :Ages, :Vard ]
                @test niv == [ :Outcome ]
                @test vars == Symbol[]
            end # @testset "Auto - some missing" for df ∈ [ MCAR1, MCAR50 ]
            @testset "Manual binary" for df ∈ [ MCAR1, MCAR50 ]
                vars = Symbol.(names(df))
                bv, cv, niv = SimpleMice.classifyvars!(vars, [ :Sexes ], nothing, nothing, df; 
                    printdropped = false)
                @test bv == [ :Sexes ] 
                @test cv == [ :Ages, :Vara, :Varb, :Varc, :Vard, :Varf ]
                @test niv == [ :Outcome ]
                @test vars == [ :Vare ]
            end # @testset "Manual binary" for df ∈ [ MCAR1, MCAR50 ]
        end # @testset "Classification of variables"
        
        @testset "No change if no missing values" begin
            micedata = mice(NOMISSINGDATA, [ :Ages, :Sexes, :Vara, :Varb, :Varc, :Vard, :Vare, :Varf ]; 
                n = 2, verbose = false)
            @test isa(micedata, SimpleMice.ImputedDataFrame)
            @test micedata.originaldf == NOMISSINGDATA
            @test micedata.numberimputed == 2
            @testset for i ∈ 1:2 @test micedata.imputeddfs[i] == NOMISSINGDATA end 
        end # @testset "No change if no missing values"
        
        @testset "Similar if only 1% missing" begin
            micedata = mice(MCAR1, [ :Ages, :Sexes, :Vara, :Varb, :Varc, :Vard, :Vare, :Varf ]; 
                m = 1000, n = 10, verbose = false)
            @test isa(micedata, SimpleMice.ImputedDataFrame)
            @test skipmissing(micedata.originaldf) == skipmissing(MCAR1)
            @test micedata.numberimputed == 10
            @testset for v ∈ [ :Ages, :Vara, :Varb, :Varc, :Vard, :Varf ]
                # I'm not sure how reasonale this test is but currently the package passes it
                nomissingmean = mean(getproperty(NOMISSINGDATA, v))
                imputedmean = mean(getvalues(micedata, v))
                @test .99 * nomissingmean <= imputedmean <= 1.01 * nomissingmean 
                nomissingvar = var(getproperty(NOMISSINGDATA, v))
                imputedvar = var(getvalues(micedata, v))
                @test .99 * nomissingvar <= imputedvar <= 1.01 * nomissingvar 
            end # @testset for v ∈ [ :Ages, :Vara, :Varb, :Varc, :Vard, :Varf ]
        end #  @testset "Similar if only 1% missing"
        
        @testset "Function works if 50% missing" begin
            micedata = mice(MCAR50, [ :Ages, :Sexes, :Vara, :Varb, :Varc, :Vard, :Vare, :Varf ]; 
                n = 5, verbose = false)
            @test isa(micedata, SimpleMice.ImputedDataFrame)
            @test skipmissing(micedata.originaldf) == skipmissing(MCAR50)
            @test micedata.numberimputed == 5
        end # @testset "Function works if 50% missing"
    end # @testset "Imputation tests"
end # @testset "SimpleMice.jl"
