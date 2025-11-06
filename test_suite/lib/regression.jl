using DataFrames
using CSV
using Random
using StatsBase
import SymbolicRegression: TemplateStructure
import SymbolicRegression: TemplateExpression
import SymbolicRegression: SRRegressor
import SymbolicRegression: L1DistLoss
import SymbolicRegression: L2DistLoss
import SymbolicRegression: HuberLoss
import SymbolicRegression: erf
import SymbolicRegression: erfc
import MLJ: machine, fit!, predict, report
import LoopVectorization
using Plots
using Bumper
using DynamicQuantities

# Julia command line arguments
using ArgParse
s = ArgParseSettings()
@add_argparse_argument(s, "--param", help="Parameter to analyze")
@add_argparse_argument(s, "--outpath", default=".", help="Output path for results")
@add_argparse_argument(s, "--inpath", default=".", help="Input path for catalogs")
parse_args(s)

param = get_argparse_value(s, "--param")
input_path = get_argparse_value(s, "--inpath")
output_path = get_argparse_value(s, "--outpath")
# Make the output path directory if needed
isdir(output_path) || mkpath(output_path)

# Read in the dataframe
df = CSV.read("$(input_path)/$(param).csv", DataFrame)
galaxies = unique(df.galaxy)
xcols = ["base_$(param)", "nres",  "sblim0"]
ycol = "$(param)"

# Sampling galaxies for training
Random.seed!(312)  # Set random seed for reproducibility
galaxies_train = sample(galaxies, 70, replace=false)  # Randomly select 70 galaxies

# Splitting the DataFrame into training and testing sets
df_train = df[in.(df.galaxy, Ref(galaxies_train)), :]
df_test = df[.!in.(df.galaxy, Ref(galaxies_train)), :]

# Define training data
Xtrain = Matrix(df_train[:,xcols]);
ytrain = df_train[:,ycol];
X = (; x1=Xtrain[:,1], x2=Xtrain[:,2], x3=Xtrain[:,3])
Y = df_train[:,ycol];

# Define the symbolic regression equation structure: param = base * f(x, y) + g(x, y)
structure = TemplateStructure{(:f, :g)}(
    ((; f, g), (x1, x2, x3)) ->x1*f(x2,x3) + g(x2,x3)
)

# Create the SRRegressor model
model = SRRegressor(;
    binary_operators=(+, -, *, /),
    unary_operators=(exp, tanh, log10, inv, sqrt),
    complexity_of_operators=[ (*)=>0, (+)=>0],
    elementwise_loss=L1DistLoss(),
    nested_constraints=[exp => [exp => 0, tanh => 0, sqrt=>0], 
                        tanh => [exp => 0, tanh => 0, sqrt=>0],
                        sqrt => [exp=>0, tanh=>0, sqrt=>0],
                        log10 => [exp=>0, tanh=>0, sqrt=>0]],
    niterations=1000,
    ncycles_per_iteration=500,
    maxsize=25,
    turbo=true,
    batching=true,
    batch_size=100,
    populations=50,
    population_size=50,
    expression_type=TemplateExpression,
    expression_options=(; structure),
    output_directory="$(output_path)/$(param)_fits",
)

# Train
mach = machine(model, X, Y)
fit!(mach)
report(mach)