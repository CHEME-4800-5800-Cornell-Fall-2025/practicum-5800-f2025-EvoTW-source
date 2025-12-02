using Random

#throw(ErrorException("Oppps! No methods defined in src/Compute.jl. What should you do here?"))
"""
    softmax(x::Array{Float64,1})::Array{Float64,1}

Compute the softmax of a vector. 
This function takes a vector of real numbers and returns a vector of the same size with the softmax of the input vector, 
i.e., the exponential of the input vector divided by the sum of the exponential of the input vector.


### Arguments
- `x::Array{Float64,1}`: a vector of real numbers.

### Returns
- `::Array{Float64,1}`: a vector of the same size as the input vector with the softmax of the input vector.
"""
function softmax(x::Array{Float64,1})::Array{Float64,1}
    
    # compute the exponential of the vector
    y = exp.(x);
    
    # compute the sum of the exponential
    s = sum(y);
    
    # compute the softmax
    return y / s;
end

"""
    binary(S::Array{Float64,2})::Array{Int64,2}
"""
function binary(S::Array{Float64,2})::Array{Int64,2}
    
    # initialize -
    number_of_rows = size(S, 1);
    number_of_columns = size(S, 2);
    B = zeros(Int64, number_of_rows, number_of_columns);

    # main -
    for i ∈ 1:number_of_rows
        for j ∈ 1:number_of_columns
            if (S[i, j] != 0.0) # if the value is not zero, then B[i,j] = 1
                B[i, j] = 1;
            end
        end
    end

    # return -
    return B;
end

function reactionstring(metabolites)::String

    # initialize -
    reactant_string = "";
    product_string = "";
    arrow = "=";

    # we need to iterate over the keys and values of the dictionary, and then we can concatenate the strings
    species_list = keys(metabolites) |> collect |> sort; # sort the species, alphabetically
    reactants = Dict{String,Any}();
    products = Dict{String,Any}();
    for species ∈ species_list
        if metabolites[species] < 0
            reactants[species] = metabolites[species];
        elseif metabolites[species] > 0
            products[species] = metabolites[species];
        end
    end

    # iterate over the reactants
    for (species, stoichiometry) ∈ reactants
        reactant_string *= "$(abs(stoichiometry)) " * species * " + ";
    end
    reactant_string = reactant_string[1:end-3]; # remove the last " + "

    # iterate over the products
    for (species, stoichiometry) ∈ products
        product_string *= "$(abs(stoichiometry)) " * species * " + ";
    end
    product_string = product_string[1:end-3]; # remove the last " + "

    # return -
    return reactant_string * " = " * product_string;
end


"""
    decode(s::AbstractVector; rows::Int = nothing, cols::Int = nothing)

Convert a flattened Hopfield state vector `s` with values in {-1, 1} to a
2D image matrix with values in [0,1] suitable for display (e.g. `Gray.`).

If `rows` and `cols` are not provided, the function will attempt to read
`number_of_rows` and `number_of_cols` from `Main` (the notebook), and if those
are absent it will infer a square image size when possible.
"""
function decode(s::AbstractVector; rows::Union{Nothing,Int}=nothing, cols::Union{Nothing,Int}=nothing)

    # determine rows/cols from provided args, notebook globals, or infer
    if rows === nothing && isdefined(Main, :number_of_rows)
        rows = getfield(Main, :number_of_rows)
    end
    if cols === nothing && isdefined(Main, :number_of_cols)
        cols = getfield(Main, :number_of_cols)
    end

    if rows === nothing && cols === nothing
        # try to infer square size
        L = length(s)
        r = round(Int, sqrt(L))
        if r * r == L
            rows = r; cols = r
        else
            throw(ArgumentError("Cannot infer image dimensions from vector length $(L). Provide rows and cols."))
        end
    elseif rows === nothing
        rows = div(length(s), cols)
    elseif cols === nothing
        cols = div(length(s), rows)
    end

    if rows * cols != length(s)
        throw(ArgumentError("Provided rows*cols = $(rows*cols) does not match length(s)=$(length(s))."))
    end

    # convert from {-1,1} to [0.0,1.0]
    mat = Float32.((Float64.(s) .+ 1.0) ./ 2.0);

    return reshape(mat, rows, cols);
end


"""
    energy(model::MyClassicalHopfieldNetworkModel, s::AbstractVector)::Float64

Compute the Hopfield network energy for state vector `s` using the model's
weights and bias: E(s) = -0.5 * s' * W * s - b' * s.
"""
function energy(model::MyClassicalHopfieldNetworkModel, s::AbstractVector)::Float64
    svec = Float64.(s);
    return -0.5 * dot(svec, model.W * svec) - dot(model.b, svec);
end


"""
    recover(model::MyClassicalHopfieldNetworkModel, s0::AbstractVector, true_image_energy::Real; maxiterations::Int = 10*get(model, :N, 0), patience::Int = 5)

Run asynchronous Hopfield updates starting from `s0`. Returns a pair
`(frames, energydictionary)` where `frames::Dict{Int,Vector{Int32}}` maps
step indices to the network state at that step and `energydictionary::Dict{Int,Float64}`
maps the same keys to the corresponding energy values.

Keyword args:
- `maxiterations`: maximum number of update attempts
- `patience`: number of consecutive identical states required to declare convergence
"""
function recover(model::MyClassicalHopfieldNetworkModel, s0::AbstractVector, true_image_energy::Real; maxiterations::Int = 10*model.N, patience::Int = 5)

    # initialize state
    N = model.N;
    if length(s0) != N
        throw(ArgumentError("Initial state length does not match model.N"));
    end

    # make state integer ±1
    s = Int32.(s0);

    frames = Dict{Int, Vector{Int32}}();
    energydictionary = Dict{Int, Float64}();

    # record initial
    step = 1;
    frames[step] = copy(s);
    energydictionary[step] = energy(model, s);

    # history for patience check
    history = Vector{Vector{Int32}}();
    push!(history, copy(s));

    rng = Random.GLOBAL_RNG;

    for t in 1:maxiterations

        # choose a random neuron to update
        i = rand(rng, 1:N);

        # compute local field h_i = sum_j w_ij s_j - b_i
        # convert to Float64 for dot product
        h = dot(model.W[i, :], Float64.(s)) - model.b[i];

        new_si = ifelse(h >= 0.0, Int32(1), Int32(-1));

        if new_si != s[i]
            s[i] = new_si;
        end

        # record this step
        step += 1;
        frames[step] = copy(s);
        energydictionary[step] = energy(model, s);

        # update history for patience
        push!(history, copy(s));
        if length(history) > patience
            deleteat!(history, 1);
        end

        # check convergence: all states in history identical
        converged = length(history) >= patience && all(x -> x == history[1], history);
        if converged
            break
        end

    end

    return frames, energydictionary;
end