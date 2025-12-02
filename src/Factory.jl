# throw(ErrorException("Oppps! No methods defined in src/Factory.jl. What should you do here?"))




# --- PUBLIC METHODS BELOW HERE -------------------------------------------------------------------------------- #
"""
    build(base::String, model::MyWeatherGridPointEndpointModel) -> String

This function is used to build a URL string that can be used to make a HTTP GET call to the National Weather Service API.
It takes two arguments, a base URL string, and a model of type `MyWeatherGridPointEndpointModel`. 

### Arguments
- `base::String` - The base URL string.
- `model::MyWeatherGridPointEndpointModel` - The model that contains the latitude and longitude of the grid point.

### Returns
- `String` - The complete URL string that can be used to make a HTTP GET call to the National Weather Service API.
"""
function build(base::String, model::MyWeatherGridPointEndpointModel)::String
    
    # TODO: implement this function, and remove the throw statement
    # throw(ArgumentError("build(base::String, model::MyWeatherGridPointEndpointModel) not implemented yet!"));

    # build the URL string -
    url_string = "$(base)/points/$(model.latitude),$(model.longitude)";

    # return the URL string -
    return url_string;
end

function build(base::String, model::MyBiggModelsEndpointModel; apiversion::String = "v2")::String
    
    # TODO: implement this function, and remove the throw statement
    # throw(ArgumentError("build(base::String, model::MyWeatherGridPointEndpointModel) not implemented yet!"));

    # build the URL string -
    url_string = "$(base)/api/$(apiversion)/models";

    # return the URL string -
    return url_string;
end

function build(base::String, model::MyBiggModelsDownloadModelEndpointModel; apiversion::String = "v2")::String

    # get data -
    bigg_id = model.bigg_id;

    # build the URL string -
    url_string = "$(base)/api/$(apiversion)/models/$(bigg_id)/download";

    # return the URL string -
    return url_string;
end

function build(base::String, model::MyClassicalHopfieldNetworkModel)
    # placeholder for other overloads; kept for compatibility
end


"""
    build(::Type{MyClassicalHopfieldNetworkModel}, params::NamedTuple)

Construct a `MyClassicalHopfieldNetworkModel` from the provided `memories`.

Arguments
- `params::NamedTuple` - must contain a key `:memories` with an array whose columns
  are the binary patterns (values ±1) to encode. Each column is a memory vector.

Returns
- `MyClassicalHopfieldNetworkModel` with `W` set by the Hebbian rule and `b` set to zeros.
"""
function build(::Type{MyClassicalHopfieldNetworkModel}, params::NamedTuple)

    # extract memories -
    if !(:memories in keys(params))
        throw(ArgumentError("build(MyClassicalHopfieldNetworkModel, params) requires a named tuple with key :memories"));
    end

    memories = params.memories;

    # sizes -
    N = size(memories, 1);
    K = size(memories, 2);

    # allocate -
    W = zeros(Float64, N, N);

    # Hebbian outer-product sum (classic Hopfield). We normalize by N.
    for μ ∈ 1:K
        s = Float64.(memories[:, μ]);
        W .+= s * s';
    end
    W ./= N;

    # zero-out diagonal (no self-connections)
    for i ∈ 1:N
        W[i, i] = 0.0;
    end

    # classical Hopfield has zero bias by default
    b = zeros(Float64, N);

    # construct model instance
    model = MyClassicalHopfieldNetworkModel(W, b);

    # compute per-memory energies: E(s) = -0.5 * s' * W * s - b' * s
    energies = zeros(Float64, K);
    for μ ∈ 1:K
        s = Float64.(memories[:, μ]);
        energies[μ] = -0.5 * dot(s, W * s) - dot(b, s);
    end

    model.energy = energies;

    return model;
end
# --- PUBLIC METHODS ABOVE HERE -------------------------------------------------------------------------------- #
