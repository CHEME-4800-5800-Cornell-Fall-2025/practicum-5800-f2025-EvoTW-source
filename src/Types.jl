# throw(ErrorException("Oppps! No methods defined in src/Types.jl. What should you do here?"))
abstract type AbstractWeatherEndpointModel end
abstract type AbstractBiggEndpointModel end

"""
    mutable struct MyWeatherGridPointEndpointModel <: AbstractWeatherEndpointModel

A model for the weather grid point endpoint.

### Fields
- `latitude::Float64` - The latitude of the grid point.
- `longitude::Float64` - The longitude of the grid point.

### Constructors
- `MyWeatherGridPointEndpointModel(; latitude::Float64 = 0.0, longitude::Float64 = 0.0)` - Returns a new `MyWeatherGridPointEndpointModel` instance.
"""
mutable struct MyWeatherGridPointEndpointModel <: AbstractWeatherEndpointModel

    # data -
    latitude::Float64
    longitude::Float64
  
    # methods -
    MyWeatherGridPointEndpointModel(; 
        latitude::Float64 = 0.0, longitude::Float64 = 0.0) = new(latitude, longitude);
end

"""
    mutable struct MyWeatherForecastEndpointModel <: AbstractWeatherEndpointModel

A model for the weather forecast endpoint. This type is empty, with no fields.
"""
mutable struct MyWeatherForecastEndpointModel <: AbstractWeatherEndpointModel
    MyWeatherForecastEndpointModel() = new(); # empty
end



struct MyBiggModelsEndpointModel <: AbstractBiggEndpointModel

    # methods -
    MyBiggModelsEndpointModel() = new();
end

mutable struct MyBiggModelsDownloadModelEndpointModel <: AbstractBiggEndpointModel

    # data -
    bigg_id::String

    # methods -
    MyBiggModelsDownloadModelEndpointModel() = new();
end


mutable struct MyClassicalHopfieldNetworkModel

    # data -
    W::Array{Float64,2}    # weight matrix (N x N)
    b::Array{Float64,1}    # bias vector (N,)
    N::Int                 # number of neurons
    energy::Array{Float64,1} # per-memory energies (K,)

    # constructors -
    MyClassicalHopfieldNetworkModel() = new(zeros(Float64,0,0), zeros(Float64,0), 0, zeros(Float64,0));
    MyClassicalHopfieldNetworkModel(N::Int) = new(zeros(Float64,N,N), zeros(Float64,N), N, zeros(Float64,0));
    MyClassicalHopfieldNetworkModel(W::Array{Float64,2}, b::Array{Float64,1}) = new(W, b, size(W,1), zeros(Float64,0));
end