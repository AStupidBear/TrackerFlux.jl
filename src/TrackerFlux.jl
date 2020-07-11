module TrackerFlux

using Flux, Tracker

Flux.param(x) = Tracker.param(x)

track(m) = fmap(x -> x isa AbstractArray ? Flux.param(x) : x, m)
untrack(m) = fmap(Tracker.data, m)

function Flux.Optimise.update!(opt, xs::Tracker.Params, gs)
    for x in xs
        Flux.Optimise.update!(opt, x, gs[x])
    end
end
function Flux.Optimise.update!(opt, x, x̄)
    Tracker.update!(x, -Flux.Optimise.apply!(opt, Tracker.data(x), Tracker.data(x̄)))
end

_truncate(x::AbstractArray) = Tracker.data(x)
_truncate(x::Tuple) = _truncate.(x)
truncate!(m::Flux.Recur) = (m.state = _truncate(m.state))
truncate!(m) = foreach(truncate!, Flux.functor(m)[1])

function Flux.destructure(m)
    xs = []
    fmap(m) do x
        x isa AbstractArray && push!(xs, x)
        return x
    end
    θ = vcat(vec.(Tracker.data.(xs))...)
    re = p -> Flux._restructure(m, p)
    return Flux.param(θ), re
end

function overload_gradient()
    @eval Flux.gradient(f, args...) = Tracker.gradient(f, args...)
end

function __init__()
    @eval Flux truncate! = $truncate!
end

end
