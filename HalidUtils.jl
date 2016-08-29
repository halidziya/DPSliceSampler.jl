using Plots
export vec2Mat
  function vec2Mat(v)
    d2 = 0
    if (ndims(v[1])==2)
      (d1,d2) = size(v[1])
      m = zeros(length(v),d1,d2)
      for i=1:length(v)
        m[i,:,:]=v[i]
      end
    elseif (ndims(v[1])==1)
      d1 = collect(size(v[1]))[1]
      m = zeros(length(v),d1)
      for i=1:length(v)
        m[i,:]=v[i]
      end
    end
    return m
  end

  function covPoints(μ,Σ,points=100)
    angle = linspace(0, 2*pi,points)
    u = hcat(cos(angle),sin(angle))
    return broadcast(+,(u*chol(Σ)),μ')
  end

  function plotCovs(μ,Σ)
    for i=1:length(μ)
      at = (covPoints(μ[i][1:2],Σ[i][1:2,1:2]))
      plot!(at[:,1],at[:,2],leg=false)
    end
  end
