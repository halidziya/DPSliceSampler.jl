include("ParallelUtils.jl")
@everywhere using Distributions
using StatsBase
using Plots
pyplot()


function stickBreaker(ustar, betastar = 1.0, α =  1) # Usually people are using this value)
  MAXCOMP = 20 # Maximum number of sticks
  v = rand(Beta(1,α),MAXCOMP)
  beta = zeros(MAXCOMP+1)
  i = 0
  totallength = betastar
  for i=1:MAXCOMP
    beta[i] = betastar*v[i]
    betastar = betastar*(1-v[i])
    if (betastar < ustar)
      break
    end
  end
  beta[i+1] = totallength - sum(beta[beta.>0])
  return beta[beta.>0]
end

function mixtureStats(x,z)
  D = size(x)[1]
  NPOINTS = size(x)[2]
  NTABLE = length(unique(z))
  samplesum = zeros(D,NTABLE)
  samplescatter = zeros(D,D,NTABLE)
  samplecount = zeros(Int32,NTABLE)
  for i=1:NTABLE
    samplecount[i] = sum(z.==i)
    samplesum[:,i] = sum(x[:,z.==i],2)
    samplescatter[:,:,i] = Base.covm(x[:,z.==i],samplesum[:,i]/samplecount[i],2,false) .* samplecount[i]
  end
  return samplecount,samplesum,samplescatter
end

function getLikelihoods(loglikelihoods,x,μ,Σ,β,u)
  NPOINTS = size(x,2)
  NTABLE = length(μ)
  for j=1:NTABLE
    mvn = MultivariateNormal(μ[j],Σ[j])
    @sync @parallel for i=1:NPOINTS
      if (β[j] > u[i])
        loglikelihoods[j,i] = logpdf(mvn,x[:,i])
      end
    end
  end
  return loglikelihoods
end

#Changes the matrix in place
@everywhere function sampleFromLog(p)
  p = exp(p-maximum(p))
  p = p/sum(p)
  sum(rand().>cumsum(p))+1
end

function sampleFromLogs(loglikelihoods)
  labels = SharedArray(Int32,size(loglikelihoods,2))
  reduceComlumnShared(labels,loglikelihoods,sampleFromLog)
  return labels
end

function mapId(z)
  at=zeros(Int32,maximum(z))
  at[unique(z)] = 1:length(unique(z))
  return at[z]
end



function SliceSampler(x,D=size(x,1),η = D + 3,Ψ = eye(D)*η ,κ₀ = 1,μ₀ = zeros(D),α = 1,NINITIAL = 10,NITER=500)
  #Initialization
  NPOINTS = size(x,2)
  z = rand(1:NINITIAL,NPOINTS)
  μ = [rand(D) for i=1:NINITIAL]
  Σ = [eye(D) for i=1:NINITIAL]
  u = zeros(NPOINTS)
  β = ones(NINITIAL)/NINITIAL
  NTABLE = NINITIAL
  loglikelihoods = SharedArray(Float64,(NTABLE,NPOINTS))

  # Slice sampler
  for iter=1:NITER
    #3
    println(iter)
    loglikelihoods = SharedArray(Float64,(NTABLE,NPOINTS))
    sharedFill(loglikelihoods,-Inf)
    getLikelihoods(loglikelihoods,x,μ,Σ,β,u) #*
    z = sampleFromLogs(loglikelihoods)
    z = mapId(z)
    samplecount,samplesum,samplescatter = mixtureStats(x,z) #*
    #5, #1
    β = rand(Dirichlet(vcat(samplecount,α)))
    u = rand(NPOINTS).*β[z]
    #2
    newsticks = stickBreaker(minimum(u),β[end])
    β=vcat(β[1:end-1],newsticks)
    NTABLE = length(β)
    #4 , I have changed order since parameters does not depend on beta
    Σ = [rand(InverseWishart(η,Ψ)) for i=1:NTABLE] #*
    μ = [rand(MultivariateNormal(μ₀,Σ[i]/κ₀)) for i=1:NTABLE] #*
    for i=1:length(samplecount)
      n =  samplecount[i]
      m = samplesum[:,i]/n
      meanscatter = m*m'
      Σ[i]=rand(InverseWishart(η + n,samplescatter[:,:,i] + Ψ + (κ₀*n/(κ₀+n))*meanscatter))
      μ[i] =rand(MultivariateNormal(samplesum[:,i]/(κ₀+n),Σ[i]/(κ₀+n)))
    end
  end
  return (z,μ,Σ,β)
end
