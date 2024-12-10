module fe_tools

using LinearAlgebra, SparseArrays

using AD4SM

using  .AD4SM
import .Materials.Hooke
import .Materials.getσ,   .Materials.getϕ
import .Elements.C3DP, .Elements.C3D, .Elements.CElems
import .Elements.C3DElems, .Elements.CPElems

export getϕ, getϕu, getϕd, getT, getVd, makeKt
export PhaseField
export Elements, Materials, adiff, Hooke
export CPElems, CElems, C3DElems, ElTypes

## Phase field
#
struct PhaseField{M}
  l0::Number 
  Gc::Number 
  mat::M
end
PhaseField(l0::T, Gc::T, mat::M) where {T<:Number, M<:Hooke} = PhaseField{M}(l0, Gc, mat)
#
# × operator
function ×(ϕ::adiff.D2{N,M,T},F::Array{adiff.D1{P,T}}) where {N,M,P,T}
  val  = ϕ.v
  grad = adiff.Grad(zeros(T,P))
  hess = adiff.Grad(zeros(T,(P+1)P÷2))
  for ii=1:N
    grad += ϕ.g[ii]*F[ii].g
  end
  for ii=2:N, jj=1:ii-1
    hess += ϕ.h[ii,jj]*(F[ii].g*F[jj].g + F[jj].g*F[ii].g)
  end  
  for ii=1:N
    hess += ϕ.h[ii,ii]F[ii].g*F[ii].g
  end
  adiff.D2(val, grad, hess)
end
function ×(ϕ::adiff.D1{N,T},F::Array{adiff.D1{P,T}}) where {N,P,T}
  val  = ϕ.v
  grad = adiff.Grad(zeros(T,P))
  for ii=1:N
    grad += ϕ.g[ii]*F[ii].g
  end  
  adiff.D1(val, grad)
end
# standard materials
function getϕ(elem::C3DElems{P,M,T,I} where {M,T,I}, u0::Matrix{D}) where {P,D}
  u,v,w = u0[1:3:end],u0[2:3:end],u0[3:3:end]
  wgt = elem.wgt
  ϕ   = zero(D) 
  for ii=1:P
    Nx,Ny,Nz = elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    F  = [Nx⋅u Ny⋅u Nz⋅u;
          Nx⋅v Ny⋅v Nz⋅v;
          Nx⋅w Ny⋅w Nz⋅w ] + I
    ϕ += wgt[ii]Materials.getϕ(F, elem.mat)
  end
  ϕ
end
function getϕ(elem::C3DElems{P,M,T,I} where {M,T,I}, 
              u0::Matrix{D}) where {P,D<:adiff.D1}
  wgt = elem.wgt
  ϕ   = zero(D) 
  for ii=1:P
    F    = Elements.getF(elem, u0, ii)
    valF = adiff.val.(F)
    ∂ϕ   = fe_tools.getϕ(adiff.D1(valF), elem.mat)
    ϕ   += wgt[ii]∂ϕ×F
  end
  ϕ
end
function getϕ(elem::C3DElems{P,M,T,I} where {M,T,I}, u0::Matrix{D}) where {P,D<:adiff.D2}
  u0      = adiff.D1.(u0)
  u, v, w = u0[1:3:end], u0[2:3:end],u0[3:3:end]
  wgt = elem.wgt
  ϕ   = zero(D) 
  for ii=1:P
    N0,Nx,Ny,Nz = elem.N0[ii],elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    F    = [Nx⋅u Ny⋅u Nz⋅u;
            Nx⋅v Ny⋅v Nz⋅v;
            Nx⋅w Ny⋅w Nz⋅w ] + I
    valF = adiff.val.(F)
    ∂ϕ   = fe_tools.getϕ(adiff.D2(valF), elem.mat)
    ϕ   += wgt[ii]∂ϕ×F
  end
  ϕ
end
# phase field functions
#=
  this is the split as in 
Amor, H.; Marigo, J.J.; Maurini, C. Regularized formulation of the variational 
brittle fracture with unilateral contact: Numerical experiments. 
Journal of the Mechanics and Physics of Solids 2009, 57, 1209–1229.
=#
function getϕ(F::Matrix{D}, d::U, ∇d::Vector{U},
              mat::PhaseField{M} where M) where {D,U}

  I1, I1sq  = get1stinvariants(F, mat.mat)
  ν, Es     = mat.mat.ν, mat.mat.E
  λ         = Es*ν/(1+ν)/(1-2ν) 
  μ         = Es/2/(1+ν) 

  γ = d + mat.l0^2*(∇d⋅∇d)
  ψ = if I1≥0
    (1-d)^2*(λ/2*I1^2 + μ*I1sq)
  else
    λ/2*I1^2 + (1-d)^2*μ*I1sq
  end

  ψ + mat.Gc/2mat.l0*γ
end
function getϕ!(F::Matrix{D}, d::U, ∇d::Vector{U},
              mat::PhaseField{M} where M,
              ϕmax::Number) where {D,U}

  I1, I1sq  = get1stinvariants(F, mat.mat)
  ν, Es     = mat.mat.ν, mat.mat.E
  λ         = Es*ν/(1+ν)/(1-2ν) 
  μ         = Es/2/(1+ν) 

  γ = d + mat.l0^2*(∇d⋅∇d)
  ψ = if I1≥0
    ϕmax = max(λ/2*I1^2 + μ*I1sq, ϕmax)
    (1-d)^2*ϕmax
  else
    ϕmax = max(μ*I1sq, ϕmax)
    λ/2*I1^2 + (1-d)^2*ϕmax
  end

  ψ + mat.Gc/2mat.l0*γ, ϕmax
end
# 
function get1stinvariants(F::Array{N,2} where N<:Number, mat::Hooke)

  if mat.small
    E = (F+transpose(F)-2I)/2 # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain 
  end

  I1   = E[1]+E[5]+E[9]
  I1sq = E[1]^2+E[5]^2+E[9]^2+2*(E[2]^2+E[3]^2+E[6]^2)

  return I1, I1sq
end
# 3D elements
function getϕ(elem::C3DElems{P,<:PhaseField}, d0::Vector{T}) where {P,T}
  ϕ   = zero{T} 
  for ii=1:P
    N0,Nx,Ny,Nz = elem.N0[ii],elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    d  = N0⋅d0
    ∇d = [Nx⋅d0, Ny⋅d0, Nz⋅d0]
    γ  = d^2 + mat.l0^2*(∇d⋅∇d)
    ϕ += wgt[ii]γ
  end
  ϕ
end
function getϕ(elem::C3DElems{P,<:PhaseField}, u0::Matrix{U}, d0::Vector{D})  where {P, U, D}

  @views u, v, w = u0[1:3:end], u0[2:3:end], u0[3:3:end]
  wgt = elem.wgt
  T   = promote_type(U,D)
  ϕ   = zero(T)
  for ii=1:P
    N0,Nx,Ny,Nz = elem.N0[ii],elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    F  = [1+Nx⋅u Ny⋅u  Nz⋅u;
          Nx⋅v  1+Ny⋅v Nz⋅v;
          Nx⋅w   Ny⋅w 1+Nz⋅w ]
    
    d  = N0⋅d0
    ∇d = [Nx⋅d0, Ny⋅d0, Nz⋅d0]
    ϕ += wgt[ii]getϕ(F, d, ∇d, elem.mat)::T
  end
  return ϕ
end
function getϕ!(elem::C3DElems{P,<:PhaseField}, 
              u0::Matrix{U}, d0::Vector{D},
              ϕmax::Vector{<:Number})  where {P, U, D}

  @views u, v, w = u0[1:3:end], u0[2:3:end], u0[3:3:end]
  wgt = elem.wgt
  T   = promote_type(U,D)
  ϕ   = zero(T)
  for ii=1:P
    N0,Nx,Ny,Nz = elem.N0[ii],elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    F  = [1+Nx⋅u Ny⋅u  Nz⋅u;
          Nx⋅v  1+Ny⋅v Nz⋅v;
          Nx⋅w   Ny⋅w 1+Nz⋅w ]
    
    d  = N0⋅d0
    ∇d = [Nx⋅d0, Ny⋅d0, Nz⋅d0]

    ϕii, ϕmax[ii] = getϕ!(F, d, ∇d, elem.mat, ϕmax[ii])
    ϕ += wgt[ii]ϕii::T
  end
  return ϕ
end
function getϕu(elem::C3DElems{P,<:PhaseField}, u0::Matrix{T}, d0::Vector{T})  where {P,T}

  u, v, w = u0[1:3:end], u0[2:3:end], u0[3:3:end]
  N       = lastindex(u0)  
  wgt     = elem.wgt
  val     = zero(T)
  grad    = zeros(T,N)
  hess    = zeros(T,(N+1)N÷2)
  δF      = zeros(T,N,9)

  for ii=1:P
    N0,Nx,Ny,Nz = elem.N0[ii],elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    δF[1:3:N,1] = δF[2:3:N,2] = δF[3:3:N,3] = Nx
    δF[1:3:N,4] = δF[2:3:N,5] = δF[3:3:N,6] = Ny
    δF[1:3:N,7] = δF[2:3:N,8] = δF[3:3:N,9] = Nz

    F    = [Nx⋅u Ny⋅u Nz⋅u;
	    Nx⋅v Ny⋅v Nz⋅v;
	    Nx⋅w Ny⋅w Nz⋅w ] + I
    d    = N0⋅d0
    ∇d   = [Nx⋅d0, Ny⋅d0, Nz⋅d0]
    ϕ    = getϕ(adiff.D2(F), d, ∇d, elem.mat)::Main.fe_tools.AD4SM.adiff.D2{9, 45, T}
    val += wgt[ii]ϕ.v
    for jj=1:9,i1=1:N
      grad[i1] += wgt[ii]*ϕ.g[jj]*δF[i1,jj]
      for kk=1:9,i2=1:i1
	hess[(i1-1)i1÷2+i2] += wgt[ii]*ϕ.h[jj,kk]*δF[i1,jj]*δF[i2,kk]
      end   
    end
  end

  adiff.D2(val, adiff.Grad(grad), adiff.Grad(hess))
end
function getϕu!(elem::C3DElems{P,<:PhaseField}, 
                u0::Matrix{T}, d0::Vector{T}, ϕmax::Vector{<:Number})  where {P,T}

  u, v, w = u0[1:3:end], u0[2:3:end], u0[3:3:end]
  N       = lastindex(u0)  
  wgt     = elem.wgt
  val     = zero(T)
  grad    = zeros(T,N)
  hess    = zeros(T,(N+1)N÷2)
  δF      = zeros(T,N,9)

  for ii=1:P
    N0,Nx,Ny,Nz = elem.N0[ii],elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    δF[1:3:N,1] = δF[2:3:N,2] = δF[3:3:N,3] = Nx
    δF[1:3:N,4] = δF[2:3:N,5] = δF[3:3:N,6] = Ny
    δF[1:3:N,7] = δF[2:3:N,8] = δF[3:3:N,9] = Nz

    F    = [Nx⋅u Ny⋅u Nz⋅u;
	    Nx⋅v Ny⋅v Nz⋅v;
	    Nx⋅w Ny⋅w Nz⋅w ] + I
    d    = N0⋅d0
    ∇d   = [Nx⋅d0, Ny⋅d0, Nz⋅d0]
    ϕ    = getϕ(adiff.D2(F), d, ∇d, elem.mat)::Main.fe_tools.AD4SM.adiff.D2{9, 45, T}
    val += wgt[ii]ϕ.v
    for jj=1:9,i1=1:N
      grad[i1] += wgt[ii]*ϕ.g[jj]*δF[i1,jj]
      for kk=1:9,i2=1:i1
	hess[(i1-1)i1÷2+i2] += wgt[ii]*ϕ.h[jj,kk]*δF[i1,jj]*δF[i2,kk]
      end   
    end
  end

  adiff.D2(val, adiff.Grad(grad), adiff.Grad(hess))
end
getϕd(elem::C3DElems{P,<:PhaseField}, u0::Matrix, d0::Vector)                where P = getϕ(elem, u0, adiff.D2(d0))
function makeresu(elem::CPElems{P,<:PhaseField}, u::Matrix{T}, 
                  d::Vector{T})  where {P,T}
  ϕ = 0
  for ii=1:P
    F  = Elements.getF(elem, adiff.D1(u), ii)
    δϕ = getϕ(adiff.D1(adiff.val.(F)), d, elem.mat)    
    ϕ += elem.wgt[ii]δϕ×F
  end
  adiff.grad(ϕ)
end
# 
# functions for array of elements
# 
# ElTypes{T,I} = Union{CPElems{P,M,T,I} where {P,M}} where {T,I}
ElTypes{T,I} = CPElems{P,M,T,I} where {P,M} where {T,I}
function getϕ(elems::Vector{<:ElTypes},  u::Matrix{T}, d::Matrix{T}) where T
  nElems = length(elems)

  Φ = Vector{T}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getϕ(elems[ii], u[:,elems[ii].nodes], d[elems[ii].nodes])
  end

  # Φ = [getϕ(elem, adiff.D2(u[:,elem.nodes]), d[elem.nodes]) for elem in elems]
  return Φ
end
function getϕu(elems::Vector{<:ElTypes}, u::Matrix{T}, d::Matrix{T}) where T
  nElems = length(elems)
  N      = length(u[:,elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getϕu(elems[ii], u[:,elems[ii].nodes], d[elems[ii].nodes])
  end

  # Φ = [getϕ(elem, adiff.D2(u[:,elem.nodes]), d[elem.nodes]) for elem in elems]
  makeϕrKt(Φ, elems, u)
end
function getϕd(elems::Vector{<:ElTypes}, u::Matrix{T}, d::Matrix{T}) where T

  nElems = length(elems)
  N      = length(d[elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    nodes = elems[ii].nodes
    Φ[ii] = getϕd(elems[ii], u[:,nodes], d[nodes])
  end
  makeϕrKt(Φ, elems, d)
end
function getϕd!(elems::Vector{<:ElTypes}, u::Matrix{T}, d::Matrix{T}, ϕmax::Vector{Vector{T}}) where T

  nElems = length(elems)
  N      = length(d[elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    nodes = elems[ii].nodes
    Φ[ii] = getϕ!(elems[ii], u[:,nodes], adiff.D2(d[nodes]), ϕmax[ii])
  end
  makeϕrKt(Φ, elems, d)
end
function makeϕrKt(Φ::Array{adiff.D2{N,M,T}, 1} where {N,M}, 
                  elems::Vector{<:ElTypes{T,I}}, u) where {T,I}
  N  = length(u) 
  Nt = 0
  for ϕ in Φ
    Nt += length(ϕ.g.v)*length(ϕ.g.v)
  end

  II = zeros(I, Nt)
  JJ = zeros(I, Nt)  
  Kt = zeros(T, Nt)  
  r  = zeros(T, N)
  ϕ  = 0
  indxs  = LinearIndices(u)

  N1 = 1
  for (ii,elem) in enumerate(elems)    
    idxii     = indxs[:, elem.nodes][:]    
    ϕ        += adiff.val(Φ[ii]) 
    r[idxii] += adiff.grad(Φ[ii]) 
    nii       = length(idxii)
    Nii       = nii*nii
    oneii     = ones(I,nii)
    idd       = N1:N1+Nii-1
    II[idd]   = idxii * transpose(oneii)
    JJ[idd]   = oneii * transpose(idxii)
    Kt[idd]   = adiff.hess(Φ[ii])
    N1       += Nii
  end
  
  ϕ, r, dropzeros(sparse(II,JJ,Kt,N,N))
end
function makeresu(elems::Array{<:ElTypes,1},
                  u::Matrix{T} where T, d::Matrix{T} where T)
  Φ = [getϕ(elem, adiff.D1(u[:,elem.nodes]), d[elem.nodes]) for elem in elems]  
  makeϕr(Φ, elems, u)
end
function makeϕr(Φ, elems, u)
  N   = length(u) 
  r   = zeros(N)
  ϕ   = 0
  idx = LinearIndices(u)
  for (ii,elem) in enumerate(elems)    
    idxii     = idx[:, elem.nodes][:]    
    ϕ        += adiff.val(Φ[ii]) 
    r[idxii] += adiff.grad(Φ[ii]) 
  end
  ϕ, r
end
function getd(elem::CElems{P,M,T,I} where{M,I}, d0) where {P,T}
  d       = zero(T)
  N0, wgt = elem.N0, elem.wgt
  for ii=1:P
    d += wgt[ii]*(N0[ii]⋅d0)
  end  
  d/elem.V
end
function makeKt(Φ::Vector{adiff.D2{N,M,T}} where M, 
                elems::Vector{<:CElems{P,M,T,I}}  where M, u, d) where {P,N,T,I}
  
  @assert length(Φ)==length(elems) "length(Φ)!=length(elems)"

  NM     = length(u)
  oneii  = ones(I,N)
  N1     = N*N
  Ntot   = length(elems)*N1
  II     = Vector{I}(undef, Ntot)
  JJ     = Vector{I}(undef, Ntot)
  Kt     = Vector{T}(undef, Ntot)
  idxs   = LinearIndices(u)

  for (ii,elem) in enumerate(elems)
    d_el      = getd(elem, d[elem.nodes])
    idxii     = idxs[:, elem.nodes][:]
    idd       = (ii-1)*N1+1:ii*N1
    II[idd]   = idxii * transpose(oneii)
    JJ[idd]   = oneii * transpose(idxii)
    Kt[idd]   = (1-d_el)^2*adiff.hess(Φ[ii])
  end

  dropzeros(sparse(II,JJ,Kt,NM,NM))
end
# function for inertia and mass matrices
function getT(elem::C3DElems{P,M} where M,
              udot0::Matrix{T}) where {T,P}
  ϕ   = zero(T) 
  for ii=1:P
    N0 = elem.N0[ii]
    d  = [N0⋅udot0[1:3:end], N0⋅udot0[2:3:end], N0⋅udot0[3:3:end]]
    ϕ += elem.mat.mat.ρ*elem.wgt[ii]* (d⋅d)
  end
  ϕ
end
function getT(elems::Vector{<:ElTypes}, 
               udot::Matrix{T}) where T
  nElems = length(elems)
  # nDoFs  = size(udot,1)
  N      = length(udot[:,elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getT(elems[ii], adiff.D2(udot[:,elems[ii].nodes]))
  end

  # Φ = [getϕ(elem, adiff.D2(u[:,elem.nodes]), d[elem.nodes]) for elem in elems]
  makeϕrKt(Φ, elems, udot)
end

# getVd
function getVd(elem::CPElems{P}, d0::Vector{T}) where {T, P}
  Vd = zero(T)
  for ii=1:P
    Vd += elem.wgt[ii]elem.N0[ii]⋅d0
  end
  Vd
end
function getVd(elems::Vector{<:CPElems}, d::Matrix{T}) where T
  Vd = zero(T)
  for elem in elems
    Vd += getVd(elem, d[elem.nodes])
  end
  Vd
end
end

