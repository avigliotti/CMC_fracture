using WriteVTK, LinearAlgebra, SparseArrays, Printf, Random
using AbaqusReader, Logging
using FileIO, Dates

include("fe_tools.jl")
using .fe_tools


function find_pairs(nodes, set1, set2; bchk = false, dTol=1e-12)
  N1,N2 = length(set1),length(set2)
  @assert N1==N2 @sprintf("length(set1)=%i!=%i=length(set2)",N1,N2)

  a = let
    cg1  = sum(nodes[set1])/N1
    cg2  = sum(nodes[set2])/N2
    cg2-cg1
  end

  pairs = Vector{Pair{Int64,Int64}}(undef, N1)
  for ii1 in 1:N1
    node1 = nodes[set1[ii1]]
    dd    = [norm(a + node1 - nodes[set2[jj]]) for jj in 1:N2]
    ii2   = argmin(dd)
    bchk && @assert dd[ii2] ≤ dTol @sprintf("mininum distance out of tolerance: %.3f ≰ %.3f",
                                            dd[ii2], dTol)
    pairs[ii1] = set1[ii1]=>set2[ii2]
  end

  a, pairs
end
function makeBEqs(all_pairs, nNodes, T=Float64)
  A = begin
    rmrow(A, ii) = A[1:size(A)[1] .!= ii, :]
    A = spzeros(T, size(all_pairs)[1], nNodes)
    for (ii, pair) = enumerate(all_pairs)
      A[ii, pair[1]] = A[ii, pair[2]] = 1
    end
    for ii=1:nNodes
      id_rows = sort(findall(A[:,ii].!=0))
      nrows   = length(id_rows)
      if nrows>1
        for jj=nrows:-1:2
          A[id_rows[1],:] = ((A[id_rows[1],:].==1) .| (A[id_rows[jj],:].==1))
          A = rmrow(A, id_rows[jj])
        end
      end
    end  
    A
  end
  B = begin
    nNodes = size(A)[2]
    q      = [sum(A[:,ii]) for ii=1:nNodes] 
    ifree  = findall(q .==0)
    nfree  = length(ifree)
    B      = spzeros(T, nfree, nNodes)
    for (ii,idx) in enumerate(ifree)
      B[ii, idx] = 1
    end  
    B
  end
  return vcat(A,B)
end
function makeB0(ufree::BitArray{N} where N, T)
  #   nDoFstot = length(ufree)
  N    = sum(ufree)
  idxx = findall(ufree[:])
  sparse(idxx, 1:N, ones(N), length(ufree), N)
end
function makeB0(BEqs::SparseMatrixCSC{TF,TI}; nDoFs=3) where {TF,TI}
  nEqs,nNodes = size(BEqs)
  nDoFstot    = nNodes*nDoFs 
  I           = zeros(TI, nDoFstot)
  J           = zeros(TI, nDoFstot)
  V           = zeros(TF, nDoFstot)

  for ii=1:nNodes
    iirows = (ii-1)*nDoFs
    for qq = BEqs[:,ii].nzind
      iicols = (qq-1)*nDoFs
      for jj = 1:nDoFs
        I[iirows+jj] = iirows+jj
        J[iirows+jj] = iicols+jj
        V[iirows+jj] = one(TF)
      end
    end
  end  
  sparse(I,J,V)
end
function makeBa(pairs, nNodes,
                T=Float64;
                nDoFsu=length(pairs),
                nDoFsω = 0)

  nDoFs = nDoFsu + nDoFsω
  ndirs = length(pairs)
  Ba    = spzeros(T, nNodes*nDoFs, nDoFsu*ndirs)

  for (jj, pairs) in enumerate(pairs)
    for pair in pairs
      iia = (jj-1)*nDoFsu
      ii1 = (pair[1]-1)*nDoFs
      ii2 = (pair[2]-1)*nDoFs
      for ii=1:nDoFsu
        Ba[ii1+ii, iia+ii] = -1/2
        Ba[ii2+ii, iia+ii] = 1/2
      end
    end
  end
  dropzeros!(Ba)
end
#=
#        e₁₁, e₂₂, e₃₃, e₂₃, e₁₃, e₁₂
Bϵ(a) = [a[1] 0 0 0 a[3] a[2];
         0 a[2] 0 a[3] 0 a[1];
         0 0 a[3] a[2] a[1] 0;]
#         e₁₁, e₂₂, e₁₂
Bϵ2D(a) = [a[1] 0 a[2];
           0 a[2] a[1]]
=#
"""
```
vec2tens(x) = [x[1] x[6] x[5]; x[6] x[2] x[4]; x[5] x[4] x[3]]
```
puts the columns of Dϵ into tensor form
"""
vec2tens(x) = [x[1] x[6] x[5]; x[6] x[2] x[4]; x[5] x[4] x[3]]
#
# make the model functions
function make_the_2Dmodel(sModelName, fiber_mat, matrix_mat, 
                          θ=0, dTol_ai=1e-6,
                          bfullint=true, bsmall=true)

  nDoFs    = 2
  # rotation matrix for the model, the sign is reversed in order for θ to point
  # to the direction of the applied strain
  M        = [cos(θ) sin(θ); -sin(θ) cos(θ)]    

  # load model from input file
  model = with_logger(Logging.NullLogger()) do
    AbaqusReader.abaqus_read_mesh(sModelName*".inp")
  end

  nNodes      = length(model["nodes"])
  nDoFstot    = nDoFs*nNodes
  nodes       = [ model["nodes"][ii][1:nDoFs] for ii in 1:nNodes ]
  elements    = model["elements"]
  node_sets   = model["node_sets"]
  elem_sets   = model["element_sets"]
  matrixnodes = node_sets["matrix"]
  hasfibers   = haskey(elem_sets, "fibers")

  @show length(node_sets["left"]),   length(node_sets["right"])
  @show length(node_sets["bottom"]), length(node_sets["top"])

  a1,a1_pairs = find_pairs(nodes, node_sets["left"], node_sets["right"])
  a2,a2_pairs = find_pairs(nodes, node_sets["bottom"], node_sets["top"])


  BEqs = makeBEqs(vcat(a1_pairs, a2_pairs), nNodes)
  B0   = makeB0(BEqs, nDoFs=2)  
  B0   = dropzeros!(B0)  
  B0d  = dropzeros!(makeB0(BEqs, nDoFs=1))  
  #
  a1[:] .= M*a1 
  a2[:] .= M*a2 
  #
  Bϵ = let
    #         e₁₁, e₂₂, e₁₂
    Beps(a) = [a[1] 0 a[2];
               0 a[2] a[1]]
    Ba   = makeBa((a1_pairs, a2_pairs), nNodes)
    Bϵ   = Ba*vcat(Beps(a1), Beps(a2)) |> sparse
    dropzeros!(Bϵ)
  end
  #
  #   constructs elements
  #
  @show hasfibers
  println("\n constructing elements ... "); flush(stdout)
  @time begin
    GP = if bfullint
      ((-0.577350269189626, 1.0), (0.577350269189626, 1.0))
    else
      ((0.0, 2.0), )
    end

    if hasfibers
      fibers_elems = [if length(elements[id])==4
                        Elements.QuadP(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                       mat=fiber_mat, GP=GP) 
                      else
                        Elements.TriaP(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                       mat=fiber_mat) 
                      end
                      for id in model["element_sets"]["fibers"]]
      Vol_fibers   =  sum([item.V for item in fibers_elems])
    else
      fibers_elems = []
      Vol_fibers   = 0.0
    end

    matrix_elems = [if length(elements[id])==4
                      Elements.QuadP(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                     mat=matrix_mat, GP=GP) 
                    else
                      Elements.TriaP(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                     mat=matrix_mat) 
                    end
                    for id in model["element_sets"]["matrix"]]
    Vol_matrix  = sum([item.V for item in matrix_elems])

    elems = if hasfibers 
      vcat(fibers_elems, matrix_elems)
    else
      matrix_elems
    end
  end
  println(" ... done\n"); flush(stdout)

  return nodes, elems, node_sets, elem_sets, B0, Bϵ, B0d , (a1,a2)
end
function make_the_3Dmodel(sModelName, fiber_mat, matrix_mat,
                          θ=0, ψ=0, ζ=0, dTol_ai=1e-6, 
                          bfullint=true, bsmall=true )

  nDoFs    = 3
  # rotation matrix for the model, the sign is reversed in order for θ to point
  # to the direction of the applied strain
  M = [1.0    0.0     0.0;
       0.0    cos(ψ)  sin(ψ); 
       0.0   -sin(ψ)  cos(ψ)] * 
  [cos(ζ)   0.0     sin(ζ); 
   0.0      1.0     0.0;
   -sin(ζ)  0.0     cos(ζ)] * 
  [ cos(θ)  sin(θ)  0.0;
   -sin(θ)  cos(θ)  0.0; 
   0.0      0.0     1.0] 

  @show M

  # load model from input file
  model = with_logger(Logging.NullLogger()) do
    AbaqusReader.abaqus_read_mesh(sModelName*".inp")
  end

  nNodes      = length(model["nodes"])
  nDoFstot    = nDoFs*nNodes
  nodes       = [ model["nodes"][ii][1:nDoFs] for ii in 1:nNodes ]
  elements    = model["elements"]
  node_sets   = model["node_sets"]
  elem_sets   = model["element_sets"]
  matrixnodes = node_sets["matrix"]
  hasfibers   = haskey(elem_sets, "fibers")

  #=
  @show length(node_sets["left"]),   length(node_sets["right"])
  @show length(node_sets["bottom"]), length(node_sets["top"])
  @show length(node_sets["front"]),  length(node_sets["back"])


  a1,a1_pairs = find_pairs(nodes, node_sets["left"],   node_sets["right"])
  a2,a2_pairs = find_pairs(nodes, node_sets["bottom"], node_sets["top"])
  a3,a3_pairs = find_pairs(nodes, node_sets["front"],  node_sets["back"])


  BEqs = makeBEqs(vcat(a1_pairs, a2_pairs, a3_pairs), nNodes)
  B0   = makeB0(BEqs)  
  B0   = dropzeros!(B0)  
  B0d  = dropzeros!(makeB0(BEqs, nDoFs=1))  

  a1[:] .= M*a1 
  a2[:] .= M*a2 
  a3[:] .= M*a3 
  Bϵ = let
    #        e₁₁, e₂₂, e₃₃, e₂₃, e₁₃, e₁₂
    Beps(a) = [a[1] 0 0 0 a[3] a[2];
               0 a[2] 0 a[3] 0 a[1];
               0 0 a[3] a[2] a[1] 0;]
    Ba   = makeBa((a1_pairs, a2_pairs, a3_pairs), nNodes)
    Bϵ   = Ba*vcat(Beps(a1), Beps(a2), Beps(a3)) |> sparse
    dropzeros!(Bϵ)
  end
  =#

  (B0, B0d, Bϵ, (a1,a2,a3)) = make_B_matrices(model, nDoFs, M)
  #
  #   constructs elements
  #
  @show hasfibers
  println("\n constructing elements ... ")
  @time begin
    # drnd(x) = x*(1+fmat_noise*randn())
    # determine wheter construct full or reduced integration elements
    GP = if bfullint
      ((-0.577350269189626, 1.0), (0.577350269189626, 1.0))
    else
      ((0.0, 2.0), )
    end

    if hasfibers
      fibers_elems = [if length(elements[id])==8
                        Elements.Hex08P(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                        mat=fiber_mat)
                      elseif length(elements[id])==4
                        Elements.Tet04P(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                        mat=fiber_mat)
                      elseif length(elements[id])==6
                        Elements.Wdg06P(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                        mat=fiber_mat)
                      else
                        error("unknown element with ", length(elements[id]), " nodes")
                      end
                      for id in model["element_sets"]["fibers"]]
      Vol_fibers   =  sum([item.V for item in fibers_elems])
    else
      fibers_elems = []
      Vol_fibers   = 0.0
    end

    matrix_elems = [if length(elements[id])==8
                      Elements.Hex08P(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                      mat=matrix_mat) 
                    elseif length(elements[id])==4
                      Elements.Tet04P(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                      mat=matrix_mat) 
                    elseif length(elements[id])==6
                      Elements.Wdg06P(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                      mat=matrix_mat)
                    else
                      error("unknown element with ", length(elements[id]), " nodes")
                    end
                    for id in model["element_sets"]["matrix"]]
    Vol_matrix  = sum([item.V for item in matrix_elems])

    elems = (fibers_elems, matrix_elems)
  end
  
  println(" ... done")

  return nodes, elems, node_sets, elem_sets, B0, Bϵ, B0d, (a1,a2,a3) 
end
function make_B_matrices(model, nDoFs=3, M=I)

  nNodes      = length(model["nodes"])
  nodes       = [ model["nodes"][ii][1:nDoFs] for ii in 1:nNodes ]
  node_sets   = model["node_sets"]

  @show length(node_sets["left"]),   length(node_sets["right"])
  @show length(node_sets["bottom"]), length(node_sets["top"])
  @show length(node_sets["front"]),  length(node_sets["back"])


  a1,a1_pairs = find_pairs(nodes, node_sets["left"],   node_sets["right"])
  a2,a2_pairs = find_pairs(nodes, node_sets["bottom"], node_sets["top"])
  a3,a3_pairs = find_pairs(nodes, node_sets["front"],  node_sets["back"])


  BEqs = makeBEqs(vcat(a1_pairs, a2_pairs, a3_pairs), nNodes)
  B0   = makeB0(BEqs)  
  B0   = dropzeros!(B0)  
  B0d  = dropzeros!(makeB0(BEqs, nDoFs=1))  

  a1[:] .= M*a1 
  a2[:] .= M*a2 
  a3[:] .= M*a3 
  Bϵ = let
    #        e₁₁, e₂₂, e₃₃, e₂₃, e₁₃, e₁₂
    Beps(a) = [a[1] 0 0 0 a[3] a[2];
               0 a[2] 0 a[3] 0 a[1];
               0 0 a[3] a[2] a[1] 0;]
    Ba   = makeBa((a1_pairs, a2_pairs, a3_pairs), nNodes)
    Bϵ   = Ba*vcat(Beps(a1), Beps(a2), Beps(a3)) |> sparse
    dropzeros!(Bϵ)
  end

  return (B0, B0d, Bϵ, (a1,a2,a3))
end
function make_the_model(sJLD2FileName::String)

  println("reading data from :\n\t", sJLD2FileName)

  svars = ("sModelName", "nDoFs", "θ", "ψ", "ζ", "dTol_ai", 
           "bfullint", "bsmall", "fiber_mat", "matrix_mat")

  (sModelName, nDoFs,
   θ, ψ, ζ, dTol_ai, 
   bfullint,
   bsmall, fiber_mat, 
   matrix_mat) = FileIO.load(sJLD2FileName, svars)

  if nDoFs==3
    nodes, elems, node_sets, 
    elem_sets, B0, Bϵ, B0d, ai  = make_the_3Dmodel(sModelName, fiber_mat, matrix_mat, 
                                               θ, ψ, ζ, dTol_ai, 
                                               bfullint, bsmall)
  elseif nDoFs==2
    nodes, elems, node_sets, 
    elem_sets, B0, Bϵ, B0d, ai  = make_the_2Dmodel(sModelName, fiber_mat, matrix_mat, 
                                               θ, dTol_ai, 
                                               bfullint, bsmall)
  else
    error("wrong number of nDoFs: ", nDoFs)
  end

  return (nodes, elems, B0, Bϵ)
end
#
# functions for writing Paraview files
function writeVTKstate(sFileName,
                       nodes::Array{Array{D,1},1} where D<:Number, 
                       elems::Vector{E},
                       u = []; 
                       pvd = nothing, 
                       elemsprop = Dict{String,Any}(),
                       nodesprop = Dict{String,Any}(),
                       ii = 0, 
                       r0 = zeros(size(nodes[1])),
                       bdef = false,
                       bcenter = true) where E<:CPElems

  # Determine cell type based on element properties
  N = length(elems[1].nodes)
  cellType = determine_cell_type(E, N)

  # Initialize node coordinates
  nNodes = length(nodes)
  nDoFs = length(nodes[1])
  points = calculate_points(nodes, u, r0, bdef, bcenter, nNodes, nDoFs)

  # Adjust file name if part of a series
  if !isnothing(pvd)
    sFileName = @sprintf("%s_%03i", sFileName, ii)
  end

  # Create mesh cells and write to VTK
  cells = [WriteVTK.MeshCell(cellType, elem.nodes) for elem in elems]
  WriteVTK.vtk_grid(sFileName, points, cells) do vtkobj
    write_vtk_data(vtkobj, elemsprop, nodesprop)
    if !isnothing(pvd)
      pvd[ii] = vtkobj
    end
  end

  return sFileName
end
function get_ϵT(elems::Vector{<:CPElems{P,<:PhaseField} where P},
                u::Array{N,2} where N)
  Dict("\$\\epsilon_T\$"=>[get_ϵT(elem, u[:,elem.nodes]) for elem in elems])
end
function get_ϵT(elem::CPElems{P,<:PhaseField},
                   u0::Array{N,2} where N,) where P
  wgt  = elem.wgt
  F    = [Elements.getF(elem, u0, ii)                 for ii=1:P]
  ϵn   = sum([wgt[ii]*(transpose(F[ii])+F[ii]-2I)/2   for ii=1:P])/elem.V
  C    = [transpose(F[ii])F[ii]                       for ii=1:P]
  λp   = sum([wgt[ii]*sqrt.(svdvals(C[ii]))           for ii=1:P])/elem.V
  ϵT   = maximum(λp) - minimum(λp)
end
# Helper function to determine the VTK cell type based on element type and number of nodes
function determine_cell_type(E, N)
  if N == 4 && E <: C3DElems
    return VTKCellTypes.VTK_TETRA
  elseif N == 8 && E <: C3DElems
    return VTKCellTypes.VTK_HEXAHEDRON
  elseif N == 6 && E <: C3DElems
    return VTKCellTypes.VTK_WEDGE
  else
    error("Element type ", E, " with ", N, " nodes not recognized")
  end
end

# Helper function to calculate point positions with optional deformation
function calculate_points(nodes, u, r0, bdef, bcenter, nNodes, nDoFs)
  points = zeros(nDoFs, nNodes)
  if bdef
    u_cg = bcenter ? sum([u[:, i] for i in 1:nNodes]) / nNodes : zeros(nDoFs)
    for i in 1:nNodes
      points[:, i] = nodes[i] + r0 + u[:, i] - u_cg
    end
  else
    for i in 1:nNodes
      points[:, i] = nodes[i] + r0
    end
  end
  return points
end

# Helper function to write VTK data properties
function write_vtk_data(vtkobj, elemsprop, nodesprop)
  if !isempty(elemsprop)
    for spropname in keys(elemsprop)
      WriteVTK.vtk_cell_data(vtkobj, elemsprop[spropname], spropname)
    end
  end
  if !isempty(nodesprop)
    for spropname in keys(nodesprop)
      WriteVTK.vtk_point_data(vtkobj, nodesprop[spropname], spropname)
    end
  end
end

