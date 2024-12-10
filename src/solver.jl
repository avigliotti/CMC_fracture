using Pardiso

function run_phase_field_qspso(;
                               sModelName = "cm_wdg1x1x1nL20_r500L1196lc0500lcf0750",
                               θ          = 0.0,
                               ψ          = 0.0,    # for 3D only
                               ζ          = 0.0,    # for 3D only
                               nSteps     = 300,
                               ϵM0        = [1, NaN, NaN, 
                                             NaN, NaN, NaN]*1e-3,
                               sPost      = "_eps1xxxxx",
                               bfullint   = true,    # full or reduced integration
                               bsmall     = true,    # small strain
                               fiber_mat  = PhaseField(0.500, 0.049025, Hooke(588.0, 0.27, 1.0, true)),
                               matrix_mat = PhaseField(0.500, 0.001195, Hooke(430.0, 0.17, 1.0, true)),
                               λV         = 1.0,      # scale factor for potential energy
                               λT         = 1.e-1,    # scale factor for kinetic energy
                               dTol_ai    = 1e-6,     # tolerance in node pairing
                               dTolu      = 1e-3,     # convergence tolerance on u 
                               dTold      = 1e-5,     # convergence tolerance on u 
                               maxiter    = 3,        # maximum global iterations
                               maxiteru   = 5,        # maximum iterations on u
                               iterupdt   = maxiteru, # number of failediteration before updating B1KtB1ff
                               maxδdTol   = 5e-2,     # tolerance in maximum damage field change for convergence check
                               sPath      = pwd(),    # path for writing output files
                               threshold  = 0.99,     # threshold for removing elements
                               bdef       = false,    # export deformed shape in paraview
                               Nsave      = 6,        # number of configurations to save
                               ikeep      = Inf,      # keep config every 
                               bwrite_vtk = true,     # write the vtk files 
                               bwritemodel= false,    # save the model in the binary data file
                               nwritevtu  = NaN,      # write vtu file after x iterations 
                               Nupdtmax   = 10,       # maximum number of steps between matrix update
                               Nupdtmin   = 3,        # minimum number of steps between matrix update
                               maxnorm    = 1e3,      # abnormal residual produces termination
                               bmake_χM   = true,     # calculate the initial stiffness
                               badaptive  = true,     # do adaptive step length
                               smeshdir   = "mesh_files",   # directory with mesh files 
                               sjld2dir   = "jld2_files",   # directory for jld2 files 
                               svtkdir    = "vtk_files",    # directory for vtk files 
                              )

  println()
  @show t_start = Dates.now()
  @show ϵM0
  @show θ, ψ, ζ
  @show λV, λT
  @show maxiter, maxiteru
  @show iterupdt, maxδdTol
  @show dTolu, dTold, Nupdtmax, Nupdtmin
  @show nwritevtu

  @show nSteps = round(Int, nSteps)
  @show Δt     = 1/(nSteps-1)
  @show badaptive

  # some local functions
  function make_χM(maxiterχM=5, dTolχM=1e-5) 

    Ktu = makeKt(Φ, elems, zeros(nDoFs,nNodes), zeros(1,nNodes))
    ps  = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    fix_iparm!(ps, :N)

    B0tKtuB0 = get_matrix(ps, transpose(B0)*(λT*MM+λV*Ktu)*B0, :N)

    u0ϵ  = zeros(size(B0,2), size(Bϵ,2))
    updt = zeros(size(B0,2), size(Bϵ,2))
    Dϵ   = B0*u0ϵ + Bϵ
    res  = transpose(B0)*Ktu*Dϵ
    @printf("ii: %2i, maximum(abs.(res)): % .3e\n", 0, maximum(abs.(res))...)

    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, B0tKtuB0, res)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, B0tKtuB0, res)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)

    for ii=1:maxiterχM
      pardiso(ps, updt, B0tKtuB0, res)
      u0ϵ -= updt
      Dϵ   = B0*u0ϵ + Bϵ
      res  = transpose(B0)*Ktu*Dϵ
      @printf("ii: %2i, maximum(abs.(res)): % .3e\n", ii, maximum(abs.(res))...)
      flush(stdout)
      maximum(abs.(updt)) ≤ dTolχM && break    
    end

    set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)

    Matrix(transpose(Dϵ)*Ktu*Dϵ)/Vol
  end

  function write_vtu_file()
    paraview_collection(spvdFileName; append=true) do pvd
      println(" state written to:\n ",
              writeVTKstate(spvdFileName, nodes, elems, u, 
                            pvd=pvd, 
                            elemsprop=get_ϵT(elems, u),
                            nodesprop=Dict("d"=>d), 
                            bdef=false, bcenter=false,ii=ii), ".vtu\n")
    end
  end

  println("\n running with ", Threads.nthreads(), " thread(s)"); flush(stdout)
  sFileName = "qs_"*sModelName*sPost
  println("\n sFileName: ", sFileName, "\n"); flush(stdout)

  if !isdir(joinpath(sPath, svtkdir))
    mkdir(joinpath(sPath, svtkdir))
  end
  spvdFileName  = joinpath(sPath, svtkdir, sFileName)
  if !isnan(nwritevtu)
    paraview_collection(spvdFileName) do pvd; end
  end

  # make the model and B-matrices
  if length(ϵM0)==6
    @show nDoFs = 3
    nodes, elems, node_sets, 
    elem_sets, B0, Bϵ, B0d, ai  = make_the_3Dmodel(joinpath(smeshdir, sModelName),
                                                   fiber_mat, matrix_mat, 
                                                   θ, ψ, ζ, dTol_ai, 
                                                   bfullint, bsmall)
  elseif length(ϵM0)==3
    @show nDoFs = 2
    nodes, elems, node_sets, 
    elem_sets, B0, Bϵ, B0d, ai  = make_the_2Dmodel(joinpath(smeshdir, sModelName),
                                                   fiber_mat, matrix_mat,
                                                   θ, dTol_ai,
                                                   bfullint, bsmall)
  else
    error("wrong ϵM0!\n ϵM0 = ", nDoFs)
  end

  @show typeof(fiber_mat)
  @show typeof(matrix_mat)

  ϕcf  = fiber_mat.Gc/4fiber_mat.l0     # critical energy for fibers
  ϕcm  = matrix_mat.Gc/4matrix_mat.l0   # critical energy for matrix

  @show nElems_f    = length(elems[1])
  @show nElems_m    = length(elems[2])
  Vols_fibers = map(elem->elem.V, elems[1])
  Vols_matrix = map(elem->elem.V, elems[2])
  Vol_fibers  = sum(elem->elem.V, elems[1])
  Vol_matrix  = sum(elem->elem.V, elems[2])
  Vol         = Vol_matrix + Vol_fibers

  elems       = vcat(elems[1], elems[2])
  B1          = hcat(B0, Bϵ)

  nElems      = length(elems)
  #
  @show Vol_matrix
  @show Vol_fibers
  @show Vol
  @show nElems    = length(elems)
  @show nNodes    = length(nodes)
  @show nDoFstot  = nNodes*nDoFs
  #
  # prepare arrays
  #
  nDoFstot, nDoFs0 = size(B0)
  bϵfree      = isnan.(ϵM0)
  iϵcnst      = findall(.!bϵfree)
  bfree       = vcat(trues(nDoFs0), bϵfree)
  ifree       = findall(bfree)
  icnst       = findall(.!bfree)
  nϵ          = length(ϵM0)

  u           = zeros(nDoFs, nNodes)
  u1          = zeros(nDoFs0+nϵ)
  d           = zeros(1, nNodes)
  d0          = zeros(size(B0d,2))

  # make the "mass" matrix
  _, _, MM    = getT(elems, u)
  B0dtB0d     = let 
    ps      = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    fix_iparm!(ps, :N)
    B0dtB0d = get_matrix(ps, transpose(B0d)*B0d, :N)

    set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)

    B0dtB0d
  end

  println("\n making hessians ... "); flush(stdout)
  @time Φ     = [getϕu(elem, u[:,elem.nodes], d[elem.nodes]) for elem in elems]   
  flush(stdout)


  if bmake_χM
    println(" calculating χM: "); flush(stdout)
    @time χM = make_χM() 
    print("\n χM = ")
    display(round.(χM, digits=2))
    flush(stdout)
  end
  _, _, Ktd    = getϕd(elems, zeros(nDoFs,nNodes), zeros(1,nNodes))
  B0dtKtdB0d   = transpose(B0d)*Ktd*B0d
  # end
  println(" ... done"); flush(stdout)

  ## the main loop
  #
  dprev     = copy(d)
  uprev     = copy(u)
  allus     = []
  Vdprev    = getVd(elems, d)/Vol
  cntupdt   = 0
  ru        = zeros(nDoFs*nNodes)
  normruii  = normrdii = 0.0

  t         = fill(NaN, nSteps)
  ϵM        = fill(NaN, length(ϵM0), nSteps)
  σM        = fill(NaN, length(ϵM0), nSteps)
  g0        = fill(NaN, nSteps)
  Vd        = fill(NaN, nSteps)
  gm        = fill(NaN, nSteps) 
  gf        = fill(NaN, nSteps) 
  γ         = fill(NaN, nSteps) 
  Vd        = fill(NaN, nSteps) 
  Vddot     = fill(NaN, nSteps) 
  normru    = fill(NaN, nSteps) 
  normrd    = fill(NaN, nSteps) 
  Δd        = fill(NaN, 2, nSteps) 
  rdii      = fill(NaN, 2, nSteps) 

  function make_B1KtB1ff(ps, u, d)

    # set_phase!(ps, Pardiso.RELEASE_ALL)
    # pardiso(ps)
    Ktu      = makeKt(Φ, elems, u, d)
    res      = transpose(B1[:,ifree])*Ktu*u[:]
    B1KtB1ff = get_matrix(ps, transpose(B1[:,ifree])*(λT*MM+λV*Ktu)*B1[:,ifree], :N)

    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, B1KtB1ff, res)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, B1KtB1ff, res)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)

    return B1KtB1ff
  end

  psu     = MKLPardisoSolver()
  set_matrixtype!(psu, Pardiso.REAL_SYM_POSDEF)
  fix_iparm!(psu, :N)

  println("\t==== finding B1KtB1ff ... "); flush(stdout)
  @time B1KtB1ff = make_B1KtB1ff(psu, u, d) 
  println("\t\t ... done "); flush(stdout)

  println("\n starting ... "); flush(stdout)
  t0       = Base.time_ns()
  totiters = 0
  ii       = 0
  LF       = 0
  try
    while ii < nSteps && LF < 1
      if ii % nwritevtu == 0
        write_vtu_file()
      end
      ii += 1
      #
      # step loop
      stept0     = Base.time_ns()
      u1[icnst]  = ϵM0[iϵcnst]*LF
      u[:]       = B1*u1
      σMold      = transpose(Bϵ)*ru/Vol
      println(" σM    = [", join([@sprintf("%+3.2e, ", x) for x in σMold[iϵcnst]]), "]")
      iter,maxδd = 0, Inf
      while iter ≤ maxiter && maxδd > maxδdTol # 1e0 #5e-3 
        t1       = Base.time_ns()
        normruii, iteru  = update_u!(psu, elems, Φ, B1KtB1ff, ifree, 
                                     ru, u, u1, d, 
                                     B1, dTolu, maxiteru)
        σMnew    = transpose(Bϵ)*ru/Vol
        ΔσM      = σMnew-σMold
        println(" ΔσM   = [", join([@sprintf("%+3.2e, ", x) for x in ΔσM[iϵcnst]]), "]")
        σMold[:] = σMnew[:]

        dold        = deepcopy(d)
        rdii[:,ii] .= update_d!(elems, u, d, B0d, d0)
        maxδd       = maximum(abs.(d-dold))

        iterutime = (Base.time_ns()-t1)/1e9
        @printf("\n step iter:%-2i, |ru|:%.3e, maxδd:%.3e, in %.2f sec\n\n", 
                iter, normruii, maxδd, iterutime )
        totiters += 1
        if iteru≥iterupdt 
          if cntupdt≥Nupdtmin
            println("\t==== updating B1KtB1ff ... ")
            @time begin
              B1KtB1ff = make_B1KtB1ff(psu, u, d) 
              GC.gc()
            end
            println("\t\t ... done "); flush(stdout) 
            cntupdt   = 0
          else
            @printf("\t---- updating B1KtB1ff in %3i steps \n", 
                    Nupdtmin-cntupdt); flush(stdout)
          end
        end

        iter     +=1
        flush(stdout)
      end

      if iter == maxiter
        @printf("\t#### not converged, continuing ####\n"); flush(stdout)
      end
      any(isnan.(u[:])) && error("NaN in u detected")
      any(isnan.(d[:])) && error("NaN in d detected")
      if any([normruii>maxnorm, normrdii>maxnorm]) 
        println("abnormal residual, quitting.")
        u[:]    .= uprev[:]
        d[:]    .= dprev[:]
        break
      end
      # ii         += 1

      Δd[:,ii]   .= extrema(d[:]-dprev[:])
      dprev[:]    = d[:]
      uprev[:]    = u[:]

      cntupdt    += 1
      t[ii]       = LF
      Vd[ii]      = getVd(elems, d)/Vol
      Vddot[ii]   = (Vd[ii]-Vdprev) # /Δt
      Vdprev      = Vd[ii]
      σM[:,ii]    = transpose(Bϵ)*ru/Vol
      ϵM[:,ii]    = u1[nDoFs0+1:end]
      normru[ii]  = normruii
      normrd[ii]  = normrdii

      fibers_gs   = getϕ(elems[1:nElems_f], u, d)
      matrix_gs   = getϕ(elems[nElems_f+1:end], u, d)
      gf[ii]      = maximum(ii->fibers_gs[ii]/Vols_fibers[ii], 1:nElems_f) 
      gm[ii]      = maximum(ii->matrix_gs[ii]/Vols_matrix[ii], 1:nElems_m) 
      g0[ii]      = sum(fibers_gs) + sum(matrix_gs)

      if badaptive
        ΔLF  = let κ=sqrt(max(gm[ii]/ϕcm, gf[ii]/ϕcf))
          LF*(1/κ-1)
        end
        LF  += ΔLF > Δt ? ΔLF/2 :  Δt
      else
        LF  += Δt
      end

      if ii%ikeep==0
        push!(allus, (copy(u), copy(d), ii))
      end
      eltime    = (Base.time_ns()-t0)/1e9/60
      steptime  = (Base.time_ns()-stept0)/1e9
      @printf("\nstep:%-3i, LF:%.3f, Vd:%-6.4f, Vddot:%.2e, in %-5.1fsec, eltime:%-6.2fmin, ETA:%-6.2fmin\n\n", 
              ii, LF, Vd[ii], Vddot[ii], steptime, eltime, (1-LF)/Δt*steptime/60); flush(stdout)
    end
    @printf("\n all done in %5.1f mins, with %-4i steps, %-4i total iterations\n", 
            (Base.time_ns()-t0)/1e9/60, ii, totiters); flush(stdout)
    #
  catch e
    u[:]    .= uprev[:]
    d[:]    .= dprev[:]

    error_msg = sprint(showerror, e)
    st        = sprint((io,v) -> show(io, "text/plain", v), 
                       stacktrace(catch_backtrace()))
    @warn "Trouble doing things:\n$(error_msg)\n$(st)" 
    # println(" quitting." ); flush(stdout)
    @printf("\n*all done in %5.1f mins, with %-4i steps, %-4i total iterations\n", 
            (Base.time_ns()-t0)/1e9/60, ii, totiters); flush(stdout)
  end
  #
  set_phase!(psu, Pardiso.RELEASE_ALL)
  pardiso(psu)

  # write the last paraview snapshot
  println(" state written to:\n ", 
          writeVTKstate(spvdFileName, nodes, elems, u, 
                        elemsprop=get_ϵT(elems, u),
                        nodesprop=Dict("d"=>d), 
                        bdef=true, bcenter=false), ".vtu\n")    
  # write_vtu_file()
  flush(stdout)

  t        = t[1:ii]     
  ϵM       = ϵM[:,1:ii]
  σM       = σM[:,1:ii] 
  g0       = g0[1:ii]
  Vd       = Vd[1:ii]
  Δd       = Δd[:,1:ii]
  rdii     = rdii[:,1:ii]
  gm       = gm[1:ii]
  gf       = gf[1:ii]
  γ        = γ[1:ii]
  normru   = normru[1:ii]
  normrd   = normrd[1:ii]
  Vddot    = Vddot[1:ii]

  t_end      = Dates.now()
  saved_vars = Dict("sModelName"=>sModelName, "sPost"=>sPost,
                    "sFileName"=>sFileName,   "sPath"=>sPath,
                    "normru"=>normru,         "normrd"=>normrd,
                    "maxiter"=>maxiter,       "maxiteru"=>"maxiteru", 
                    "dTolu"=>dTolu, "dTold"=>dTold,
                    "Nupdtmax"=>Nupdtmax,     "Nupdtmin"=>Nupdtmin, 
                    "bsmall"=>bsmall, "maxδdTol"=>maxδdTol,
                    "bfullint"=>bfullint,     "λV"=>λV,     "λT"=>λT,
                    "Vol"=>Vol,   "Vol_fibers"=>Vol_fibers, "Vol_matrix"=>Vol_matrix, 
                    "ai"=>ai,     "nDoFs"=>nDoFs,   "ϵM0"=>ϵM0, 
                    "θ"=>θ,       "ψ"=>ψ,           "ζ"=>ζ,   
                    "dTol_ai"=>dTol_ai,       "bϵfree"=>bϵfree,
                    "ϵM"=>ϵM,     "σM"=>σM,   "Vd"=>Vd, 
                    "Δd"=>Δd,     "rdii"=>rdii, "χM"=>χM,     "Cm"=>inv(χM),
                    "g0"=>g0,     "gm"=>gm,     "gf"=>gf, "γ"=>γ, 
                    "fiber_mat"=>fiber_mat,     "matrix_mat"=>matrix_mat,
                    "t"=>t, "nSteps"=>nSteps, "Δt"=>Δt,
                    "t_start"=>t_start, "t_end"=>t_end, "Vddot"=>Vddot,
                    "allus"=>allus)

  if bwritemodel
    merge!(saved_vars, Dict("elems"=>elems, "nodes"=>nodes, "B0"=>B0, "Bϵ"=>Bϵ, 
                            "node_sets"=>node_sets, "elem_sets"=>elem_sets))
  end

  if !isdir(joinpath(sPath, sjld2dir))
    mkdir(joinpath(sPath, sjld2dir))
  end
  @time let sFileName = joinpath(sPath,sjld2dir,sFileName*".jld2")
    println("\n saving binary data to \n\t", sFileName, " ..."); flush(stdout)
    FileIO.save(sFileName, saved_vars)
  end

  println(" exiting. ")
  @show Dates.now()
  saved_vars
end

function update_u!(ps, elems, Φ, B1KtB1ff, ifree, ru, u, u1, d, 
                   B1, dTolu, maxiter)

  iter    = 0
  N       = length(ifree)
  normres = normupdt= NaN
  Ktu     = makeKt(Φ, elems, u, d)
  ru[:]   = Ktu*u[:]
  res     = transpose(B1[:,ifree])*Ktu*u[:]
  updt    = similar(res)

  println("\tu-iter\tnormresu\tnormupdtu")
  while true 
    normres = maximum(abs.(res))
    if iter≥maxiter || normres≤dTolu
      @printf("\t%i\t%.2e\t%.2e\n", 
              iter, normres, normupdt)
      break
    end

    pardiso(ps, updt, B1KtB1ff, res)
    u1[ifree] -= updt
    u[:]       = B1*u1
    normupdt   = √(updt⋅updt)
    ru[:]      = Ktu*u[:]
    res        = transpose(B1[:,ifree])*ru
    @printf("\t%i\t%.2e\t%.2e\n", 
            iter, normres, normupdt)
    iter      += 1
  end
  flush(stdout)

  normres, iter
end
function update_d!(elems, u, d, B0d, d0; 
                   dTol=1e-5, maxiter=10)
  print("\n doing update_d, ")

  updt = zero(d0)
  r    = zero(d0)

  _, res, Ktd  = getϕd(elems, u, d)
  Ktd  = transpose(B0d)*Ktd*B0d
  res  = transpose(B0d)*res[:]

  # initial free DoFs vector
  idd  = res .< 0
  previdd = copy(idd)
  iter = 1
  while true
    updt[:]   .= 0
    updt[idd]  = -Ktd[idd,idd]\res[idd]
    r[:]      .= Ktd*updt+res

    if all(updt[idd].≥ -dTol) && all(r[.!idd].≥ -dTol)
      println("done in $iter iterations")
      break
    elseif iter ≥ maxiter
      printstyled("did not converge after $iter iterations \n", color=:red)
      break
    end
    iter += 1

    # update the free DoFs
    previdd[:] .= idd[:]
    idd[.!idd]  = r[.!idd]  .< -dTol
    idd[idd]    = updt[idd] .> dTol
    if previdd == idd
      printstyled("exited after $iter iterations \n", color=:blue)
      break
    end
  end
  d0[idd]  += updt[idd] 
  clamp!(d0, 0, 1)
  d[:]      = B0d*d0

  extremar  = any(.!idd) ? extrema(r[.!idd])  : (NaN, NaN)
  extremau  = any(idd)   ? extrema(updt[idd]) : (NaN, NaN)

  @printf("\t extrema(updt[idd]) : % .3e, % .3e\n", extremau...)
  @printf("\t extrema(r[.!idd])  : % .3e, % .3e\n", extremar...)
  @printf("\t extrema(d)         : % .3e, % .3e\n", extrema(d0)...)

  return extremar
end

