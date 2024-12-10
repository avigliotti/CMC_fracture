
include("helper_funcs.jl") 
include("solver.jl")  

l0        = 0.5000
Em, Ef    = 430.00, 588.00
νm, νf    = 0.17, 0.27
ϵdm, ϵdf  = 1.512e-03, 6.497e-03
Gcf       = 2Ef*l0*ϵdf^2
Gcm       = 2Em*l0*ϵdm^2

fiber_mat  = PhaseField(l0, Gcf, Hooke(Ef, νf, small=true))
matrix_mat = PhaseField(l0, Gcm, Hooke(Em, νm, small=true))

run_phase_field_qspso(sModelName="cm_hex1x1x1nL20_r500L1100lc0125lcf0500", 
                     ϵM0=[NaN, NaN, 1.000, NaN, NaN, NaN]*9.745e-03, 
                     θ=0.000000, ψ=0.392699, ζ=0.000000,
                     fiber_mat=fiber_mat, matrix_mat=matrix_mat,
                     nSteps=301, nwritevtu=NaN, 
                     maxiter=5, λV=1.00e+00, λT=1.00e-02,
                     maxiteru=3,maxδdTol=5.000e-02, 
                     dTolu=1.0e-03, dTold=Inf, Nupdtmax=10, Nupdtmin=10,
                     sPost="AT1nhl0500psi18t")

