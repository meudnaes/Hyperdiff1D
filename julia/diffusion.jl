"""
    function hyperdiffusion(solver::Solver, 
        rho::AbstractVector, 
        u::AbstractVector, 
        e::AbstractVector, 
        f::AbstractVector)

Calculate hyperdiffusion, f defined at i+1/2
"""
function hyperdiffusion(solver::Solver,
                        rho::AbstractVector, 
                        u::AbstractVector, 
                        e::AbstractVector, 
                        f::AbstractVector)

    # centered
    P = solver.eos.(e,mode=:pressure)

    # centered
    src1 = solver.parameters["nu_1"]*solver.c_s.(P,rho)[10:end-9]
    
    # shifted i+1/2
    src2 = solver.parameters["nu_2"]*abs.(u)
    # centered
    src2 = x_shift(src2,shift=-1)[7:end-6]

    # centered
    src3 = solver.parameters["nu_3"]*solver.dx*deriv_6th(solver,u,shift=-1)[7:end-6]

    # centered
    df = deriv_6th(solver,f,shift=-1)

    # centered
    Q = quenchx(df)

    # centered
    inner = @. solver.dx*(src1 + src2 + src3)*df[4:end-3]*Q

    # Derivative returns at cell face, i+1/2
    return deriv_6th(solver,inner,shift=0) #[12:-12]
    
end