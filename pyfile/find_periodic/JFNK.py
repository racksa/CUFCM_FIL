import numpy as np
import newton_solver
import os
import time
import util

def main():
    # Define 'global variables'
    NSEG = 20
    NFIL = 159
    NBLOB = 9000
    AR = 8

    # NFIL = 639       # Number of filaments
    # NBLOB = 40961
    # AR = 15

    # output_filename = f"data/expr_sims/20240208_periodic/psi_guess{NFIL}.dat"
    output_filename = f"data/JFNK/20240320_JFNK_d/psi_guess{NFIL}.dat"

    # Number of time steps (ndts) and fixT
    ndts = 300
    fixT = False
 
    # n = 3*(NSEG-1)*NFIL+1 #4#3 * (N - 1) * Nf + 1  # Dimension of system, including unknown params
    n = 2*NFIL+1
    mgmres = 5 # 10  # max GMRES iterations
    nits = 150  # max Newton iterations
    rel_err_ini = 1e-8  # 1e-8 Relative error |F|/|x|
    del_value_ini = -1  # These rarely need changing for any problem
    mndl_ini = 1e-20
    mxdl_ini = 1e20
    gtol = 5e-3  # 1e-4
    epsJ = 1e-5 # 1e-6  # epsilon used in Jacobian approximation

    # f_range = np.arange(0.010, 0.049, 0.001)[::-1]
    # f_range = np.arange(0.010, 0.061, 0.001)
    f_range = np.arange(0.025, 0.031, 0.005)[::-1]
    # f_range = np.arange(0.030, 0.062, 0.002)
    # f_range = np.arange(0.015, 0.045, 0.005)
    for k in f_range:

        print('-----------Spring constant = ' + str(k))
        new_x = find_new_x(fixT,NSEG,NFIL,output_filename)
        newton = newton_solver.NEWTON_SOLVER(new_x,epsJ,ndts,fixT,k,NFIL, NSEG, NBLOB, AR)

       # Scale parameters by |x| then call black box
        aux = np.copy(newton.new_x[1:]) # using the F(x) norm instead of F(x+T)
        # detach error from norm
        aux[:NFIL] = np.ones(NFIL)*np.pi
        d = np.linalg.norm(aux)
        

        # Do we want rel_err?
        tol = rel_err_ini * d
        print(f"Requested tol={tol}")
        del_value = del_value_ini * d
        mndl = mndl_ini * d
        mxdl = mxdl_ini * d

        info = 1

        # info = newton.NewtonHook(mgmres, n, gtol, tol, del_value, mndl, mxdl, nits, info)
        info = newton.NewtonHook(mgmres, n, gtol, tol, del_value, mndl, mxdl, nits, info)
        

        with open(output_filename, "ab") as f:
            f.write(b"\n")
            np.savetxt(f, np.concatenate(([k], newton.new_x)), newline = " ")

def find_new_x(fixT,NSEG,NFIL,input_filename):

    with open(input_filename, 'r') as file:
        num_lines = sum(1 for line in file)

        if num_lines == 1:
            full_input = np.loadtxt(input_filename)
        else:
            full_input = np.loadtxt(input_filename)[-1]
            
        full_input[2:2+NFIL] = util.box(full_input[2:2+NFIL], 2*np.pi)

        return full_input[1:]



# Run code
main()
