import bempp.api
from utils.BC import BC
import numpy as np


def mixed(grid: bempp.api.Grid, neumann_BC: BC, dirichlet_BC: BC):
    # Set the sides of the mesh to Neumann/Dirichlet
    neumann_segments = neumann_BC.allocation
    dirichlet_segments = dirichlet_BC.allocation

    # Define the spaces in which to find Neumann/Dirichlet solutions globally
    global_neumann_space = bempp.api.function_space(
        grid, neumann_BC.space[0], neumann_BC.space[1]
    )
    global_dirichlet_space = bempp.api.function_space(
        grid, dirichlet_BC.space[0], dirichlet_BC.space[1]
    )

    # Space to search for Neumann unkown where Dirichlet data is defined
    neumann_space_dirichlet_segment = bempp.api.function_space(
        grid, neumann_BC.space[0], neumann_BC.space[1], segments=dirichlet_segments
    )

    # Space on which to hold the given Neumann data
    neumann_space_neumann_segment = bempp.api.function_space(
        grid, neumann_BC.space[0], neumann_BC.space[1], segments=neumann_segments
    )

    # Space on which to hold the given Dirichlet data
    dirichlet_space_dirichlet_segment = bempp.api.function_space(
        grid,
        dirichlet_BC.space[0],
        dirichlet_BC.space[1],
        segments=dirichlet_segments,
        include_boundary_dofs=True,  # As we have mixed BCs, at least one of the spaces needs to contain the boundary DOFs, so here we set that to the Dirichlet data space
        truncate_at_segment_edge=False,  # By setting this to False we extend the basis functions on these boundary DOFs (ctsly) into the complement
    )

    # Space on which to find the unkown Dirichlet solution defines on the segments where Neumann data is given
    dirichlet_space_neumann_segment = bempp.api.function_space(
        grid, dirichlet_BC.space[0], dirichlet_BC.space[1], segments=neumann_segments
    )

    # the dual of P1 is P1? but then in op. def. the dual is P0 ...
    dual_dirichlet_space = bempp.api.function_space(
        grid,
        dirichlet_BC.space[0],
        dirichlet_BC.space[1],
        segments=dirichlet_segments,
        include_boundary_dofs=True,
    )

    # Now to define the operators between these spaces:

    # Dirichlet seg.: Vu_N - Ku_D = (0.5 + K)g_D - Vg_N
    # Neumann seg.:   Wu_D + K'u_N = (0.5 - K')g_N - Wg_D

    # V
    slp_DD = bempp.api.operators.boundary.laplace.single_layer(
        neumann_space_dirichlet_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    slp_DN = bempp.api.operators.boundary.laplace.single_layer(
        neumann_space_neumann_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    # K
    dlp_DN = bempp.api.operators.boundary.laplace.double_layer(
        dirichlet_space_neumann_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    dlp_DD = bempp.api.operators.boundary.laplace.double_layer(
        dirichlet_space_dirichlet_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    # K'
    adlp_ND = bempp.api.operators.boundary.laplace.adjoint_double_layer(
        neumann_space_dirichlet_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    adlp_NN = bempp.api.operators.boundary.laplace.adjoint_double_layer(
        neumann_space_neumann_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    # W
    hyp_NN = bempp.api.operators.boundary.laplace.hypersingular(
        dirichlet_space_neumann_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    hyp_ND = bempp.api.operators.boundary.laplace.hypersingular(
        dirichlet_space_dirichlet_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    # I
    id_DD = bempp.api.operators.boundary.sparse.identity(
        dirichlet_space_dirichlet_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    id_NN = bempp.api.operators.boundary.sparse.identity(
        neumann_space_neumann_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    # Define the linear system of operators to solve for unknowns
    blocked = bempp.api.BlockedOperator(2, 2)

    blocked[0, 0] = slp_DD
    blocked[0, 1] = -dlp_DN
    blocked[1, 0] = adlp_ND
    blocked[1, 1] = hyp_NN

    # Define the given data i.e. boundary condition functions g_D, g_N
    dirichlet_grid_fun = bempp.api.GridFunction(
        dirichlet_space_dirichlet_segment,
        fun=dirichlet_BC.data,
        dual_space=dual_dirichlet_space,
    )

    neumann_grid_fun = bempp.api.GridFunction(
        neumann_space_neumann_segment,
        fun=neumann_BC.data,
        dual_space=dirichlet_space_neumann_segment,
    )

    # Compile RHS data
    rhs_fun1 = (0.5 * id_DD + dlp_DD) * dirichlet_grid_fun - slp_DN * neumann_grid_fun
    rhs_fun2 = -hyp_ND * dirichlet_grid_fun + (0.5 * id_NN - adlp_NN) * neumann_grid_fun

    (neumann_solution, dirichlet_solution), _, it = bempp.api.linalg.gmres(
        blocked, [rhs_fun1, rhs_fun2], return_iteration_count=True
    )

    # And then finally, in order to puzzle together the final solutions, we want
    # to put the solutions in function spaces defined on different portions of the
    # boundary into the same space.
    neumann_imbedding_dirichlet_segment = bempp.api.operators.boundary.sparse.identity(
        neumann_space_dirichlet_segment, global_neumann_space, global_neumann_space
    )

    neumann_imbedding_neumann_segment = bempp.api.operators.boundary.sparse.identity(
        neumann_space_neumann_segment, global_neumann_space, global_neumann_space
    )

    dirichlet_imbedding_dirichlet_segment = (
        bempp.api.operators.boundary.sparse.identity(
            dirichlet_space_dirichlet_segment,
            global_dirichlet_space,
            global_dirichlet_space,
        )
    )

    dirichlet_imbedding_neumann_segment = bempp.api.operators.boundary.sparse.identity(
        dirichlet_space_neumann_segment, global_dirichlet_space, global_dirichlet_space
    )

    # Forming global solutions
    dirichlet = (
        dirichlet_imbedding_dirichlet_segment * dirichlet_grid_fun
        + dirichlet_imbedding_neumann_segment * dirichlet_solution
    )

    neumann = (
        neumann_imbedding_neumann_segment * neumann_grid_fun
        + neumann_imbedding_dirichlet_segment * neumann_solution
    )

    return neumann, dirichlet, global_neumann_space, global_dirichlet_space, it


def Dirichlet(grid, dirichlet_BC):

    # Define a 'Neumann'/'Dirichlet' space
    dp0_space = bempp.api.function_space(grid, "DP", 0)
    p1_space = bempp.api.function_space(grid, "P", 1)

    # Dirichlet seg.: Vu_N = (0.5 + K)g_D

    # I
    identity = bempp.api.operators.boundary.sparse.identity(
        p1_space, p1_space, dp0_space
    )

    # K
    dlp = bempp.api.operators.boundary.laplace.double_layer(
        p1_space, p1_space, dp0_space
    )

    # V
    slp = bempp.api.operators.boundary.laplace.single_layer(
        dp0_space, p1_space, dp0_space
    )

    # Define Dirichlet data function
    dirichlet_fun = bempp.api.GridFunction(p1_space, fun=dirichlet_BC.data)

    # Define RHS
    rhs = (0.5 * identity + dlp) * dirichlet_fun

    neumann_fun, info = bempp.api.linalg.cg(slp, rhs, tol=1e-3)

    return neumann_fun


def Neumann(grid, Neumann_BC):

    # Dirichlet seg.: - Ku_D =  - Vg_N
    # Neumann seg.:   Wu_D +  = (0.5 - K')g_N

    # space = bempp.api.function_space(grid, "DP", 0)
    # u, info = gmres(double_layer - 0.5 * identity, single_layer * lambda_fun, tol=1E-5)

    # -------------------------------------------
    #     # Define a 'Neumann'/'Dirichlet' space
    #     dp0_space = bempp.api.function_space(grid, 'DP', 0)
    #     p1_space = bempp.api.function_space(grid, 'P', 1)

    #     # Neumann seg.:   Wu_D = (0.5 - K')g_N

    space = bempp.api.function_space(grid, "DP", 0)

    # I
    identity = bempp.api.operators.boundary.sparse.identity(space, space, space)

    # K
    dlp = bempp.api.operators.boundary.laplace.double_layer(space, space, space)

    # V
    slp = bempp.api.operators.boundary.laplace.single_layer(space, space, space)

    nfun = bempp.api.GridFunction(space, fun=Neumann_BC.data)

    dfun, info = bempp.api.linalg.gmres(dlp - 0.5 * identity, slp * nfun, tol=1e-5)

    return dfun, nfun
