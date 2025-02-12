import bempp.api

class ConvergenceAnalysis:

    @staticmethod
    def convergence_analysis_mixed(solver, n_BC, d_BC, n_ex, d_ex, steps: int):
        h_values = []
        L2 = []
        H1 = []
        its = []
        ndofs = []
        for i in range(2, steps):
            h = 2 ** -(i / 2)
            grid = bempp.api.shapes.cube(h)
            neumann_sol, dirichlet_sol, global_neumann_space, global_dirichlet_space, it = (
                solver(grid, n_BC, d_BC)
            )
            f_fun = bempp.api.GridFunction(global_dirichlet_space, fun=d_ex)
            g_fun = bempp.api.GridFunction(global_neumann_space, fun=n_ex)
            h_values.append(h)
            ndofs.append(global_dirichlet_space.global_dof_count)
            L2.append((dirichlet_sol - f_fun).l2_norm())
            H1.append((neumann_sol - g_fun).l2_norm())
            its.append(it)
            # H1.append(L2[-1] + (neumann_sol - g_fun).l2_norm())
        return h_values, ndofs, L2, H1, its

    # TODO: implement Neumann convergence method
    # TODO: test Dirichlet method
    @staticmethod
    def convergence_analysis_Dirichlet(solver, BC, u_ex, steps: int):
        h_values = []
        L2 = []
        ndofs = []
        for i in range(2, steps):
            h = 2 ** -(i / 2)
            grid = bempp.api.shapes.cube(h)
            dirichlet_sol, global_dirichlet_space = solver(grid, u_ex)
            f_fun = bempp.api.GridFunction(global_dirichlet_space, fun=u_ex)
            h_values.append(h)
            ndofs.append(global_dirichlet_space.global_dof_count)
            L2.append((dirichlet_sol - f_fun).l2_norm())
        return h_values, ndofs, L2
