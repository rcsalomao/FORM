import math
from collections import deque, namedtuple
from copy import deepcopy

import numpy as np
import scipy.stats as st
from scipy.differentiate import jacobian
from scipy.optimize import minimize_scalar


def get_rho_ij(rho_ij: dict, i, j):
    if i < j:
        return rho_ij[(i, j)]
    else:
        return rho_ij[(j, i)]


def set_rho_ij(rho_ij: dict, i, j, value):
    if i < j:
        rho_ij[(i, j)] = value
    else:
        rho_ij[(j, i)] = value


def calc_rho_ij(alphas_gX):
    n_gX = len(alphas_gX)
    rho_ij = {}
    for i in range(n_gX):
        for j in range(i, n_gX):
            rho = alphas_gX[i].dot(alphas_gX[j])
            set_rho_ij(rho_ij, i, j, rho)
    return rho_ij


def calc_A(beta_k):
    return st.norm.pdf(-beta_k) / st.norm.cdf(-beta_k)


def calc_B(A, beta_k):
    return A * (-beta_k + A)


def calc_beta_i_k(beta_i, rho_i_k, A, B):
    return (beta_i - rho_i_k * A) / np.sqrt(1 - rho_i_k**2 * B)


def calc_rho_12_k(B, rho_12, rho_1_k, rho_2_k):
    return (rho_12 - rho_1_k * rho_2_k * B) / (
        np.sqrt(1 - rho_1_k**2 * B) * np.sqrt(1 - rho_2_k**2 * B)
    )


def calc_beta_12_k(beta_12, rho_12_k, A, B):
    return (beta_12 - rho_12_k * A) / np.sqrt(1 - rho_12_k**2 * B)


def calc_bivariate_cdf(system_type, beta_1, beta_2, rho_12):
    if system_type == "parallel":
        return st.multivariate_normal.cdf(
            [-beta_1, -beta_2],
            cov=np.array([[1.0, rho_12], [rho_12, 1.0]]),
            allow_singular=True,
        )
    if system_type == "serial":
        return 1.0 - st.multivariate_normal.cdf(
            [beta_1, beta_2],
            cov=np.array([[1.0, rho_12], [rho_12, 1.0]]),
            allow_singular=True,
        )
    raise ValueError(
        "Invalid system type: {0}. Only 'parallel' or 'serial' types allowed".format(
            system_type
        )
    )


def calc_beta_12(system_type, beta_1, beta_2, rho_12):
    return -st.norm.ppf(calc_bivariate_cdf(system_type, beta_1, beta_2, rho_12))


def get_limit_states_count(system_definition, beta_i):
    for system_components in system_definition.values():
        for component in system_components:
            if isinstance(component, dict):
                get_limit_states_count(component, beta_i)
            else:
                beta_i[component]["count"] += 1


def compound_values(system_type, system_components, beta_i, rho_ij):
    m = system_components.pop()
    n = system_components.pop()
    beta_number = max(beta_i.keys()) + 1
    system_components.append(beta_number)
    beta_1 = beta_i[m]["value"]
    beta_2 = beta_i[n]["value"]
    rho_12 = get_rho_ij(rho_ij, m, n)
    beta_12 = calc_beta_12(system_type, beta_1, beta_2, rho_12)
    beta_i[m]["count"] -= 1
    beta_i[n]["count"] -= 1
    beta_i[beta_number] = {"value": beta_12, "count": 1}
    for k in beta_i.keys():
        if k == beta_number or beta_i[k]["count"] < 1:
            continue
        beta_k = beta_i[k]["value"]
        rho_1_k = get_rho_ij(rho_ij, m, k)
        rho_2_k = get_rho_ij(rho_ij, n, k)
        A = calc_A(beta_k)
        B = calc_B(A, beta_k)
        beta_1_k = calc_beta_i_k(beta_1, rho_1_k, A, B)
        beta_2_k = calc_beta_i_k(beta_2, rho_2_k, A, B)
        rho_12_k = calc_rho_12_k(B, rho_12, rho_1_k, rho_2_k)
        bivariate_cdf_k = calc_bivariate_cdf(system_type, beta_1_k, beta_2_k, rho_12_k)
        min_result = minimize_scalar(
            lambda x: np.abs(
                bivariate_cdf_k - st.norm.cdf(-calc_beta_12_k(beta_12, x, A, B))
            ),
            bounds=(-1, 1),
            method="bounded",
        )
        equivalent_rho_12_k = min_result.x
        set_rho_ij(rho_ij, beta_number, k, equivalent_rho_12_k)


def calc_system_beta_helper(system_definition, beta_i, rho_ij):
    for system_type, system_components in system_definition.items():
        break
    queue = deque()
    for i in system_components:
        if isinstance(i, dict):
            queue.appendleft(calc_system_beta_helper(i, beta_i, rho_ij))
        else:
            queue.appendleft(i)
    while len(queue) > 1:
        compound_values(system_type, queue, beta_i, rho_ij)
    return queue.pop()


def calc_system_beta(system_definition, betas_gX, alphas_gX):
    beta_i = {i: {"value": v, "count": 0} for i, v in enumerate(betas_gX)}
    get_limit_states_count(system_definition, beta_i)
    rho_ij = calc_rho_ij(alphas_gX)
    s = deepcopy(system_definition)
    return beta_i[calc_system_beta_helper(s, beta_i, rho_ij)]["value"]


def compute_D_neq(random_vars, z_k, x_k):
    n_rv = len(random_vars)
    numerador = st.norm.pdf(z_k)
    denominador = np.zeros(n_rv)
    for i in range(n_rv):
        denominador[i] = random_vars[i].pdf(x_k[i])
    return np.eye(len(x_k)) * (numerador / denominador)


def compute_dgX(gX, x_k, n_Xi, d, **kwargs):
    def f(x):
        return gX(x[0:n_Xi], x[n_Xi:], d)

    res = jacobian(f, x_k, **kwargs)
    if res:
        return res.df
    else:
        raise RuntimeError("Could not compute g(X,d) Jacobian.")


def criterio_convergencia_epsilon(dgY, y_k, epsilon):
    return (
        1 + math.fabs((np.dot(dgY, y_k)) / (np.linalg.norm(dgY) * np.linalg.norm(y_k)))
    ) < epsilon


def criterio_convergencia_delta(gY, delta):
    return math.fabs(gY) < delta


class FORM(object):
    def __init__(self):
        self.limit_states_trace_data = []

    def HLRF(
        self,
        limit_state_functions,
        system_definitions=None,
        Xi=None,
        Xd=None,
        d=None,
        correlation_matrix=None,
        epsilon=2.0001,
        delta=0.0001,
        max_number_iterations=1000,
        jacobian_tolerances=None,
        jacobian_maxiter=10,
        jacobian_order=8,
        jacobian_initial_step=0.5,
        jacobian_step_factor=2.0,
        jacobian_step_direction=0,
    ):
        if system_definitions is None:
            system_definitions = []
        if Xi is None:
            Xi = []
        if Xd is None:
            Xd = []
        if d is None:
            d = []

        n_Xi = len(Xi)
        n_Xd = len(Xd)
        random_vars = Xi + Xd
        n_rv = n_Xi + n_Xd

        if correlation_matrix is None:
            correlation_matrix = np.eye(n_rv)
        Jzy = np.linalg.cholesky(correlation_matrix)
        Jyz = np.linalg.inv(Jzy)

        self.limit_states_trace_data.clear()
        gX_trace_data = namedtuple(
            "gX_trace_data", "beta_k_trace, alpha_k_trace, x_k_trace, y_k_trace, k"
        )
        for gX in limit_state_functions:
            x_k = np.zeros(n_rv)
            z_k = np.zeros(n_rv)
            for i in range(n_rv):
                x_k[i] = random_vars[i].mean()
                z_k[i] = st.norm.ppf(random_vars[i].cdf(random_vars[i].mean()))
            y_k = np.dot(Jyz, z_k)

            inv_B_k = np.eye(len(y_k))

            gx = gX(x_k[0:n_Xi], x_k[n_Xi:], d)
            dgx = compute_dgX(
                gX,
                x_k,
                n_Xi,
                d,
                tolerances=jacobian_tolerances,
                maxiter=jacobian_maxiter,
                order=jacobian_order,
                initial_step=jacobian_initial_step,
                step_factor=jacobian_step_factor,
                step_direction=jacobian_step_direction,
            )
            Jxz = compute_D_neq(random_vars, z_k, x_k)
            Jxy = np.dot(Jxz, Jzy)
            dgy = np.dot(Jxy.transpose(), dgx)
            beta_k = np.sqrt(np.dot(y_k, y_k))
            alpha_k = dgy / np.linalg.norm(dgy)

            beta_k_trace = [beta_k]
            alpha_k_trace = [alpha_k]
            x_k_trace = [x_k]
            y_k_trace = [y_k]

            k = 0
            converg = False
            while (not converg) and (k < max_number_iterations):
                y_k_menos1 = np.copy(y_k)
                dgx_k_menos1 = np.copy(dgx)
                d_k = np.dot(
                    np.dot(np.dot(np.dot(dgy, inv_B_k), y_k) - gx, inv_B_k), dgy
                ) / (np.dot(np.dot(dgy, inv_B_k), dgy)) - np.dot(inv_B_k, y_k)
                y_k += d_k
                z_k = np.dot(Jzy, y_k)
                for i in range(n_rv):
                    x_k[i] = random_vars[i].ppf(st.norm.cdf(z_k[i]))
                gx = gX(x_k[0:n_Xi], x_k[n_Xi:], d)
                dgx = compute_dgX(
                    gX,
                    x_k,
                    n_Xi,
                    d,
                    tolerances=jacobian_tolerances,
                    maxiter=jacobian_maxiter,
                    order=jacobian_order,
                    initial_step=jacobian_initial_step,
                    step_factor=jacobian_step_factor,
                    step_direction=jacobian_step_direction,
                )
                Jxz = compute_D_neq(random_vars, z_k, x_k)
                Jxy = np.dot(Jxz, Jzy)
                dgy = np.dot(Jxy.transpose(), dgx)
                beta_k = np.sqrt(np.dot(y_k, y_k))
                alpha_k = dgy / np.linalg.norm(dgy)

                beta_k_trace.append(beta_k)
                alpha_k_trace.append(alpha_k)
                x_k_trace.append(x_k)
                y_k_trace.append(y_k)

                if criterio_convergencia_epsilon(
                    dgy, y_k, epsilon
                ) and criterio_convergencia_delta(gx, delta):
                    converg = True

                xsi_k = (gx - np.dot(np.dot(dgy, inv_B_k), y_k)) / (
                    np.dot(np.dot(dgy, inv_B_k), dgy)
                )
                lambda_k = xsi_k
                p_k = y_k - y_k_menos1
                q_k = lambda_k * dgx - lambda_k * dgx_k_menos1
                inv_B_k += (
                    1 + np.dot(np.dot(q_k, inv_B_k), q_k) / np.dot(p_k, q_k)
                ) * (np.dot(p_k, p_k) / np.dot(p_k, q_k)) - (
                    np.dot(np.dot(p_k, q_k), inv_B_k)
                    + np.dot(np.dot(inv_B_k, q_k), p_k)
                ) / (np.dot(p_k, q_k))
                k += 1
            self.limit_states_trace_data.append(
                gX_trace_data(beta_k_trace, alpha_k_trace, x_k_trace, y_k_trace, k)
            )
        betas_gX = np.array(
            [gXs_data.beta_k_trace[-1] for gXs_data in self.limit_states_trace_data],
            dtype=np.float64,
        )
        alphas_gX = np.array(
            [gXs_data.alpha_k_trace[-1] for gXs_data in self.limit_states_trace_data],
            dtype=np.float64,
        )
        pfs_gX = st.norm.cdf(-betas_gX)
        betas_sys = np.array(
            [
                calc_system_beta(system_definition, betas_gX, alphas_gX)
                for system_definition in system_definitions
            ],
            dtype=np.float64,
        )
        pfs_sys = st.norm.cdf(-betas_sys)
        gXs_results = namedtuple("gXs_results", "pfs, betas")
        systems_results = namedtuple("systems_results", "pfs, betas")
        result = namedtuple("result", "gXs_results, systems_results")
        return result(
            gXs_results(pfs_gX, betas_gX), systems_results(pfs_sys, betas_sys)
        )
