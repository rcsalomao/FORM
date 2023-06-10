import math
from collections import namedtuple
from copy import deepcopy
from functools import partial

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar, root


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
        for j in range(1 + i, n_gX):
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


def compound_values(system_type, system_components, beta_i, rho_ij):
    m = system_components[0]
    n = system_components[1]
    beta_1 = beta_i[m]
    beta_2 = beta_i[n]
    rho_12 = get_rho_ij(rho_ij, m, n)
    beta_12 = calc_beta_12(system_type, beta_1, beta_2, rho_12)
    beta_number = max(beta_i.keys()) + 1
    system_components.pop(0)
    system_components[0] = beta_number
    beta_i.pop(m)
    beta_i.pop(n)
    beta_i[beta_number] = beta_12
    for k in beta_i.keys():
        if k == beta_number:
            continue
        beta_k = beta_i[k]
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


def calc_system_beta(system_definition, betas_gX, alphas_gX):
    beta_i = {i: v for i, v in enumerate(betas_gX)}
    rho_ij = calc_rho_ij(alphas_gX)
    current_system = deepcopy(system_definition)
    system_pointer = current_system
    parent_system_pointer_pointer = current_system
    system_pointer_position = 0
    while True:
        for current_system_type, current_system_components in current_system.items():
            break
        if len(current_system_components) == 1 and isinstance(
            current_system_components[0], int
        ):
            return beta_i[current_system_components[0]]
        if (
            len(current_system_components) == 2
            and isinstance(current_system_components[0], int)
            and isinstance(current_system_components[1], int)
        ):
            m = current_system_components[0]
            n = current_system_components[1]
            beta_1 = beta_i[m]
            beta_2 = beta_i[n]
            rho_12 = get_rho_ij(rho_ij, m, n)
            return calc_beta_12(current_system_type, beta_1, beta_2, rho_12)
        for system_pointer_type, system_pointer_components in system_pointer.items():
            break
        if isinstance(system_pointer_components[0], dict):
            parent_system_pointer_pointer = system_pointer
            system_pointer = system_pointer_components[0]
            system_pointer_position = 0
            continue
        if len(system_pointer_components) == 1:
            for (
                _,
                parent_system_pointer_pointer_components,
            ) in parent_system_pointer_pointer.items():
                break
            parent_system_pointer_pointer_components[
                system_pointer_position
            ] = system_pointer_components[0]
            system_pointer = current_system
            parent_system_pointer_pointer = current_system
            continue
        if isinstance(system_pointer_components[1], dict):
            parent_system_pointer_pointer = system_pointer
            system_pointer = system_pointer_components[1]
            system_pointer_position = 1
            continue
        compound_values(system_pointer_type, system_pointer_components, beta_i, rho_ij)


def error_RV(x, rv, mean, std, fixed_params, search_params):
    sp = {s: v for (s, v) in zip(search_params, x)}
    random_var = partial(rv, **fixed_params)(**sp)
    return [mean - random_var.mean(), std - random_var.std()]


def generate_RV(
    rv,
    mean,
    std,
    fixed_params: dict[str:float],
    search_params: list[str],
    x0=None,
    method="lm",
    tol=1e-4,
):
    if x0 is None:
        x0 = [4.2, 2.4][: len(search_params)]
    assert len(x0) == len(search_params)
    sol = root(
        error_RV,
        x0=x0,
        args=(rv, mean, std, fixed_params, search_params),
        method=method,
    )
    sp = {s: v for (s, v) in zip(search_params, sol.x)}
    random_var = partial(rv, **fixed_params)(**sp)
    assert (abs(random_var.mean() - mean) < tol) and (abs(random_var.std() - std) < tol)
    return random_var


def compute_D_neq(random_vars, z_k, x_k):
    n_rv = len(random_vars)
    numerador = st.norm.pdf(z_k)
    denominador = np.zeros(n_rv)
    for i in range(n_rv):
        denominador[i] = random_vars[i].pdf(x_k[i])
    return np.eye(len(x_k)) * (numerador / denominador)


def compute_dgX(gX, xi, xd, d, h=1e-5):
    n_xi = len(xi)
    n_xd = len(xd)
    dgX = np.zeros(n_xi + n_xd)
    g = gX(xi, xd, d)
    for i in range(n_xi):
        xi_temp = np.copy(xi)
        xi_temp[i] += h
        dgX[i] = (gX(xi_temp, xd, d) - g) / h
    for i in range(n_xd):
        xd_temp = np.copy(xd)
        xd_temp[i] += h
        dgX[i + n_xi] = (gX(xi, xd_temp, d) - g) / h
    return dgX


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
        delta=1e-4,
        max_number_iterations=1000,
        h_numerical_gradient=1e-3,
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
            dgx = compute_dgX(gX, x_k[0:n_Xi], x_k[n_Xi:], d, h_numerical_gradient)
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
                dgx = compute_dgX(gX, x_k[0:n_Xi], x_k[n_Xi:], d, h_numerical_gradient)
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
                ) / (
                    np.dot(p_k, q_k)
                )
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
