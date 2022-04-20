import numpy as np
import scipy.stats as st
from scipy.optimize import root
import math


def error_RV_2_param(x, rv, mean, std):
    random_var = rv(x[0], scale=x[1])
    return [mean-random_var.mean(), std-random_var.std()]


def generate_RV_2_param(rv, mean, std, x0=(4.2, 2.4), method='lm', tol=1e-5):
    sol = root(error_RV_2_param, x0=x0, args=(rv, mean, std), method=method)
    random_var = rv(sol.x[0], scale=sol.x[1])
    assert((abs(random_var.mean()-mean) < tol) and (abs(random_var.std() - std) < tol))
    return random_var


def compute_D_neq(random_vars, z_k, x_k):
    n_rv = len(random_vars)
    numerador = st.norm.pdf(z_k)
    denominador = np.zeros(n_rv)
    for i in range(n_rv):
        denominador[i] = random_vars[i].pdf(x_k[i])
    return np.eye(len(x_k))*(numerador/denominador)


def compute_dgX(gX, xi, xd, d, h=1e-5):
    n_xi = len(xi)
    n_xd = len(xd)
    dgX = np.zeros(n_xi+n_xd)
    g = gX(xi, xd, d)
    for i in range(n_xi):
        xi_temp = np.copy(xi)
        xi_temp[i] += h
        dgX[i] = (gX(xi_temp, xd, d) - g)/h
    for i in range(n_xd):
        xd_temp = np.copy(xd)
        xd_temp[i] += h
        dgX[i+n_xi] = (gX(xi, xd_temp, d) - g)/h
    return dgX


def criterio_convergencia_epsilon(dgY, y_k, epsilon):
    return (1 + math.fabs((np.dot(dgY, y_k))/(np.linalg.norm(dgY)*np.linalg.norm(y_k)))) < epsilon


def criterio_convergencia_delta(gY, delta):
    return math.fabs(gY) < delta


class FORM(object):

    def __init__(self):
        self.k = None
        self.beta_k_trace = []
        self.alpha_k_trace = []
        self.y_k_trace = []
        self.x_k_trace = []

    def HLRF(
            self,
            gX,
            Xi=None,
            Xd=None,
            d=None,
            correlation_matrix=None,
            epsilon=2.0001,
            delta=1e-5,
            max_number_iterations=1000,
            h_numerical_gradient=1e-4):
        if Xi is None:
            Xi = []
        if Xd is None:
            Xd = []
        if d is None:
            d = []

        n_Xi = len(Xi)
        n_Xd = len(Xd)
        random_vars = Xi+Xd
        n_rv = n_Xi+n_Xd

        Jzy = np.linalg.cholesky(correlation_matrix)
        Jyz = np.linalg.inv(Jzy)

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
        alpha_k = dgy/np.linalg.norm(dgy)

        self.x_k_trace.append(x_k)
        self.y_k_trace.append(y_k)
        self.beta_k_trace.append(beta_k)
        self.alpha_k_trace.append(alpha_k)

        k = 0
        converg = False
        while (not converg) and (k < max_number_iterations):
            y_k_menos1 = np.copy(y_k)
            dgx_k_menos1 = np.copy(dgx)
            d_k = (
                np.dot(np.dot(np.dot(np.dot(dgy, inv_B_k), y_k) - gx, inv_B_k), dgy) /
                (np.dot(np.dot(dgy, inv_B_k), dgy)) -
                np.dot(inv_B_k, y_k)
            )
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
            alpha_k = dgy/np.linalg.norm(dgy)

            self.x_k_trace.append(x_k)
            self.y_k_trace.append(y_k)
            self.beta_k_trace.append(beta_k)
            self.alpha_k_trace.append(alpha_k)

            if criterio_convergencia_epsilon(dgy, y_k, epsilon) and criterio_convergencia_delta(gx, delta):
                converg = True

            xsi_k = (gx - np.dot(np.dot(dgy, inv_B_k), y_k))/(np.dot(np.dot(dgy, inv_B_k), dgy))
            lambda_k = xsi_k
            p_k = y_k - y_k_menos1
            q_k = lambda_k*dgx - lambda_k*dgx_k_menos1
            inv_B_k += (
                (1 + np.dot(np.dot(q_k, inv_B_k), q_k)/np.dot(p_k, q_k))*(np.dot(p_k, p_k)/np.dot(p_k, q_k)) -
                (np.dot(np.dot(p_k, q_k), inv_B_k) + np.dot(np.dot(inv_B_k, q_k), p_k))/(np.dot(p_k, q_k))
            )
            k += 1
        self.k = k
        return (st.norm.cdf(-beta_k), beta_k)
