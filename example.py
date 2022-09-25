from FORM import FORM, serial_system, parallel_system
import numpy as np
import scipy.stats as st


def g1(Xi, Xd, d):
    Xd1, Xd2 = Xd
    return Xd1 ** 2 * Xd2 / 20.0 - 1.0


def g2(Xi, Xd, d):
    Xd1, Xd2, Xd3 = Xd
    return Xd1 * Xd2 - Xd3


def g3(Xi, Xd, d):
    Xd1, Xd2 = Xd
    return Xd1 ** 3 + Xd2 ** 3 - 18


def g4(Xi, Xd, d):
    return sum([d[i] * Xi[i] for i in range(len(Xi))])


# Xd = [st.norm(3.43908504, 0.3), st.norm(3.28657327, 0.3)]
# corr_matrix = np.eye(len(Xd))
# f = FORM()
# # corr_matrix[0, 1] = 0.6
# # corr_matrix[1, 0] = 0.6
# print(f.HLRF([g1], Xd=Xd, correlation_matrix=corr_matrix))
# print(f.limit_states_trace_data[0].k)
# print(f.limit_states_trace_data[0].alpha_k_trace[-1])
# print(f.limit_states_trace_data[0].y_k_trace[-1])
# print(f.limit_states_trace_data[0].x_k_trace[-1])


# Xd = [st.norm(40, 5), st.norm(50, 2.5), st.norm(1000, 200)]
# corr_matrix = np.eye(len(Xd))
# f = FORM()
# print(f.HLRF([g2], Xd=Xd, correlation_matrix=corr_matrix))
# print(f.limit_states_trace_data[0].k)
# print(f.limit_states_trace_data[0].alpha_k_trace[-1])
# print(f.limit_states_trace_data[0].y_k_trace[-1])
# print(f.limit_states_trace_data[0].x_k_trace[-1])


def sys1(pfs):
    return parallel_system(*pfs)


def sys2(pfs):
    return serial_system(*pfs)


Xd = [st.norm(10, 5), st.norm(9.9, 5)]
f = FORM()
res = f.HLRF([g3, g3], system_functions=[sys1, sys2], Xd=Xd, calc_serial_system=True)
print(res.gXs_results)
print(res.systems_results)
print(res.serial_system_result)
print(res.gXs_results.pfs[0], res.gXs_results.betas[0])
gx1_trace_data = f.limit_states_trace_data[0]
print(gx1_trace_data.beta_k_trace)
print(gx1_trace_data.k)

# Xi = [st.norm(10, 7), st.norm(10, 5), st.norm(15, 5)]
# d = [0.1, 0.1, 0.8]
# corr_matrix = np.eye(len(Xi))
# f = FORM()
# print(f.HLRF([g4], Xi=Xi, d=d, correlation_matrix=corr_matrix))
# print(f.limit_states_trace_data[0].k)
# print(f.limit_states_trace_data[0].alpha_k_trace[-1])
# print(f.limit_states_trace_data[0].y_k_trace[-1])
# print(f.limit_states_trace_data[0].x_k_trace[-1])
