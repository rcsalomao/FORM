from FORM import FORM
import numpy as np
import scipy.stats as st


def example1():
    def g1(Xi, Xd, d):
        Xd1, Xd2 = Xd
        return Xd1**3 + Xd2 - 9

    def g2(Xi, Xd, d):
        Xd1, Xd2 = Xd
        return Xd1**3 + Xd2**3 - 18

    f = FORM()
    Xd = [st.norm(10, 5), st.norm(9.9, 5)]
    system_definitions = [
        {"parallel": [0, 1]},
        {"serial": range(2)},
    ]
    res = f.HLRF([g1, g2], system_definitions=system_definitions, Xd=Xd)
    print(res.gXs_results)
    print(res.systems_results)
    print(res.gXs_results.pfs)
    print(res.gXs_results.betas)

    gx1_trace_data = f.limit_states_trace_data[0]
    print(gx1_trace_data.beta_k_trace)
    print(gx1_trace_data.k)


def example2():
    def g3(Xi, Xd, d):
        return sum([d[i] * Xi[i] for i in range(len(Xi))])

    f = FORM()
    Xi = [st.norm(10, 7), st.norm(10, 5), st.norm(15, 5)]
    d = [0.1, 0.1, 0.8]
    corr_matrix = np.eye(len(Xi))
    print(f.HLRF([g3], Xi=Xi, d=d, correlation_matrix=corr_matrix))
    print(f.limit_states_trace_data[0].k)
    print(f.limit_states_trace_data[0].alpha_k_trace[-1])
    print(f.limit_states_trace_data[0].y_k_trace[-1])
    print(f.limit_states_trace_data[0].x_k_trace[-1])


if __name__ == "__main__":
    example1()
    # example2()
