from FORM import FORM
import numpy as np
import scipy.stats as st


d = np.array([])
muZ = np.array([3.43908504, 3.28657327])
# muZ = np.array([5.0, 5.0])
muX = np.array([])
sd_covZ = np.array([0.3, 0.3])
sd_covX = np.array([])
option = 0  # sd:0 | cov:1


def gX(ng, d, Z, X):
    if ng == 0:
        Z1, Z2 = Z
        return Z1**2*Z2/20.0 - 1.0
    elif ng == 1:
        Z1, Z2 = Z
        return (Z1+Z2-5.0)**2/30.0 + (Z1-Z2-12.0)**2/120.0 - 1.0
    elif ng == 2:
        Z1, Z2 = Z
        return 80.0/(Z1**2 + 8.0*Z2 + 5.0) - 1.0


def g1(X, Z, d):
    Z1, Z2 = Z
    return Z1**2*Z2/20.0 - 1.0


# f = FORM(gX, d, muZ, muX, option, sd_covZ, sd_covX)
# f.HLRF(1)
# print(f.beta)

Zi = [st.norm(3.43908504, 0.3), st.norm(3.28657327, 0.3)]
f = FORM()
corrMatrix = np.eye(2)
# corrMatrix[0, 1] = 0.6
# corrMatrix[1, 0] = 0.6
print(f.HLRF(g1, Zi=Zi, correlationMatrix=corrMatrix))
print(f.k)
print(f.alphaKTrace[-1])
print(f.xKTrace[-1])
print(f.yKTrace[-1])
