from FORM import FORM
import numpy as np
import scipy.stats as st


def g1(Xi, Xd, d):
    Xd1, Xd2 = Xd
    return Xd1**2*Xd2/20.0 - 1.0


Xd = [st.norm(3.43908504, 0.3), st.norm(3.28657327, 0.3)]
f = FORM()
corrMatrix = np.eye(2)
# corrMatrix[0, 1] = 0.6
# corrMatrix[1, 0] = 0.6
print(f.HLRF(g1, Xd=Xd, correlationMatrix=corrMatrix))
print(f.k)
print(f.alphaKTrace[-1])
print(f.xKTrace[-1])
print(f.yKTrace[-1])
