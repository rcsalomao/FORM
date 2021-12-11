import numpy as np
import scipy.stats as st
from scipy.optimize import root
import math


def errorRV2Param(x, rv, mean, std):
    randomVar = rv(x[0], scale=x[1])
    return [mean-randomVar.mean(), std-randomVar.std()]


def genRV2Param(rv, mean, std, x0=(4.2, 2.4), method='lm', tol=1e-5):
    sol = root(errorRV2Param, x0=x0, args=(rv, mean, std), method=method)
    randomVar = rv(sol.x[0], scale=sol.x[1])
    assert((abs(randomVar.mean()-mean) < tol) and (abs(randomVar.std() - std) < tol))
    return randomVar


def calcD(randomVars, xK, zK):
    nRV = len(randomVars)
    numerador = st.norm.pdf(zK)
    denominador = np.zeros(nRV)
    for i in range(nRV):
        denominador[i] = randomVars[i].pdf(xK[i])
    D = np.eye(len(xK))*(numerador/denominador)
    return D


def gradienteGX(gx, xi, xd, d, h=1e-3):
    nXi = len(xi)
    nXd = len(xd)
    dGX = np.zeros(nXi+nXd)
    g = gx(xi, xd, d)
    for i in range(nXi):
        xiTemp = np.copy(xi)
        xiTemp[i] += h
        dGX[i] = (gx(xiTemp, xd, d) - g)/h
    for i in range(nXd):
        xdTemp = np.copy(xd)
        xdTemp[i] += h
        dGX[i+nXi] = (gx(xi, xdTemp, d) - g)/h
    return dGX


def yKMais1(alpha, beta, GY, dGY):
    return -alpha*(beta + GY/np.linalg.norm(dGY))


def criterioConvergEpsilon(dGY, yK, epsilon):
    return (1 + math.fabs((np.dot(dGY, yK))/(np.linalg.norm(dGY)*np.linalg.norm(yK)))) < epsilon


def criterioConvergDelta(GY, delta):
    return math.fabs(GY) < delta


class FORM(object):

    def __init__(self):
        self.k = None
        self.betaKTrace = []
        self.alphaKTrace = []
        self.yKTrace = []
        self.xKTrace = []

    def HLRF(self, gx, Xi=None, Xd=None, d=None, correlationMatrix=None, epsilon=2.001, delta=1e-4, maxNumIter=1000, numGradH=1e-3):
        if Xi is None:
            Xi = []
        if Xd is None:
            Xd = []
        if d is None:
            d = []

        nXi = len(Xi)
        nXd = len(Xd)
        randomVars = Xi+Xd
        nRV = nXi+nXd

        Jzy = np.linalg.cholesky(correlationMatrix)
        Jyz = np.linalg.inv(Jzy)

        xK = np.zeros(nRV)
        zK = np.zeros(nRV)
        for i in range(nRV):
            xK[i] = randomVars[i].mean()
            zK[i] = st.norm.ppf(randomVars[i].cdf(randomVars[i].mean()))
        yK = np.dot(Jyz, zK)

        GX = gx(xK[0:nXi], xK[nXi:], d)
        dGX = gradienteGX(gx, xK[0:nXi], xK[nXi:], d, numGradH)

        Jxz = calcD(randomVars, xK, zK)
        # Jzx = np.linalg.inv(D)
        Jxy = np.dot(Jxz, Jzy)
        # Jyx = np.dot(Jyz, Jzx)

        dGY = np.dot(Jxy, dGX)
        betaK = np.sqrt(np.dot(yK, yK))
        alphaK = dGY/np.linalg.norm(dGY)
        converg = False
        yK = yKMais1(alphaK, betaK, GX, dGY)
        if criterioConvergEpsilon(dGY, yK, epsilon) and criterioConvergDelta(GX, delta):
            converg = True

        self.betaKTrace.append(betaK)
        self.alphaKTrace.append(alphaK)
        self.yKTrace.append(yK)
        self.xKTrace.append(xK)
        k = 1
        while (not converg) and (k < maxNumIter):
            zK = np.dot(Jzy, yK)
            for i in range(nRV):
                xK[i] = randomVars[i].ppf(st.norm.cdf(zK[i]))
            GX = gx(xK[0:nXi], xK[nXi:], d)
            dGX = gradienteGX(gx, xK[0:nXi], xK[nXi:], d, numGradH)
            Jxz = calcD(randomVars, xK, zK)
            Jxy = np.dot(Jxz, Jzy)
            dGY = np.dot(Jxy, dGX)
            betaK = np.sqrt(np.dot(yK, yK))
            alphaK = dGY/np.linalg.norm(dGY)
            yK = yKMais1(alphaK, betaK, GX, dGY)
            if criterioConvergEpsilon(dGY, yK, epsilon) and criterioConvergDelta(GX, delta):
                converg = True
            self.betaKTrace.append(betaK)
            self.alphaKTrace.append(alphaK)
            self.yKTrace.append(yK)
            self.xKTrace.append(xK)
            k += 1
        self.k = k

        return (st.norm.cdf(-betaK), betaK)
