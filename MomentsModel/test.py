import numpy as np
import scipy.optimize as spo
import lsst.meas.modelfit.regularizedMoments as RM


def buildTestImage(moments, imsize, noise):
    indY, indX = np.indices((imsize, imsize))
    image = RM.makeGaussian(indX, indY, *moments)
    noiseImage = np.random.normal(0, noise, image.size).reshape(image.shape)
    return image+noiseImage


def fitFunctionFactory(measMoments, weightMoments, uncertanty, imSize):
    momentsModel = RM.MomentsModel(weightMoments)
    uncert = np.linalg.pinv(RM.buildUncertanty((imSize, imSize), W, uncertanty))
    L = np.linalg.cholesky(uncert)
    Linv = np.linalg.pinv(L)

    def fitFunction(params):
        momentsModel.at(params)
        result = momentsModel.computeValues()
        resultVec = result - measMoments
        return np.dot(Linv, resultVec)

    def jacobian(params):
        return np.dot(Linv, momentsModel.computeJacobian())

    def fitFunctionChi(params):
        momentsModel.at(params)
        result = momentsModel.computeValues()
        resultVec = measMoments - result
        chi = np.dot(resultVec, uncert) * resultVec
        return np.sum(chi)

    return fitFunction, jacobian, fitFunctionChi,  momentsModel


W = np.array((1, 50, 50, 28.4, 0.15, 24.4))
Q = np.array((3000, 50, 50, 30.4, 0.15, 27.4))
imSize = 101
uncert = 2
tImage = buildTestImage(Q, imSize, uncert)
measMoments, weightImage = RM.measureMoments(tImage, W)

fitFunc, jacFunc, fitFunctionChi, momentsModel = fitFunctionFactory(measMoments, W, uncert, imSize)

guess = np.array((2427, 50, 50, 27.4, 0.15, 26.4))
output = spo.leastsq(fitFunc, guess, Dfun=jacFunc)


yInd, xInd = np.indices((imSize, imSize))

d = (yInd - 50)**2 + (xInd - 50)**2
mask = d < 31

guessMoments = []
guessMoments.append(np.sum(tImage))
guessMoments.append(np.sum(tImage*xInd)/guessMoments[0])
guessMoments.append(np.sum(tImage*yInd)/guessMoments[0])
guessMoments.append(np.sum(tImage*(xInd-guessMoments[1])**2)/guessMoments[0])
guessMoments.append(np.sum(tImage*(xInd-guessMoments[1])*(yInd-guessMoments[2]))/guessMoments[0])
guessMoments.append(np.sum(tImage*(yInd-guessMoments[2])**2)/guessMoments[0])
guessMoments[0] = 1

outputVector = np.zeros((40, 6))
adapW = np.array(guessMoments)
adapW[0] = 1
for i in range(outputVector.shape[0]):
    measMoments, weightImage = RM.measureMoments(tImage, adapW)
    fitFunc, jacFunc, _, _ = fitFunctionFactory(measMoments, adapW, uncert, imSize)
    tOut = spo.leastsq(fitFunc, guess, Dfun=jacFunc)[0]
    outputVector[i] = tOut
    vecDiff = tOut - adapW
    print(adapW, vecDiff)
    adapW = adapW + vecDiff
    adapW[0] = 1
