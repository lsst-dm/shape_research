import numpy as np
from demczs import demczs
import pickle
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable

import lsst.meas.extensions.shapeHSM as hsm
ITERATIONS = 1e4
THINNING = 1e1
PROCESSES = 4

#np.seterr(all='raise')

def printShapes(catalog, number):
    baseHsmKey = "ext_shapeHSM_HsmSourceMoments"
    baseSimpleKey = "ext_simpleShape_SimpleShape"
    baseSdssKey = "base_SdssShape"
    for n in [baseHsmKey, baseSimpleKey, baseSdssKey]:
        print(catalog[number][afwTable.QuadrupoleKey(catalog.schema[n])], n)


def plotEllipse(stamp, record, fig=plt.figure(), position=111,
                show=True, extra=None):
    baseHsmKey = "ext_shapeHSM_HsmSourceMoments"
    baseSimpleKey = "ext_simpleShape_SimpleShape"
    baseSdssKey = "base_SdssShape"
    allKeys = [baseHsmKey, baseSimpleKey, baseSdssKey]
    colors = ['w', 'm', 'k']
    ax = fig.add_subplot(position)
    imshowData = ax.imshow(stamp.array)
    for color, shapeKey in zip(colors, allKeys):
        eKey = afwTable.QuadrupoleKey(record.schema[shapeKey])
        print(record.get(eKey))
        print(afwGeom.ellipses.Axes(record.get(eKey)))
        ellipse = afwGeom.ellipses.Axes(record.get(eKey))
        centerE = record.getCentroid() - stamp.getXY0()
        centerE = centerE.asPoint()
        center = (centerE.getX(), centerE.getY())
        ax.add_patch(mpatch.Ellipse(center, ellipse.getA(), ellipse.getB(),
                                    np.rad2deg(ellipse.getTheta()), fill=False,
                                    color=color, label=shapeKey))
    fig.colorbar(imshowData)
    if extra is not None:
        quad = afwGeom.ellipses.Quadrupole(*extra)
        axes = afwGeom.ellipses.Axes(quad)
        ax.add_patch(mpatch.Ellipse(center, axes.getA(), axes.getB(),
                                    np.rad2deg(axes.getTheta()), fill=False,
                                    color='r', label='Fit'))

    ax.legend()
    if show:
        fig.show()


def gaussian(x, y, sigmax, sigmay, sigmaxy, centerx, centery, amp):
    rho = sigmaxy/(sigmax**0.5*sigmay**0.5)
    norm = 1/(2*np.pi*sigmax*sigmay*(1-rho**2)**0.5)
    psf = norm*np.exp(-1/(2*(1-rho**2))*((x-centerx)**2/sigmax +
                                         (y-centery)**2/sigmay -
                                         2*rho*(x-centerx)*(y-centery)/(sigmax**0.5*sigmay**0.5)))
    return psf*amp, psf


def fitFunc(parameters, ind, extra):
    xIndicies = ind[1]
    yIndicies = ind[0]
    centerX = extra[0]
    centerY = extra[1]
    sigX = parameters[0]
    sigY = parameters[1]
    sigXY = parameters[2]
    amp = parameters[3]
    return gaussian(xIndicies, yIndicies, sigX, sigY, sigXY, centerX, centerY,
                    amp)[0]


def chiFunc(model, data, errors, extra):
    if np.sum(np.isnan(model)) > 0:
        return 1e999
    return np.sum((model-data)**2/errors)


def scipyMinFunc(params, *args):
    ind = args[0]
    extra = args[1]
    model = fitFunc(params, ind, extra)
    data = args[2]
    errors = args[3]
    return chiFunc(model, data, errors, extra)


def boundsFunc(parameters, bounds):
    if parameters[0] < bounds[0][0] or parameters[1] < bounds[1][0] or\
       parameters[2] < bounds[2][0] or parameters[3] < bounds[3][0]:
        return 1e99
    if parameters[0] > bounds[0][1] or parameters[1] > bounds[1][1] or\
       parameters[2] > bounds[2][1] or parameters[3] > bounds[3][1]:
        return 1e99
    rho = parameters[2]/(parameters[0]**0.5*parameters[1]**0.5)
    if rho**2 >= 1:
        return 1e99
    return 0


def processRecord(record, image, variance, psf, it=ITERATIONS):
    centroid = record.getCentroid() - image.getXY0()
    centerX = centroid.getX()
    centerY = centroid.getY()
    extra = (centerX, centerY)

    data = image.array
    var = variance.array
    ind = np.indices(data.shape)

    peak = record.getFootprint().peaks[0].getPeakValue()

    parameters = np.array([psf.getIxx(), psf.getIyy(), psf.getIxy(), peak])
    stepSize = abs(parameters)/10.

    constraints = [[0, 100], [0, 100], [-100, 100], [0, 1000]]

    output = demczs(it, data, ind, var, fitFunc, chiFunc, boundsFunc,
                    parameters, stepSize, constraints, extra,  16,
                    int(THINNING), 4, hist_mult=1000)
    return output

def trianglePlot(data, start):
    numParams = data.shape[1]
    plt.figure()
    fig = 1
    labs = ['xx', 'yy', 'xy', 'h']
    for i in range(numParams):
        for j in range(i+1, numParams):
            number, xedge, yedge = np.histogram2d(data[start:,i], data[start:,j], bins=20)
            xmid = xedge[0:-1] + (xedge[1]-xedge[0])/2.
            ymid = yedge[0:-1] + (yedge[1]-yedge[0])/2.
            xind, yind = np.meshgrid(ymid, xmid)
            fig = i*(numParams-1) + j
            plt.subplot(numParams-1, numParams-1, fig)
            plt.contour(yind, xind, number)
            plt.xlabel(labs[i])
            plt.ylabel(labs[j])
            plt.subplots_adjust(wspace=0.73, hspace=0.73)
    plt.show()


# stampPath = '/home/nate2/shape_work/results/stamps.p'
# variancePath = '/home/nate2/shape_work/results/variance.p'
# psfPath = '/home/nate2/shape_work/results/psf.p'
# catPath = '/home/nate2/shape_work/results/catalog.fits'

stampPath = '/Users/nate/random_lsst_scripts/shapeWork/shape_work/results/stamps.p'
replacedPath = '/Users/nate/random_lsst_scripts/shapeWork/shape_work/results/replaced.p'
variancePath = '/Users/nate/random_lsst_scripts/shapeWork/shape_work/results/variance.p'
psfPath = '/Users/nate/random_lsst_scripts/shapeWork/shape_work/results/psf.p'
catPath = '/Users/nate/random_lsst_scripts/shapeWork/shape_work/results/catalog.fits'
psfCatPath = '/Users/nate/random_lsst_scripts/shapeWork/shape_work/results/psfCatalog.fits'
plotsPath = '/Users/nate/random_lsst_scripts/shapeWork/shape_work/results/plots/'

with open(stampPath, 'rb') as f:
    stamps = pickle.load(f)

with open(replacedPath, 'rb') as f:
    replaced = pickle.load(f)

with open(variancePath, 'rb') as f:
    variance = pickle.load(f)

with open(psfPath, 'rb') as f:
    psfList = pickle.load(f)

psfCatalog = afwTable.ExposureCatalog.readFits(psfCatPath)

catalog = afwTable.SourceCatalog.readFits(catPath)

results = []
size = len(catalog)
iterable = np.array(range(size),dtype=int)
subSize = 50
sample = list(set(np.random.randint(0, size, subSize)))
for i in iterable[sample]:
    #results.append(processRecord(catalog[int(i)], replaced[int(i)], variance[int(i)],
    #               psfList[int(i)]))
    out = processRecord(catalog[int(i)], replaced[int(i)], variance[int(i)],
                        psfList[int(i)])
    f = mpl.figure.Figure()
    canvas = FigureCanvas(f)
    plotEllipse(replaced[int(i)], catalog[int(i)], extra=out[0][:-1], fig=f, show=False)
    canvas.print_figure(plotsPath+"{}.png".format(i))
    del f, canvas

for i, result in enumerate(results):
    f = plt.figure()
    plotEllipse(replaced[sample[i]], catalog[sample[i]], 'base_SdssShape', extra=result[0][:-1], fig=f, show=False)
    f.savefig(plotsPath+"{}.png".format(sample[i]))
'''
#bonus hsmwork
n=30594
import lsst.meas.extensions.shapeHSM as hsm
import lsst.meas.base as measBase
center = catalog[n].getCentroid()
ctrl = hsm.HsmSourceMomentsControl()

schem = afwTable.SourceTable.makeMinimalSchema()
schem.setAliasMap(catalog.schema.getAliasMap())
inst = hsm.HsmSourceMomentsAlgorithm(ctrl, 'nate_test',schem)
mask = afwImage.Mask(replaced[n].getBBox())
var = afwImage.ImageF(replaced[n].getBBox())
var.array[:, :] = np.std(replaced[n].array)
maskedImage = afwImage.MaskedImageF(replaced[n],mask,var)
exp = afwImage.ExposureF(maskedImage)
exp.setPsf(psfCatalog[n].getPsf())
inst.measure(catalog[n], exp)

sdssCtrl =  measBase.sdssShape.SdssShapeControl()
sdssAlg = measBase.sdssShape.SdssShapeAlgorithm(sdssCtrl, 'base_SdssShape', schem)




'''
# scipy minimization
outPrime = np.array([np.median(out[3][:, i]) for i in range(4)])
centroid = catalog[4572].getCentroid() - stamps[4572].getXY0()
centerX = centroid.getX()
centerY = centroid.getY()
spOut = minimize(scipyMinFunc, outPrime, args=(np.indices(replaced[4572].array.shape), [centerX, centerY], stamps[4572].array, variance[4572].array))
