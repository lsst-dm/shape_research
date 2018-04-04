import itertools as it
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatch
import numpy as np
import pickle

from astropy.io import fits
from astropy.coordinates import ICRS, match_coordinates_sky
from astropy import units as u

from lsst.daf.persistence import Butler
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable

from lsst.meas.base import measurementInvestigationLib as mil


repoPath = '/datasets/hsc/repo/rerun/private/nate/shapeTest2'
tract = 9813
filt = "HSC-I"
patches = ['{},{}'.format(x, y) for x in range(4, 7) for y in range(4, 7)]
repo = Butler(repoPath)

images = []
catalogs = []

for patch in patches:
    dataId = {'tract': tract, 'patch': patch, 'filter': filt}
    tmpIm = repo.get('deepCoadd_calexp', dataId)
    tmpCat = repo.get('deepCoadd_meas', dataId)
    images.append(tmpIm)
    catalogs.append(tmpCat)


count = 0
for cat in catalogs:
    count += len(cat)

totalCatalog = afwTable.SourceCatalog(catalogs[0].schema)
totalCatalog.reserve(count)
imageId = []
for n in range(len(catalogs)):
    for record in catalogs[n]:
        tmpRecord = totalCatalog.addNew()
        tmpRecord.assign(record)
        imageId.append(n)


baseHsmKey = "ext_shapeHSM_HsmSourceMoments"
baseSdssKey = "base_SdssShape"
fluxKey = "base_GaussianFlux_flux"

SdssKey = afwTable.QuadrupoleKey(totalCatalog.schema[baseSdssKey])
HsmKey = afwTable.QuadrupoleKey(totalCatalog.schema[baseHsmKey])

flux = []
radius = []
shape = []
positions = []


catLen = len(totalCatalog)
for i, record in enumerate(totalCatalog):
    if i % 1000 == 0:
        print("Record {}  out of {}".format(i, catLen))

    gausFlux = record[fluxKey]
    if np.isnan(gausFlux):
        continue
    psfShape = np.array([record['slot_PsfShape_xx'],
                         record['slot_PsfShape_yy'],
                         record['slot_PsfShape_xy']])
    quadShape = record[SdssKey]
    if np.isnan(quadShape.getIxx()):
        quadShape = record[HsmKey]
        if np.isnan(quadShape.getIxx()):
            continue
    intermediateShape = quadShape.getParameterVector() - \
        psfShape
    intermediateShape += np.array([4, 4, 0])
    quadShape = afwGeom.ellipses.Quadrupole(*intermediateShape)
    sep = afwGeom.ellipses.SeparableConformalShearTraceRadius(quadShape)
    rad = sep.getDeterminantRadius()
    if np.isnan(rad):
        continue
    shp = np.sqrt(sep.getE1()**2+sep.getE2()**2)
    if np.isnan(shp):
        continue
    flux.append(gausFlux)
    radius.append(rad)
    shape.append(shp)
    positions.append(i)

flux = np.array(flux)
radius = np.array(radius)
shape = np.array(shape)
positions = np.array(positions)

gausFluxPath = '/home/nate2/shape_work/results/gausFlux.npy'
conRadPath = '/home/nate2/shape_work/results/conRad.npy'
conShapePath = '/home/nate2/shape_work/results/conShape.npy'
positionsPath = '/home/nate2/shape_work/results/positions.npy'
totalCatalogPath = '/home/nate2/shape_work/results/totalCatalog.fits'

np.save(gausFluxPath, flux)
np.save(conRadPath, radius)
np.save(conShapePath, shape)
np.save(positionsPath, positions)
totalCatalog.writeFits(totalCatalogPath)


### Ploting

import numpy as np
import lsst.afw.table as afwTable
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

flux = np.load('../shape_work/results/gausFlux.npy')
radius = np.load('../shape_work/results/conRad.npy')
shape = np.load('../shape_work/results/conShape.npy')
positions = np.load('../shape_work/results/positions.npy')
totalCatalogPath = '../shape_work/results/totalCatalog.fits'
totalCatalog = afwTable.SourceCatalog.readFits(totalCatalogPath)

fig = plt.figure()


extend = totalCatalog['base_ClassificationExtendedness_value'][positions]
gal = extend == 1
star = extend == 0

good = np.zeros(len(totalCatalog), dtype=bool)
failed = totalCatalog.schema.extract('*_flag').keys()
failed = ['base_SdssShape_flag']
for key in failed:
    good += totalCatalog[key]

goodFlip = good == False
goodTrim = goodFlip[positions]

badTrim = good[positions]

starGood = np.bitwise_and(star, goodTrim)
galGood = np.bitwise_and(gal, goodTrim)

starBad = np.bitwise_and(star, badTrim)
galBad = np.bitwise_and(gal, badTrim)

fluxStarGood = flux[starGood]
bcfluxStar, bcfluxStarLambda = sp.stats.boxcox(fluxStarGood)
bcRadiusStar, bcRadiusStarLambda = sp.stats.boxcox(radius[starGood])
bcShapeStar, bcShapeStarLambda = sp.stats.boxcox(shape[starGood])

starGausMix = BayesianGaussianMixture(4)
indDataStar = np.vstack([bcfluxStar, bcRadiusStar, bcShapeStar]).transpose()
starGausMix.fit(indDataStar)

fluxGalGood = flux[galGood]
bcfluxGal, bcfluxGalLambda = sp.stats.boxcox(fluxGalGood)
bcRadiusGal, bcRadiusGalLambda = sp.stats.boxcox(radius[galGood])
bcShapeGal, bcShapeGalLambda = sp.stats.boxcox(shape[galGood])


indDataGal = np.vstack([bcfluxGal, bcRadiusGal, bcShapeGal]).transpose()
H, edges = np.histogramdd(indDataGal, bins=30)
fluxEdges = edges[0]
radiusEdges = edges[1]
shapeEdges = edges[2]
fluxMid = edges[0][:-1] + (edges[0][1]-edges[0][0])
radiusMid = edges[1][:-1] + (edges[1][1]-edges[1][0])
shapeMid = edges[2][:-1] + (edges[2][1]-edges[2][0])

binsToDrop = np.where(np.bitwise_and(H<25,H>0))
fluxBinsBad = binsToDrop[0]
radiusBinsBad = binsToDrop[1]
shapeBinsBad = binsToDrop[2]
dropMask = np.ones(len(fluxGalGood),dtype=bool)
for i in range(len(fluxGalGood)):
    if i % 1000 == 0:
        print(i/len(fluxGalGood))
    for j in range(len(binsToDrop[0])):
        fTest = bcfluxGal[i] >= fluxEdges[fluxBinsBad[j]] and bcfluxGal[i] < fluxEdges[fluxBinsBad[j]+1]
        rTest = bcRadiusGal[i] >= radiusEdges[radiusBinsBad[j]] and bcRadiusGal[i] < radiusEdges[radiusBinsBad[j]+1]
        sTest = bcShapeGal[i] >= shapeEdges[shapeBinsBad[j]] and bcShapeGal[i] < shapeEdges[shapeBinsBad[j]+1]
        if fTest and rTest and sTest:
            dropMask[i] = False
            break

H, edges = np.histogramdd(np.vstack([bcfluxGal[dropMask], bcRadiusGal[dropMask], bcShapeGal[dropMask]]).transpose(),bins=30)
fluxind, radind, shapeind = np.indices(H.shape)
nonzero = np.where(H != 0)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(fluxind[nonzero], radind[nonzero], shapeind[nonzero], c=H[nonzero]/np.sum(H))

galGausMix = GaussianMixture(5, covariance_type='full')
indDataGal = np.vstack([bcfluxGal[dropMask], bcRadiusGal[dropMask], bcShapeGal[dropMask]]).transpose()
orDataGal = np.vstack([fluxGalGood, radius[galGood], shape[galGood]]).transpose()
galGausMix.fit(indDataGal)
samples = galGausMix.score_samples(indDataGal)

S, edges, extra = sp.stats.binned_statistic_dd(np.vstack([bcfluxGal[dropMask], bcRadiusGal[dropMask], bcShapeGal[dropMask]]).transpose(),
                                        samples, bins=30)
nonzeroS = np.where(S != 0)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(fluxind[nonzeroS], radind[nonzeroS], shapeind[nonzeroS], c=S[nonzeroS])


magsStarGood = -2.5*np.log10(flux[starGood])+27
magsStarBad = -2.5*np.log10(flux[starBad])+27

plt.semilogy(magsStarGood, radius[starGood], '.')
plt.semilogy(magsStarBad, radius[starBad], '.')

plt.loglog(radius[starGood], shape[starGood], '.')
plt.loglog(radius[starBad], shape[starBad], '.')

plt.loglog(radius[galGood], shape[galGood], '.')
plt.loglog(radius[galBad], shape[galBad], '.')

plt.figure()
plt.semilogy(-2.5*np.log10(flux[goodTrim])+27, shape[goodTrim], '.')
plt.semilogy(-2.5*np.log10(flux[badTrim])+27, shape[badTrim], '.')

plt.figure()
plt.loglog(radius[goodTrim], shape[goodTrim], '.')

sampleMatrix = np.transpose(np.vstack((radius[goodTrim],shape[goodTrim],flux[goodTrim])))

kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sampleMatrix)

slcFlux = np.exp((17.5-27)/-2.5)
slcSamples = []
for i in np.linspace(radius[goodTrim].min(), radius[goodTrim].max(), 100):
    for j in np.linspace(shape[goodTrim].min(), shape[goodTrim].max(), 100):
        slcSamples.append((i, j, slcFlux))

slcSamples = np.array(slcSamples)
slcScores = kde.score_samples(slcSamples)

plt.figure()
plt.scatter(slcSamples[:, 0], slcSamples[:, 1], c=1.3**slcScores)


histout = np.histogramdd(np.transpose(np.vstack((radius, shape, flux))), bins=100)

radMid = histout[1][0][0:] + (histout[1][0][1] - histout[1][0][0])/2.
shpMid = histout[1][1][0:] + (histout[1][1][1] - histout[1][1][0])/2.

newx = []
newy = []
newz = []

for i in range(histout[0].shape[0]):
    for j in range(histout[0].shape[1]):
        for k in range(histout[0].shape[2]):
            if histout[0][i,j,k] != 0:
                newx.append(radMid[i])
                newy.append(shpMid[j])
                newz.append(histout[0][i,j,k])
