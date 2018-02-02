import itertools as it
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
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

for i, record in enumerate(totalCatalog):
    gausFlux = record[fluxKey]
    if np.isnan(gausFlux):
        continue
    quadShape = record[SdssKey]
    if np.isnan(quadShape.getIxx()):
        quadShape = record[HsmKey]
        if np.isnan(quadShape.getIxx()):
            continue
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors.kde import KernelDensity

flux = np.load('shape_work/results/gausFlux.npy')
radius = np.load('shape_work/results/conRad.npy')
shape = np.load('shape_work/results/conShape.npy')
positions = np.load('shape_work/results/positions.npy')
totalCatalogPath = 'shape_work/results/totalCatalog.fits'
totalCatalog = afwTable.SourceCatalog.readFits(totalCatalogPath)

fig = plt.figure()

extend = totalCatalog['base_ClassificationExtendedness_value'][positions]
gal = extend == 1
star = extend == 0

plt.semilogy(-2.5*np.log10(flux[star])+27, radius[star], '.')
plt.semilogy(-2.5*np.log10(flux[gal])+27, radius[gal], '.')

plt.semilogy(-2.5*np.log10(flux[star])+27, shape[star], '.')
plt.semilogy(-2.5*np.log10(flux[gal])+27, shape[gal], '.', alpha=0.5)

plt.loglog(shape[star], radius[star], '.')
plt.loglog(shape[gal], radius[gal], '.')

good = np.zeros(len(totalCatalog), dtype=bool)
failed = totalCatalog.schema.extract('*_flag').keys()
failed = ['base_SdssShape_flag']
for key in failed:
    good += totalCatalog[key]

goodFlip = good == False
goodTrim = goodFlip[positions]

badTrim = good[positions]

plt.semilogy(-2.5*np.log10(flux[goodTrim])+27, radius[goodTrim], '.')
plt.semilogy(-2.5*np.log10(flux[badTrim])+27, radius[badTrim], '.')

weirdPlace = np.where((radius[badTrim] < 3.3) * (radius[badTrim] > 2.25))

plt.figure()
plt.semilogy(-2.5*np.log10(flux[goodTrim])+27, shape[goodTrim], '.')
plt.semilogy(-2.5*np.log10(flux[badTrim])+27, shape[badTrim], '.')
plt.semilogy(-2.5*np.log10(flux[badTrim][weirdPlace])+27, shape[badTrim][weirdPlace], '.')

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
