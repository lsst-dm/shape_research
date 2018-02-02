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
truthPath = '/home/nate2/shape_work/acs_clean_only.fits'
tract = 9813
filt = "HSC-I"
patches = ['{},{}'.format(x, y) for x in range(4, 7) for y in range(4, 7)]
repo = Butler(repoPath)
truthCat = fits.open(truthPath)[1].data

images = []
catalogs = []

for patch in patches:
    dataId = {'tract': tract, 'patch': patch, 'filter': filt}
    tmpIm = repo.get('deepCoadd_calexp', dataId)
    tmpCat = repo.get('deepCoadd_meas', dataId)
    images.append(tmpIm)
    catalogs.append(tmpCat)

cutFunctions = []
baseHsmKey = "ext_shapeHSM_HsmSourceMoments"
baseSimpleKey = "ext_simpleShape_SimpleShape"
baseSdssKey = "base_SdssShape"
allKeys = (baseHsmKey, baseSimpleKey, baseSdssKey)
extensions = ("_xx",  "_yy", "_xy")
productList = [x+y for x, y in it.product(allKeys, extensions)]


def plotEllipse(stamp, record, shapeKey, fig=plt.Figure(), position=111,
                show=True):
    ax = fig.add_subplot(position)
    ax.imshow(stamp.array)
    eKey = afwTable.QuadrupoleKey(record.schema[shapeKey])
    ellipse = afwGeom.ellipses.Axes(record.get(eKey))
    centerE = record.getCentroid() - stamp.getXY0()
    centerE = centerE.asPoint()
    center = (centerE.getX(), centerE.getY())
    ax.add_patch(mpatch.Ellipse(center, ellipse.getA(), ellipse.getB(),
                                ellipse.getTheta(), fill=False))
    if show:
        plt.show()


def matchCatalogs(catalog):
    raKey = 'coord_ra'
    decKey = 'coord_dec'
    truthRaKey = 'ALPHAPEAK_J2000'
    truthDecKey = 'DELTAPEAK_J2000'
    hscRa = np.array([rec[raKey].asDegrees() for rec in catalog])
    hscDec = np.array([rec[decKey].asDegrees() for rec in catalog])
    hscCoords = ICRS(hscRa*u.deg, hscDec*u.deg)
    truthRa = truthCat[truthRaKey]
    truthDec = truthCat[truthDecKey]
    truthCoords = ICRS(truthRa*u.deg, truthDec*u.deg)
    idx, d2d, d3d = match_coordinates_sky(hscCoords, truthCoords)
    # This corresponds to ten arc seconds in degrees
    threshold = 10*(1/3600.)
    matchMask = np.array(d2d) <= threshold
    return catalog[matchMask], idx[matchMask]


def cutDec(func):
    cutFunctions.append(func)
    return func


@cutDec
def isNan(catalog, extra=None):
    mask = np.ones(len(catalog), dtype=bool)
    for product in productList:
        column = catalog[product]
        mask *= np.isfinite(column)
    return mask == False


@cutDec
def checkFlag(catalog, extra=None):
    mask = np.zeros(len(catalog), dtype=bool)
    for key in allKeys:
        column = catalog[key+"_flag"]
        mask += column == True
    return mask


def iterativeFilter(array, number=10):
    mask = np.ones(len(array), dtype=bool)
    for i in range(number):
        expectation = np.median(array[mask])
        std = np.std(array[mask]-expectation)
        mask *= abs(array - expectation) < 3*std
    return mask == False


def piecewise(difference, mag, slope=-2):
    totalmask = np.ones(len(difference), dtype=bool)
    magMask = mag < 22
    brightMask = difference[magMask] > 10
    totalmask[magMask] *= brightMask
    objective = slope*(mag[magMask == False]-22)+20
    faintMask = abs(difference[magMask == False]) > objective
    totalmask[magMask == False] *= faintMask
    return totalmask == False


@cutDec
def checkShape(catalog, extra):

    mask = np.ones(len(catalog), dtype=bool)
    subTruth = truthCat[extra[0]]
    psf = extra[1]
    tempMags = -2.5*np.log10(catalog['modelfit_CModel_flux']) + 27
    for key in allKeys:
        hscXX = catalog[key+'_xx']
        hstXX = subTruth['CXX_IMAGE']
        # 11.56 is the conversion between hst and hsc pixels**2
        xx = hscXX - psf.getIxx() - hstXX*11.56
        xxFinite = xx[np.isfinite(xx)]
        xxPiecewiseMask = piecewise(xxFinite, tempMags[np.isfinite(xx)])
        mask[np.isfinite(xx)] *= xxPiecewiseMask

        hscYY = catalog[key+'_yy']
        hstYY = subTruth['CYY_IMAGE']
        yy = hscYY - psf.getIyy() - hstYY*11.56
        yyFinite = yy[np.isfinite(yy)]
        yyPiecewiseMask = piecewise(yyFinite, tempMags[np.isfinite(yy)])
        mask[np.isfinite(yy)] *= yyPiecewiseMask

        hscXY = catalog[key+'_xy']
        hstXY = subTruth['CXY_IMAGE']
        xy = hscXY - psf.getIxy() - hstXY*11.56
        xyFinite = xy[np.isfinite(xy)]
        xyPiecewiseMask = piecewise(xyFinite, tempMags[np.isfinite(xy)])
        mask[np.isfinite(xy)] *= xyPiecewiseMask

    return mask == False


@cutDec
def checkVariance(catalog, extra=None):
    threshold = 3
    mask = np.ones(len(catalog), dtype=bool)
    numComponents = 3
    tmpContainer = np.zeros((len(allKeys), numComponents, len(catalog)),
                            dtype=float)
    for i, key in enumerate(allKeys):
        for j, record in enumerate(catalog):
            xx = record[key+"_xx"]
            yy = record[key+"_yy"]
            xy = record[key+"_xy"]
            quad = afwGeom.ellipses.Quadrupole(xx, yy, xy)
            ax = afwGeom.ellipses.Axes(quad)
            tmpContainer[i, :, j] = (ax.getA(), ax.getB(), ax.getTheta())
    medVal = np.median(tmpContainer, axis=0)
    stdVal = np.std(tmpContainer, axis=0)
    for i in range(len(allKeys)):
        for j in range(numComponents):
            mask *= abs(tmpContainer[:] - medVal)[i, j, :] < threshold*stdVal[j, :]
    return mask == False


@cutDec
def checkAxisRatio(catalog, extra):
    mask = np.ones(len(catalog), dtype=bool)
    lowerLimit = 1/5.
    upperLimit = 5.
    for key in allKeys:
        xx = catalog[key+"_xx"]
        yy = catalog[key+"_yy"]
        ratio = xx/yy
        mask *= ratio > lowerLimit
        mask *= ratio < upperLimit
    return mask == False


def checkParent(catalog):
    primary = catalog['detect_isPrimary'] == True
    child = catalog['deblend_nChild'] == 0
    return primary*child


def filterObjects(image, catalog, positions, expCat):
    # filter the input catalog to objects that can be found in the HST catalog
    imageList = []
    replacedList = []
    varList = []
    mask = np.zeros(len(catalog), dtype=bool)
    psfShape = image.getPsf().computeShape()
    for filtFunc in cutFunctions:
        mask += filtFunc(catalog, (positions, psfShape))
    # filter for parents
    parentMask = checkParent(catalog)
    mask *= parentMask
    filteredCat = catalog[mask]
    noiseReplaceImage = afwImage.ExposureF(image, deep=True)
    noiseReplacer = mil.rebuildNoiseReplacer(noiseReplaceImage, catalog)
    for record in filteredCat:
        noiseReplacer.insertSource(record.getId())
        bbox = record.getFootprint().getBBox()
        if bbox.getWidth() == 0 or bbox.getHeight == 0:
            center = record.getCentroid()
            bbox = afwGeom.Box2I(afwGeom.Point2I(center.getX()-10,
                                                 center.getY()-10),
                                 afwGeom.Point2I(center.getX()+10,
                                                 center.getY()+10))
        bbox.grow(10)
        bbox.clip(image.getImage().getBBox())
        imageList.append(afwImage.ImageF(image.getImage(), bbox, deep=True))
        replacedList.append(afwImage.ImageF(noiseReplaceImage.getImage(), bbox,
                                            deep=True))
        varList.append(afwImage.ImageF(image.getVariance(), bbox, deep=True))
        noiseReplacer.removeSource(record.getId())
        rec = expCat.addNew()
        rec.setPsf(image.getPsf())
    noiseReplacer.end()
    return imageList, replacedList, varList, filteredCat,\
        [psfShape]*len(varList)


matchedCat = []
truthPositions = []
for i, cat in enumerate(catalogs):
    print("Matching catalog {}".format(i))
    match, positions = matchCatalogs(cat)
    matchedCat.append(match)
    truthPositions.append(positions)

truthPositions = np.array(truthPositions)

stamps = []
replaced = []
variance = []
filteredCat = []
psfList = []
expCat = afwTable.ExposureCatalog(afwTable.ExposureTable.makeMinimalSchema())

place = 0
for im, cat, positions in zip(images, matchedCat, truthPositions):
    print("processing {}".format(place))
    result = filterObjects(im, cat, positions, expCat)
    stamps += result[0]
    replaced += result[1]
    variance += result[2]
    filteredCat.append(result[3])
    psfList += result[4]
    place += 1

finalCatalog = afwTable.SourceCatalog(filteredCat[0].schema)
finalCatalog.reserve(len(stamps))
for n in range(len(filteredCat)):
    for record in filteredCat[n]:
        tmpRecord = finalCatalog.addNew()
        tmpRecord.assign(record)

# random extra
xx = finalCatalog[allKeys[0]+"_xx"]
yy = finalCatalog[allKeys[0]+"_yy"]
xy = finalCatalog[allKeys[0]+"_xy"]
tempMags = -2.5*np.log10(finalCatalog['modelfit_CModel_flux']) + 27

count = 0
for cat in matchedCat:
    count += len(cat)

totalCatalog = afwTable.SourceCatalog(matchedCat[0].schema)
totalCatalog.reserve(count)
for n in range(len(matchedCat)):
    for record in matchedCat[n]:
        tmpRecord = totalCatalog.addNew()
        tmpRecord.assign(record)

xxFull = totalCatalog[allKeys[0]+"_xx"]
yyFull = totalCatalog[allKeys[0]+"_yy"]
xyFull = totalCatalog[allKeys[0]+"_xy"]
tempMagsFull = -2.5*np.log10(totalCatalog['modelfit_CModel_flux']) + 27


with open('/home/nate2/shape_work/results/stamps.p', 'wb') as f:
    pickle.dump(stamps, f)
with open('/home/nate2/shape_work/results/replaced.p', 'wb') as f:
    pickle.dump(replaced, f)
with open('/home/nate2/shape_work/results/variance.p', 'wb') as f:
    pickle.dump(variance, f)
with open('/home/nate2/shape_work/results/psf.p', 'wb') as f:
    pickle.dump(psfList, f)
expCat.writeFits('/home/nate2/shape_work/results/psfCatalog.fits')
finalCatalog.writeFits('/home/nate2/shape_work/results/catalog.fits')
