# ====================== Create Time-Series-Plots ========================

import os, osgeo,numpy, glob, matplotlib
from osgeo import gdal, ogr
from osgeo.gdal import *
from osgeo.ogr import *     
from matplotlib import pyplot as plt


def ReadOutModisSatData(absEpochs,VisBand,shp_x,shp_y):
    
    if absEpochs==None:
        rasData = 0
        
    else:

        ras = gdal.Open(absEpochs, gdal.GA_ReadOnly)
        rasBand = ras.GetRasterBand(VisBand)
        rasGeoRef = ras.GetGeoTransform()

        ras_xor = rasGeoRef[0]      # X Origin of satScene
        ras_yor = rasGeoRef[3]      # Y origin of satScene
        ras_res = rasGeoRef[1]
        ras_cols = ras.RasterXSize
        ras_rows = ras.RasterYSize
        ras_X_ul = ras_xor
        ras_Y_ul = ras_yor
        ras_X_lr = ras_xor+rasGeoRef[1]*ras_cols
        ras_Y_lr = ras_yor+rasGeoRef[5]*ras_rows


        if shp_x >= ras_X_ul or shp_x <= ras_X_lr:
            if shp_y <= ras_Y_ul and shp_y >= ras_Y_lr:
                #print "Shp Coords in SatExtend"
                ras_x_ind_BildMatrix = int(abs((abs(ras_xor)-abs(shp_x))/ras_res))
                ras_y_ind_BildMatrix = int(abs((abs(ras_yor)-abs(shp_y))/ras_res))
                rasData = rasBand.ReadAsArray(ras_x_ind_BildMatrix,ras_y_ind_BildMatrix,1,1)
                if rasData<=0:
                    rasData=0

            else:
                print "Shp y Coordinate not ins SatSzene Extent"
                rasData=0
                
        else:
            print "Shp x Coordinate not ins SatSzene Extent"
            rasData=0
    return rasData





ras_dir = r"L:\MODIS_TILES_STACKED\h18v04"
shpPOI =  r"L:\LANDSAT_STACK\S2BIOM\shp\POI.shp"
plotsDir = r"L:\LANDSAT_STACK\TSPlots"
shpPoiKoords = []
TSData = []

VisBand = 1

TSBeginn = 2000
TSEnde = 2001

shpDrv = ogr.GetDriverByName('ESRI Shapefile')
ds = shpDrv.Open(shpPOI)
lyr = ds.GetLayer()
numPoi = lyr.GetFeatureCount()

#iterate through Features

for ft in range(0,numPoi,1):
    actPoi = lyr.GetFeature(ft)
    shpPoiGeom = actPoi.GetGeometryRef()
    xPoi = shpPoiGeom.GetX()
    yPoi = shpPoiGeom.GetY()

    shpPoiKoords.append([xPoi,yPoi])




for roots, dirs, files in os.walk(ras_dir):

    os.chdir(roots)
    raseps = glob.glob('*.tif')
    if raseps:
        if int(roots.split("\\")[-1])>=TSBeginn and int(roots.split("\\")[-1])<=TSEnde:
            
            
            for e in raseps:
                TSData.append(os.path.join(roots,e))
        else:
            continue
    else:continue

fig = plt.figure(figsize=(100, 15))
for shp in range(0,len(shpPoiKoords),1):
    print "Proccessing POI Nr :",shp+1," Koords : ",shpPoiKoords[shp]
    shp_x = shpPoiKoords[shp][0]
    shp_y = shpPoiKoords[shp][1]

    TSSatData = []
    DOY = []
    for ts in TSData:
        TSSatData.append(ReadOutModisSatData(ts,VisBand,shp_x,shp_y)[0][0])
        DOY.append(ts.split("\\")[-1].split(".")[1][1:])
        
    # Matplotlib
    # ----------
    xAxeArray = numpy.arange(1,len(TSSatData)+1,1)
    sp = fig.add_subplot(1,numPoi,shp+1)
    pw = sp.plot(xAxeArray,TSSatData,'b',markersize=1,label='relativ Kal.')
    sp.set_ylabel('Landsat Reflectence [%]')
    sp.set_xlabel('Year - DOY []')
    sp.set_xticks(numpy.arange(xAxeArray.min(),xAxeArray.max()+1,1.0))
    sp.set_xticklabels(DOY,xAxeArray,rotation=45)
    sp.legend(bbox_to_anchor=(0.30, 1), loc=1)
    sp.grid()
    sp.set_title('Timeseries starting for the interval year :'+str(TSBeginn)+' to year: '+str(TSEnde)+' with shp Fid: '+str(shp)+'\nMCD-Band: '+str(VisBand))
        
        
figName = os.path.join(plotsDir,'FullTimeseries_'+str(TSBeginn)+'_'+str(TSEnde)+'_MCDBand_'+str(VisBand)+'.png')
fig.savefig(figName,dpi=300)    





print "Programm ENDE"
