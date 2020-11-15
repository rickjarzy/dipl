# ====================== Create Time-Series-Plots ========================

import os, osgeo,numpy, glob, matplotlib
from osgeo import gdal, ogr
from osgeo.gdal import *
from osgeo.ogr import *     
from matplotlib import pyplot as plt


def ReadOutSatData(absEpochs,VisBand,shp_x,shp_y):
    
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



# USER INPUT"
tile = "\\h18v04"
band = "Band1"
qBand = "BRDF_Albedo_Quality"
fitBand = band+'.Polyfit.NanSum.W15'
#==============



raw_dir = r"E:\MODIS_Data\tiff\MCD43A4" + tile
fit_dir = r"E:\MODIS_Data\fitted\MCD43A4" + tile
qal_dir = r"E:\MODIS_Data\tiff\MCD43A2" + tile
shp_dir =  r"E:\LANDSAT_STACK\S2BIOM\shp\POI.shp"
plotsDir = r"E:\LANDSAT_STACK\TSPlots"
shpPoiKoords = []
TSData = []
raseps = []
inDir = []
VisBand = 1
rasDataRoot = []
rawData=[]
qalData=[]
fitData=[]
doy = []

inDir.append(raw_dir)
inDir.append(fit_dir)
inDir.append(qal_dir)


shpDrv = ogr.GetDriverByName('ESRI Shapefile')
ds = shpDrv.Open(shp_dir)
lyr = ds.GetLayer()
numPoi = lyr.GetFeatureCount()

#iterate through Features

for ft in range(0,numPoi,1):
    actPoi = lyr.GetFeature(ft)
    shpPoiGeom = actPoi.GetGeometryRef()
    xPoi = shpPoiGeom.GetX()
    yPoi = shpPoiGeom.GetY()

    shpPoiKoords.append([xPoi,yPoi])


os.chdir(fit_dir)
fitRas = glob.glob("*"+fitBand+".tif")

for e in fitRas:
    tiles = e.split(".")
    fitData.append(os.path.join(fit_dir,e))
    rawData.append(os.path.join(raw_dir,tiles[0]+"."+tiles[1]+"."+tiles[2]+"."+tiles[3]+"."+tiles[4]+"."+tiles[5]+"."+tiles[-1]))
    os.chdir(qal_dir)
    qalData.append(glob.glob(tiles[0][:-2]+'A2'+"."+tiles[1]+"*.BRDF_Albedo_Quality."+tiles[-1])[0])
    doy.append(tiles[1][1:])

        
    #for e in raseps:
     #   TSData.append(os.path.join(roots,e))
    
rasDataRoot.append(rawData)
rasDataRoot.append(fitData)
rasDataRoot.append(qalData)


for shp in range(0,len(shpPoiKoords),1):
    print "Proccessing POI Nr :",shp+1," Koords : ",shpPoiKoords[shp]
    shp_x = shpPoiKoords[shp][0]
    shp_y = shpPoiKoords[shp][1]


    # Listen fuer die auszulesenden daten
    rawValues = numpy.zeros([len(fitData),1],dtype=int)
    fitValues = numpy.zeros([len(fitData),1],dtype=int)
    qalValues = numpy.zeros([len(fitData),1],dtype=int)
    
    for e in range(0,len(rawData),1):
        rawValues[e]=ReadOutSatData(rawData[e],VisBand,shp_x,shp_y)
        fitValues[e]=ReadOutSatData(fitData[e],VisBand,shp_x,shp_y)
        qalValues[e]=ReadOutSatData(qalData[e],VisBand,shp_x,shp_y)
        
            
    # umkehr der qualitätsinfo --->255=0, 1=100, 0 == 200
    qalGut = numpy.where(qalValues==0,500,None)
    qalMit = numpy.where(qalValues==1,250,None)
    qalShl = numpy.where(qalValues==255,100,None)


    # NoDatavalues umsetzten
    
    rawValues[rawValues==32767]=0
    
    rawValues[rawValues==0]= numpy.max(rawValues) + (5000 - numpy.max(rawValues))-300
    
    # Plot der ergebnisse
    
    xArr = range(1,len(fitValues)+1,1)
    fignr1 = plt.figure(figsize=(20,15))
    plt.plot(xArr,numpy.array(rawValues),'k*',linestyle='--',label='MODIS Raw')
    plt.plot(xArr,numpy.array(fitValues),'co',markersize=3, linestyle='--',label=r'SG Fit $\ f(t)=a_0+a_1 t + a_2t^2$')
    plt.plot(xArr,numpy.array(qalGut),'go',label=r'quality good')
    plt.plot(xArr,numpy.array(qalMit),'yo',label=r'quality medium')
    plt.plot(xArr,numpy.array(qalShl),'ro',label=r'quality bad')
    plt.ylim(0,5000)
    plt.xlim(0,len(fitValues))
    plt.xticks(xArr[0::3],doy[0::3],rotation='vertical')
    figTitle = 'Vergleich der rohen MODIS-Zeitreihe mit der gefilterten MODIS-Zeitreihe ShpNr '+ str(shp)
    plt.title(figTitle)
    plt.xlabel('Zeit [Day Of Year]')
    plt.ylabel('Reflexionswert MODIS '+band+'[%]')
    plt.legend()
    plt.grid()
    plt.show()
    figName = os.path.join(plotsDir, figTitle + "_" + band )
    fignr1.savefig(figName, dpi=300)
    
    
        
        
    





print "Programm ENDE"
