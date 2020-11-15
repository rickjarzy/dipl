import os, glob, osgeo, numpy
from osgeo import gdal, gdalconst, ogr
from  matplotlib import pyplot as plt

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



if __name__ == "__main__":

    # USER INPUT"
    tile = "\\h18v04"
    VisBand = 2
    band = 2
    qBand = "BRDF_Albedo_Quality"
    fitBand = str(band)+'.Polyfit.NanSum.W15'
    #==============
    
    inDir = r"L:\MODIS_TILES_STACKED\h18v04"
    shpDir = r"L:\LANDSAT_STACK\S2BIOM\shp\POI.shp"
    rasData = []
    shpPoiKoords=[]
    doy = []
    os.chdir(inDir)
    
    #collecting RasFilePaths
    for root, dirs, files in os.walk(inDir):
        os.chdir(root)
        rasFiles = glob.glob('*.tif')
        if rasFiles:
            #rasData.extend(rasFiles)
            for h in rasFiles:
                print h
                rasData.append(os.path.join(root,h))
                tiles = h.split(".")
                doy.append(tiles[1][1:])
                
        else:
            continue

    shpDrv = ogr.GetDriverByName('ESRI Shapefile')
    ds = shpDrv.Open(shpDir)
    lyr = ds.GetLayer()
    numPoi = lyr.GetFeatureCount()

    #iterate through Features

    for ft in range(0,numPoi,1):
        actPoi = lyr.GetFeature(ft)
        shpPoiGeom = actPoi.GetGeometryRef()
        xPoi = shpPoiGeom.GetX()
        yPoi = shpPoiGeom.GetY()

        shpPoiKoords.append([xPoi,yPoi])

    
    for shp in range(0,len(shpPoiKoords),1):
        print "Proccessing POI Nr :",shp+1," Koords : ",shpPoiKoords[shp]
        shp_x = shpPoiKoords[shp][0]
        shp_y = shpPoiKoords[shp][1]


        # Listen fuer die auszulesenden daten
        fitValues = numpy.zeros([len(rasData),1],dtype=int)
        
        
        
        for e in range(0,len(rasData),1):
            
            fitValues[e]=ReadOutSatData(rasData[e],VisBand,shp_x,shp_y)
            
      
                

       
        # Plot der ergebnisse
        
        xArr = range(1,len(fitValues)+1,1)
        fignr1 = plt.figure(figsize=(20,15))
        
        plt.plot(xArr,numpy.array(fitValues),'co',markersize=3, linestyle='--',label=r'SG Fit $\ f(t)=a_0+a_1 t + a_2t^2$')
                      
        plt.ylim(0,5000)
        plt.xlim(0,len(fitValues))
        plt.xticks(xArr[0::3],doy[0::3],rotation='vertical')
        figTitle = 'gefilterten MODIS-Zeitreihe ShpNr '+ str(shp)
        plt.title(figTitle)
        plt.xlabel('Zeit [Day Of Year]')
        plt.ylabel('Reflexionswert MODIS Band '+str(band)+'[%]')
        plt.legend()
        plt.grid()
        plt.show()
        #figName = os.path.join(plotsDir, figTitle)
        #fignr1.savefig(figName, dpi=300)    

    print "Programm ENDE"
