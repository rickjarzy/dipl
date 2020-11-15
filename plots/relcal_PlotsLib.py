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

def GetRasterNames(srchPattern,roots):
    
    epoch = roots.split("\\")[-1]
    sat     = epoch[:3]         # Satname
    pr      = epoch[3:9]        # Path and Row
    year    = epoch[9:13]       # Year
    doy     = epoch[13:16]      # DOY

    if sat == 'LE7':
        satAbsName = 'ls7.'+pr+'.'+year+doy+'.msp'
        PifName = 'ls7_'+pr+'_'+year+doy+'_msp_NoComp_PIF.tif'
        RbaName = 'ls7_'+pr+'_'+year+doy+'_msp_rc.tif'
    elif sat == 'LT5':
        satAbsName = 'ls5.'+pr+'.'+year+doy+'.msp'
        PifName = 'ls5_'+pr+'_'+year+doy+'_msp_NoComp_PIF.tif'
        RbaName = 'ls5_'+pr+'_'+year+doy+'_msp_rc.tif'
        
    AbsRas = satAbsName+'.tif'
    RelRas = satAbsName+srchPattern[1:]



    # find Pifs
    

    return os.path.join(roots,AbsRas),os.path.join(roots,RelRas),os.path.join(roots,PifName),os.path.join(roots,RbaName),int(year),int(doy)



def ReadOutSatData(absEpochs,shp_x,shp_y):
    
    if absEpochs==None:
        rasData = 0
        
    else:

        ras = gdal.Open(absEpochs, gdal.GA_ReadOnly)
        rasBand = ras.GetRasterBand(VisBand)
        rasGeoRef = ras.GetGeoTransform()

        ras_xor = rasGeoRef[0]      # X Origin of LandsatScene
        ras_yor = rasGeoRef[3]      # Y origin of LandsatScene
        ras_res = rasGeoRef[1]
        ras_cols = ras.RasterXSize
        ras_rows = ras.RasterYSize
        ras_X_ul = ras_xor
        ras_Y_ul = ras_yor
        ras_X_lr = ras_xor+rasGeoRef[1]*ras_cols
        ras_Y_lr = ras_yor+rasGeoRef[5]*ras_rows


        if shp_x >= ras_X_ul or shp_x <= ras_X_lr:
            if shp_y <= ras_Y_ul and shp_y >= ras_Y_lr:
                print "Shp Coords in SatExtend"
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
