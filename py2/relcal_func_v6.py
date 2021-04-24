# Functionlisting for relative calibration of Landsat and MODIS Data
# Paul Arzberger
# 19.04.2013 - V 1.0
# 
import numpy, osgeo, glob, os
from osgeo import gdal, gdalconst, osr
from osgeo.gdal import *
from osgeo.gdalconst import *
from osgeo.osr import *
from scipy import stats, ndimage, odr





def linOdrFit(params,x):
    return params[0]+params[1]*x


def createRas(driver,name,cols,rows,bands,dataType,ref_info):
    ras = driver.Create(name,cols,rows,bands,dataType,options=['COMPRESS=LZW'])
    ras.SetProjection(ref_info[0])
    ras.SetGeoTransform(ref_info[1])

    return ras

def writeOnBand(rasObject,band,data,noData=0):
    rasBand = rasObject.GetRasterBand(band)
    rasBand.SetNoDataValue(noData)
    rasBand.WriteArray(data)
    rasBand.FlushCache()


'''
===== REPROJECTING LANDSAT TO MODIS PROJECTION ========
                    reprostack
=======================================================
repro(  1. - sat_raw        : [string] contains the satellite stack name
        2. - ls_cm          : [string] contains the string to the cloudmask of the actual satellite/landsatszene
        3. - sat_bands      : [array]  contains the bandnumbers which shall be processed
        4. - ref_info_mo    : [array]  contains the reference and extend information for the modis mosaic
        )

return  1. - out_ras        : [object] is the object for the in_memory reprojectetd, resampled satellite/landsat stack
        2. - out_qal        : [object] is the object for the in_memory reprojectetd, resampled satellite/landsat cloudmask
        3. - driver         : [object] driver for GeoTiff
        4. - driver_mem     : [object] driver for in_memory
        
        
'''
def reprostack(sat_raw,ls_cm,sat_bands,ref_info_mo,in_path,CaliDataType):        # in_path loeschen!!!
    ref_info_sat = {}

    

    # select specified landsat bands and create a stack in memory
    
    ras_sat = gdal.Open(sat_raw, GA_ReadOnly)
    
    ref_info_sat[0] = ras_sat.GetProjection()
    ref_info_sat[1] = ras_sat.GetGeoTransform()

    driver_mem = gdal.GetDriverByName("MEM")
    driver_tif = ras_sat.GetDriver()

    cols_sat = ras_sat.RasterXSize
    rows_sat = ras_sat.RasterYSize


    print "Create Reprojectionstack in Memory for :",sat_raw,"\n..."
    out_ras2rep = createRas(driver_mem,"",cols_sat,rows_sat,len(sat_bands),gdalconst.GDT_Int16,ref_info_sat)
    
    for b in range(0,len(sat_bands),1):         # writeing data of landsatstack into file which will be projected
        print "Reading band :\t\t", sat_bands[b]

        writeOnBand(out_ras2rep,b+1,ras_sat.GetRasterBand(int(sat_bands[b])).ReadAsArray().astype(int),0)
        
        
        
    # ========== REPROJECTION Satellite DATA ===================            satellite Date reprojection is correct
    # ==========================================================            checked on 22.01.2014
    print "\nReprojectin' from WGS --> Sinusodial"
    print "------------------------------------\n..."

    # CHANGE HERE!!!!!
    # ================
    nu_res = float("%7.5f" % (ref_info_mo[1][1]/15))               # nu resolution for reporjected Ls-Scene, makes it easier to aggregate
    #=================
    
    ref2rep = osr.SpatialReference()
 
    #ref2rep.ImportFromEPSG(32633)
    ref2rep.ImportFromWkt(ref_info_sat[0])
    sinos = osr.SpatialReference()
 
    #sinos.ImportFromEPSG(6842)
    sinos.ImportFromWkt(ref_info_mo[0])

    rep = osr.CoordinateTransformation(ref2rep,sinos) # define reporjection from wgs84 to sinusodial


    x_size = cols_sat
    y_size = rows_sat

    # Read Out Extend Coordinates of Landsatscene
    left   = ref_info_sat[1][0]
    print "left: \t\t", left
    top    = ref_info_sat[1][3]
    print "top: \t\t", top
    right  = ref_info_sat[1][0] + x_size*ref_info_sat[1][1]
    print "right : \t", right
    bottom = ref_info_sat[1][3] + y_size*ref_info_sat[1][5]
    print "bottom: \t", bottom

    nusat_coords_X = []
    nusat_coords_Y = []
    # Transform Extend Coords to Sinusodial
    #UL = rep.TransformPoint(left,top)
    
    UL = rep.TransformPoint(ref_info_sat[1][0],ref_info_sat[1][3],0)
    nusat_coords_X.append(UL[0])
    nusat_coords_Y.append(UL[1])
    print "UL:\t\t",UL
    UR = rep.TransformPoint(right,top,0)
    nusat_coords_X.append(UR[0])
    nusat_coords_Y.append(UR[1])
    print "UR:\t\t",UR
    LL = rep.TransformPoint(left,bottom,0)
    nusat_coords_X.append(LL[0])
    nusat_coords_Y.append(LL[1])
    print "LL:\t\t",LL
    LR = rep.TransformPoint(right,bottom,0)
    nusat_coords_X.append(LR[0])
    nusat_coords_Y.append(LR[1])
    print "LR:\t\t",LR

    # Create nu GeoTransform for Projected Raster
    '''nuGeoTrans = (UL[0]      --> LeftCoordinate
                     ref_ld[1]  --> Pixel Size e.g.: 30.88..
                     ref_ld[2]  --> 0
                     UR[2]      --> TopCoordinate)
                     ref_ld[4]  --> 0
                     ref_ld[5]  --> -30.88...
                     '''


    
    #nuGeoTrans = (UL[0]+8900, nu_res, ref_info_sat[1][2], UL[1], ref_info_sat[1][4], -nu_res)

    nu_left  = numpy.min(nusat_coords_X)
    nu_right = numpy.max(nusat_coords_Y)
    nuGeoTrans = (nu_left, nu_res, ref_info_sat[1][2], nu_right, ref_info_sat[1][4], -nu_res)
    
    nu_cols = int(numpy.ceil(abs(numpy.max(nusat_coords_X)-numpy.min(nusat_coords_X))/nu_res))  # calculate nu cols and rows for reprojected raster
    nu_rows = int(numpy.ceil(abs(numpy.max(nusat_coords_Y)-numpy.min(nusat_coords_Y))/nu_res))

    # Create Raster for Reprojected Stack in Memory
    ras_out_rep = driver_mem.Create("",nu_cols, nu_rows, len(sat_bands), gdalconst.GDT_Int16)

    for b in range(0,len(sat_bands),1):

        band_out_ras2rep = out_ras2rep.GetRasterBand(b+1)
        bandorig = ras_sat.GetRasterBand(sat_bands[b])
        bandrep  = ras_out_rep.GetRasterBand(b+1)
        
        if CaliDataType == '.abscal_1500':
            bandrep.SetNoDataValue(bandorig.GetNoDataValue())
            band_out_ras2rep.SetNoDataValue(bandorig.GetNoDataValue())
        elif CaliDataType == '.relcal_1500':
            bandrep.SetNoDataValue(0)
            band_out_ras2rep.SetNoDataValue(0)
        
    ras_out_rep.SetProjection(sinos.ExportToWkt())
    ras_out_rep.SetGeoTransform(nuGeoTrans)

    
    repro = gdal.ReprojectImage(out_ras2rep, ras_out_rep, ref2rep.ExportToWkt(), sinos.ExportToWkt(), gdal.GRA_NearestNeighbour)

    #out_ras = driver_mem.CreateCopy("",ras_out_rep,0)
    
    del rep
    # ========== REPROJECTION QUALITY DATA ===================
    # ========================================================
    ref_info_qal = {}
    nuqal_coords_X = []
    nuqal_coords_Y = []
    
    
    # create a Quality raster in memory
    qal = gdal.Open(os.path.join(str(ls_cm[0])), GA_ReadOnly)           # open cloudmask
    ref_info_qal[0] = qal.GetProjection()
    ref_info_qal[1] = qal.GetGeoTransform()
    qalInBand = qal.GetRasterBand(1)
    qal_NoData = qalInBand.GetNoDataValue()

    refqal2rep = osr.SpatialReference()
    #refqal2rep.ImportFromEPSG(32633)
    refqal2rep.ImportFromWkt(ref_info_qal[0])
    
    rep = osr.CoordinateTransformation(refqal2rep,sinos) # define reporjection from wgs84 to sinusodial

    x_size = cols_sat       # landsat scene and mask from absolut calibration have same number of rows and cols (checked 22.01.2014)
    y_size = rows_sat

    # Read Out Extend Coordinates of Landsatscene
    left   = ref_info_qal[1][0]
    top    = ref_info_qal[1][3]
    right  = ref_info_qal[1][0] + x_size*ref_info_qal[1][1]
    bottom = ref_info_qal[1][3] + y_size*ref_info_qal[1][5]

    
    
    # Transform Extend Coords to Sinusodial
    UL = rep.TransformPoint(left,top)
    nuqal_coords_X.append(UL[0])
    nuqal_coords_Y.append(UL[1])

    UR = rep.TransformPoint(right,top)
    nuqal_coords_X.append(UR[0])
    nuqal_coords_Y.append(UR[1])
    
    LL = rep.TransformPoint(left,bottom)
    nuqal_coords_X.append(LL[0])
    nuqal_coords_Y.append(LL[1])
    
    LR = rep.TransformPoint(right,bottom)
    nuqal_coords_X.append(LR[0])
    nuqal_coords_Y.append(LR[1])

    
    # Create nu GeoTransform for Projected Raster
    '''nuGeoTrans = (UL[0]      --> LeftCoordinate
                     ref_ld[1]  --> Pixel Size e.g.: 30.88..
                     ref_ld[2]  --> 0
                     UR[2]      --> TopCoordinate)
                     ref_ld[4]  --> 0
                     ref_ld[5]  --> -30.88...
                     '''
    nu_left  = numpy.min(nuqal_coords_X)
    nu_right = numpy.max(nuqal_coords_Y)
    nuGeoTrans = (nu_left, nu_res, ref_info_qal[1][2], nu_right, ref_info_qal[1][4], -nu_res)

    nu_cols = int(numpy.ceil(abs(numpy.max(nuqal_coords_X)-numpy.min(nuqal_coords_X))/nu_res))
    nu_rows = int(numpy.ceil(abs(numpy.max(nuqal_coords_Y)-numpy.min(nuqal_coords_Y))/nu_res))

    
    # create a Raster for the Reprojected Quality Data in memory
    #qal_Band = qal.GetRasterBand(1)
    #qal_data = qal_Band.ReadAsArray()
    
    
    qal_reprojected = driver_mem.Create("", nu_cols, nu_rows,1,gdalconst.GDT_Byte)
    qal_reprojected.SetProjection(sinos.ExportToWkt())
    qal_reprojected.SetGeoTransform(nuGeoTrans)
    qalOutBand = qal_reprojected.GetRasterBand(1)
    qalOutBand.SetNoDataValue(qal_NoData)
    
    repro_qal = gdal.ReprojectImage(qal,qal_reprojected, refqal2rep.ExportToWkt(), sinos.ExportToWkt(), gdal.GRA_NearestNeighbour)
    print "Storing reprojected Raster in RAM\n..."
    
##    
    
    
    
    #out_qal = driver_mem.CreateCopy("", qal_reprojected, 0)
    
    # Create Pyhsical Copies of reprojected Landsat And MAsk
    # ------------------------------------------------------
    nameSatRas = os.path.join(in_path,sat_raw[:-4]+".rep"+CaliDataType+".tif")
    nameQalRas = os.path.join(in_path,ls_cm[0][:-4]+".rep"+CaliDataType+".tif")
    print "Create physic copies of reprojected Rasters\nSat_Raster :\t\t",nameSatRas,"\nQal Raster :\t\t",nameQalRas
    check = driver_tif.CreateCopy(nameSatRas,ras_out_rep,0,options=['COMPRESS=LZW'])
    checkq = driver_tif.CreateCopy(nameQalRas,qal_reprojected,0,options=['COMPRESS=LZW'])
    nusat_coords = []
    nuqal_coords = []
    nusat_coords.append(nusat_coords_X)
    nusat_coords.append(nusat_coords_Y)
    nuqal_coords.append(nuqal_coords_X)
    nuqal_coords.append(nuqal_coords_Y)

    print "Stacking and Reprojection complete!\n"
    return ras_out_rep,qal_reprojected,driver_tif,driver_mem
'''
===== REPROJECTIN LANDSAT TO MODIS PROJECTION ========
                        repro
======================================================
repro(  1. - ls_raw         : [array] list with all rasters
        2. - ls_cm          : [string] contains the string to the cloudmask of the actual landsatszene
        2. - e_ws           : [str] path to the workspace of the curent epoch e.g.: 'G:\\LANDSAT\\Landsat_Oesterreich\\p189_r26\\LE71890262000119SGS00'
        3. - epoch          : [str] name of the current epochfolder e.g: 'LE71890262000119SGS00'
        4. - b_cou          : [int] counter that points/opens the single bands of the copy of the landsat image in memory
        5. - ref_info_mo    : [array] contains the reference and extend information for the modis mosaic
        
        )

return  1. - out_ras        : [object] is the object for the in_memory reprojectetd, resampled landsat stack
        2. - driver         : [object] driver for GeoTiff
        3. - driver_mem     : [object] driver for in_memory
        4. - stack_name     : [str] retourns string wich names output stack for current epoch
        5. - bands          : [int] counts how many bands are in the reprojected landsat stack
'''




'''
===== Aggregate Satellite Scene to MODIS Resolution ========
============================================================
agg(    1. - out_path       : [string] path to actuial workingdirectory
        2. - out_ras        : [object] object with the pointer to the satellite/landsatscene which should be aggregated
        2. - out_qal        : [object] object with the pointer to the satellite/landsatscene's cloudmask which should be aggregated
        3. - ref_info_mo    : [array] contains the reference and extend information for the modis mosaic
        4. - bands          : [int] number of bands in the satellite stack
        5. - driver         : [object] driver to create the aggregated stack
        
        )

return  1. - stack_out      : [object] is the object for the aggregated satellite/landsatstack
        2. - DMPx_UL        : [array] contains on position DMPx_UL[1][:] the UL Koordinates of the Aggregated Landsat scene

'''
def FocalMean(x):
    return numpy.nanmean(x)

def agg(out_path,out_ras,out_qal,ref_info_mo,bands,driver,sat_raw,CaliDataType):
    from scipy.stats import nanmean
    
    # Aggregation of Satellite Data
    # =====================================
    ref_info_agg = {}
    # Calculate the Row and Column Index For UL and LR of the Landsat Image in the MODIS Mosaic
    # with this position its possible to extract the exact data for the rel calibration

    cols_rep = out_ras.RasterXSize
    rows_rep = out_ras.RasterYSize
    ref_info_agg={}

    ref_info_agg[0] = out_ras.GetProjection()
    ref_info_agg[1] = out_ras.GetGeoTransform()

    
    ls_x_col_ul = ref_info_agg[1][0]    # Landsat extend in UL in Koordinates
    ls_y_row_ul = ref_info_agg[1][3]

    index_mos_col_ul = ((abs(ref_info_mo[1][0]) + ls_x_col_ul)/ref_info_mo[1][1])  # Landsat extend in Row and Col Index of the MODIS mosaic
    index_mos_row_ul = ((abs(ref_info_mo[1][3]) - ls_y_row_ul)/ref_info_mo[1][1])

    ls_x_col_lr = ref_info_agg[1][0] + ref_info_agg[1][1]*cols_rep  # Landsat extend in UL in Koordinates
    ls_y_row_lr = ref_info_agg[1][3] - ref_info_agg[1][1]*rows_rep

    index_mos_col_lr = ((abs(ref_info_mo[1][0]) + ls_x_col_lr)/ref_info_mo[1][1])
    index_mos_row_lr = ((abs(ref_info_mo[1][3]) - ls_y_row_lr)/ref_info_mo[1][1])


    cols_agg = int(abs(index_mos_col_lr-index_mos_col_ul))
    rows_agg = int(abs(index_mos_row_ul-index_mos_row_lr))

    # Calculate the MODISPixel coordinates on which the Landsat UL Corner lies on
    # in this array the UL and LR Extend of the underlying MODISPixel is stored
    DMPx_UL = numpy.empty([2,2], dtype = float)         # [[UL_X,UL_Y] -  here its the Extend for the Modispixel in the Uper Left Landsatcorner and shows its pixelextend in the LR Pixelcorner
                                                        # [LR_X,LR_Y]] -  here its the Extend for the Modispixel in the Uper Left Landsatcorner and shows its pixelextend in the LR Pixelcorner
    
    

    
    
    # Calculate the Position of the MODIS Pixel on the UL Extend of under the Landsat Image
        # in UL PixelCorner
    DMPx_UL[0,0] = ref_info_mo[1][0] + ref_info_mo[1][1]*numpy.floor(index_mos_col_ul)
    DMPx_UL[0,1] = (abs(ref_info_mo[1][3] - ref_info_mo[1][1]*numpy.floor(index_mos_row_ul)))
        # in LR PixelCorner
    DMPx_UL[1,0] = DMPx_UL[0,0] + ref_info_mo[1][1]
    DMPx_UL[1,1] = DMPx_UL[0,1] - ref_info_mo[1][1]
                        

                                 
        
    




    

    
    # aggregate bands
    block = int(ref_info_mo[1][1]/ref_info_agg[1][1])
    fakt = ref_info_mo[1][1]/block
    
    #cols_nu = int(numpy.ceil(abs((DMPx_UL[1][0] - DMPx_UR[1][0])/ref_info_mo[1][1])))
    #rows_nu = int(numpy.ceil(abs((DMPx_UL[1][1] - DMPx_LL[0][1])/ref_info_mo[1][1])))
    cols_nu = int(numpy.ceil(numpy.float(cols_rep)/block))
    rows_nu = int(numpy.ceil(numpy.float(rows_rep)/block))

    
    offst_c = round(abs(ref_info_agg[1][0] - DMPx_UL[1][0])/ref_info_agg[1][1])
    offst_r = round(abs(ref_info_agg[1][3] - DMPx_UL[1][1])/ref_info_agg[1][1])

    print "Pixeloffset c:",offst_c
    print "Pixeloffset r:",offst_r
    
    # Aggregated Create OutputStack
    ras_agg = numpy.empty([rows_nu,cols_nu], dtype=float)
    qal_agg = numpy.empty([rows_nu,cols_nu], dtype=float)
    dataqal_agg = numpy.empty([rows_nu,cols_nu], dtype=float)
    
    qal_name = os.path.join(out_path,sat_raw[:-4]+".agg_qal"+CaliDataType+".tif")
    stack_name = os.path.join(out_path,sat_raw[:-4]+".agg"+CaliDataType+".tif")
    
    print "Stack NAME:",stack_name
    driver_mem = gdal.GetDriverByName("MEM")
    stack_out = driver_mem.Create("",cols_nu,rows_nu, bands, gdalconst.GDT_Float32)
    

    
    #gdal.SetConfigOption('HFA_USE_RRD', 'YES')
    #stack_out.BuildOverviews(overviewlist=[2,4,8,16,32,64,128,265,512,1024,2048,4096,8192])
    
    ref_info_nu={}
    ref_info_nu[0] = ref_info_mo[0]
    ref_info_nu[1] = [ref_info_agg[1][0]+offst_c*fakt, block*fakt, \
                      ref_info_mo[1][2], ref_info_agg[1][3]-offst_r*fakt, ref_info_mo[1][4], -block*fakt]
    stack_out.SetProjection(ref_info_nu[0])
    stack_out.SetGeoTransform(ref_info_nu[1])

    # create Outputraster for aggregated focalmean
    driver_tif = gdal.GetDriverByName("GTiff")
    stack_name_focal = os.path.join(out_path,sat_raw[:-4]+".agg.FocalMean"+CaliDataType+".tif")
    print "Stack Name Focal:",stack_name_focal
    stackFocal = createRas(driver_tif,stack_name_focal,cols_nu,rows_nu,bands,gdalconst.GDT_Float32,ref_info_nu)
    
    cou_ag= 0
    qal  = out_qal.ReadAsArray()                        # read out quality data
    qalND = out_qal.GetRasterBand(1).GetNoDataValue()

    
    print "\nStart Aggregating : \t", stack_name.split("\\")[-1],"\n------------------\n"
    #qal_data = out_qal.ReadAsArray()
    for b in range(0,bands,1):
        print "Processing band : \t\t", b+1
        print "Qaldata NoDataValue: \t", qalND
        
        bandAgg = out_ras.GetRasterBand(b+1)
        outBand = stack_out.GetRasterBand(b+1)

        data = bandAgg.ReadAsArray().astype(float)
        bandND = bandAgg.GetNoDataValue()
        print "SatData NoDataValue: \t", qalND

        

        
        print "rows_rep:\t\t", rows_rep
        print "cols_rep:\t\t",cols_rep
        

        
        
        
        cou_ag = 0
        ind_ag = 0
        
        if b == 0:
                    
            cou_ag = 0
            ind_ag = 0
            #for r in range(int(offst_r),rows_rep,block):
            for r in range(int(offst_r),rows_rep,block):
                
                ind_ag=0
                
                if r+block>=rows_rep:
                    numrows = abs(rows_rep-r)
                else:
                    numrows = block
                
                #for c in range(int(offst_c),cols_rep,block):
                for c in range(int(offst_c),cols_rep,block):
                    if c+block >=cols_rep:
                        numcols = abs(cols_rep-c)

                    else:
                        numcols = block
                    
                    #Ausmaskieren
                    data_block = data[r:r+numrows,c:c+numcols]
                    data_block = numpy.where(data_block==0,numpy.nan,data_block)
                    data_block = numpy.where(data_block==bandND,numpy.nan,data_block)

                    
                    
                    qal_block  = qal[r:r+numrows,c:c+numcols]

                    # Nodatacontamination
                    dataqal_block = numpy.where(qal_block == 0,numpy.nan,1)

                    data_block = numpy.where(qal_block==2,numpy.nan,data_block)
                    data_block = numpy.where(qal_block==qalND,numpy.nan,data_block)

                    # looking for outliers in the data_block
                    #data_blockV = robust_grubb(data_block.reshape(data_block.shape[0]*data_block.shape[1]))
                    #data_block = data_blockV.reshape(data_block.shape[0],data_block.shape[1])
                    
                    # cloud verseuchung
                    qal_block = numpy.where(qal_block==2,1,numpy.nan)
                    
                    
                    d_mean = numpy.nanmean(data_block)
                    d_mean_q = numpy.nansum(qal_block)/(qal_block.shape[1]*qal_block.shape[0])
                    d_mean_dq = numpy.nansum(dataqal_block)/(dataqal_block.shape[1]*dataqal_block.shape[0])
                    
                    
                    
                    if numpy.isnan(d_mean):
                        ras_agg[cou_ag,ind_ag] = 0
                        qal_agg[cou_ag,ind_ag] = 0          # matrix fuer cloudcontamination
                        dataqal_agg[cou_ag,ind_ag] = 0      # matrix fuer noDatacontamination
                    else:
                        ras_agg[cou_ag,ind_ag]   = d_mean
                        qal_agg[cou_ag,ind_ag] = d_mean_q
                        dataqal_agg[cou_ag,ind_ag] = d_mean_dq


##                    if ind_ag==cols_nu:
##                        break
                    ind_ag += 1
##                if cou_ag == rows_nu:
##                    break
                cou_ag += 1

        else:
        
            cou_ag = 0
            ind_ag = 0
            #for r in range(int(offst_r),rows_rep,block):
            for r in range(int(offst_r),rows_rep,block):
                ind_ag=0
                
                if r+block>=rows_rep:
                    numrows = abs(rows_rep-r)
                else:
                    numrows = block
                
                #for c in range(int(offst_c),cols_rep,block):
                for c in range(int(offst_c),cols_rep,block):
                    if c+block >=cols_rep:
                        numcols = abs(cols_rep-c)
                        
                    else:
                        numcols = block


                    #Ausmaskieren
                    data_block = data[r:r+numrows,c:c+numcols]
                    data_block = numpy.where(data_block==0,numpy.nan,data_block)
                    data_block = numpy.where(data_block==bandND,numpy.nan,data_block)

                    #data_blockV = robust_grubb(data_block.reshape(data_block.shape[0]*data_block.shape[1]))
                    #data_block = data_blockV.reshape(data_block.shape[0],data_block.shape[1])
                    
                    qal_block  = qal[r:r+numrows,c:c+numcols]


                    data_block = numpy.where(qal_block==2,numpy.nan,data_block)
                    data_block = numpy.where(qal_block==qalND,numpy.nan,data_block)
                    
                    
                    
                    d_mean = numpy.nanmean(data_block)
                    

                    
                    if numpy.isnan(d_mean):
                        ras_agg[cou_ag,ind_ag] = 0
                    
                    else:
                        ras_agg[cou_ag,ind_ag]   = d_mean
                    

##                    if ind_ag==cols_nu:
##                        print ind_aq
##                        break
                    ind_ag += 1
##                if cou_ag == rows_nu:
##                    print cou_ag
##                    break
                cou_ag += 1

        # Moving window Focal Mean
        print "Focal Mean for band"
        mask = numpy.ones([3,3])
        dataFocal= ndimage.filters.generic_filter(ras_agg,FocalMean,size=3,footprint=mask,cval=numpy.nan,mode='constant')
        data_agg = numpy.nan_to_num(dataFocal)
        writeOnBand(stackFocal,b+1,data_agg,32767)

        # Write Data onto stack in memory
        print "Write into stack\n..."
        data_agg = numpy.where(data_agg == 0,32767,ras_agg )
        outBand.SetNoDataValue(32767)
        outBand.WriteArray((data_agg))
        outBand.FlushCache()
        del outBand, bandAgg

    # Create Physical Aggregated Stack
    # --------------------------------
    print stack_name
    check = driver.CreateCopy(stack_name,stack_out,0,options=['COMPRESS=LZW'])
    print "Finished aggregating !"

    p_mat = numpy.where(qal_agg>=0.00001,0,numpy.absolute(numpy.nan_to_num(qal_agg)-numpy.nan_to_num(dataqal_agg)))   # Gewichtsmatrix die hinaus geschrieben wird

##    # Kontrolle Qalitaet
##    # ==================
##    qualityName = os.path.join(out_path,sat_raw[:-4]+".NodataContamination"+CaliDataType+".tif")
##    print "Creating out qality  \n path to:",qualityName
##    dataqal_ras = createRas(driver,qualityName,cols_nu,rows_nu,1,gdalconst.GDT_Float32,ref_info_nu)
##    writeOnBand(dataqal_ras,1,dataqal_agg,0)
##    del dataqal_ras
##    
##    noDataConName = os.path.join(out_path,sat_raw[:-4]+".Cloudcontamination"+CaliDataType+".tif")
##    print "Creating out nodataconatmination  \n path to:",noDataConName
##    dataqal_ras = createRas(driver, noDataConName,cols_nu,rows_nu,1,gdalconst.GDT_Float32,ref_info_nu)
##    dataqal_ras = createRas(driver, noDataConName,cols_nu,rows_nu,1,gdalconst.GDT_Float32,ref_info_nu)
##    writeOnBand(dataqal_ras,1,qal_agg,0)
##    del dataqal_ras,check,stackFocal

    return stack_out, DMPx_UL, p_mat




'''
===== Create Array with Date and Path Information = ========
============================================================
dataarray(    1. - sat_raw        : [string] file name of the to relcalibrate satellite image
              2. - sat_date       : [string] the date of the satellite image in 2000+doy
              3. - roots          : [string] the path of the actual satellite image
              4. - time_mod       : [array]  contains all year and DOY information of the MODIS TimeSeries [2000057,etc]
              5. - time_src       : [array]  contains all paths to alls MODIS epochs
              6. - win            : [int]    the windowsize for the rel calibration
        
        )

return  1. - src_data       : [array] contains all paths which are needed for the relative calibration sorted by date
        

'''

def dataarray(sat_raw,sat_date,roots,time_mod,time_src,win,CaliDataType):
        import array
        # Determine the Date befor and after the LandsatImage Date
        # ========================================================
        time_id  = range(0,len(time_mod),1)
        
        d_doy = abs(time_mod-int(sat_date))
        d_doy_tr = time_mod-int(sat_date)

        mod_id = numpy.where(d_doy==numpy.min(d_doy))   # index for the MODIS date where the LandsatDate is closes
        mod_id = mod_id[0][0]

        d_val_tr = d_doy_tr[mod_id][0]

        up_check = int(sat_date) - time_mod[0]           # checks if the LandsatDate is younger the the oldest MODISDate
                                                        # e.g: 2000066-2000057 - Landsat is younger then oldes MODIS epoch
        dw_check = int(sat_date) - time_mod[-1]                                                    

        if d_val_tr <0:                         # means the Landsatepoch is newer then the nearest MODIS epoch
            if up_check < win*8:                # if LS DOY is close to beginning of MODIS TS eg.: LS 2000066 -->  [2000057:2000065],LS_Scene,[2000073:2000121]
                mod_id = mod_id+1               
                up=time_id[0:mod_id]
                dw=time_id[mod_id:mod_id+win]
                pos = len(up)
                time_line = array.array('i',up)
                time_line.extend([int(sat_date)])
                time_line.extend(dw)
            else:                               # eg.: LS 2000114 --> [2000065:2000113],LS_Szene,[2000121:2000169]
                mod_id = mod_id+1
                up = time_id[mod_id-win:mod_id]     
                dw = time_id[mod_id:mod_id+win]
                pos = len(up)
                time_line = array.array('i',up)
                time_line.extend([int(sat_date)])
                time_line.extend(dw)
                
            if dw_check >0:
                up = time_src[-7:]
                dw = None
                pos = len(up)
                modis_data = "Not Availaible"
                time_line = array.array('i',up)
                time_line.extend([int(sat_date)])
                

                
        elif d_val_tr==0:                       # means the Landsatepoch is same DOY as MODIS Epoch 
            if up_check < win*8:                # if LS DOY is close to beginning of MODIS TS eg.: LS 2000089 -->  [2000057:2000081],LS_Scene,[2000089:2000137]
                up = time_id[0:mod_id]
                dw = time_id[mod_id:mod_id+win]
                pos = len(up)
                time_line = array.array('i',up)
                time_line.extend([int(sat_date)])
                time_line.extend(dw)
            else:                               # eg.: LS 2000129 --> [200073:2000121],LS_Scene,[2000129:2000177]
               up = time_id[mod_id-win:mod_id]
               dw = time_id[mod_id:mod_id+win]
               pos = len(up)
               time_line = array.array('i',up)
               time_line.extend([int(sat_date)])
               time_line.extend(dw)

            if dw_check >0:
                up = time_mod[-7:]
                dw = None
                pos = len(up)
                modis_data = "Not Availaible"
                time_line = array.array('i',up)
                time_line.extend([int(sat_date)])
                
            
        elif d_val_tr >0:                       # means the Landsatepoch is older then the nearest  MODIS Epoch                   
            if up_check < win*8:                # if LS DOY is close to beginning of MODIS TS eg.: LS 2000066 -->  [2000057:2000065],LS_Scene,[2000073:2000121]
                up = time_id[0:mod_id]         
                dw = time_id[mod_id:mod_id+win]
                pos = len(up)
                time_line = array.array('i',up)
                time_line.extend([int(sat_date)])
                time_line.extend(dw)
            else:                               # eg.: LS 2000120 --> [2000065:2000113],LS_Szene,[2000121:2000169]
                up = time_id[mod_id-win:mod_id]
                dw = time_id[mod_id:mod_id+win]
                pos = len(up)
                time_line = array.array('i',up)
                time_line.extend([int(sat_date)])
                time_line.extend(dw)

            if dw_check >0:
                up = time_mod[-7:]
                dw = None
                pos = len(up)
                modis_data = "Not Availaible"
                time_line = array.array('i',up)
                time_line.extend([int(sat_date)])





        src_data = {}
        for i in range(0,len(time_line),1):
            if i == pos:
                src_data[i] = [os.path.join(roots,sat_raw[:-4]+".agg.FocalMean"+CaliDataType+".tif")]     # landsatepoch
                #print [os.path.join(roots,sat_raw[:-4]+"_agg.tif")]
            else:
                src_data[i] = time_src[time_line[i]]
                #print time_src[time_line[i]]

        return src_data,pos
'''
===== linear Fit of the Satellite Data =====================
============================================================
fitl(   1. - lv        : [array] contains the modis spectral percentage
        2. - pv        : [array] weigtht of the observations
        3. - xv        : [array] first column of the Designmatrix
        4. - lsv       : [array] second column of the Designmatrix contains the differntial of the b1*x 
        )

return  1. - a0       : [array] first parameters of the linear regression
        2. - a1       : [array] second parameters of the linear regression
        
   
'''
def fitl_mat(lv,xv,lsv,pv):

##    lv  = numpy.reshape(lv,[lv.shape[0],1])
##    lsv = numpy.reshape(lsv,[lsv.shape[0],1])
##    pv  = numpy.reshape(pv,[pv.shape[0],1])
##    xv  = numpy.reshape(xv,[xv.shape[0],1])

    
    ax0 = xv
    ax1 = lsv

    a11 = numpy.nansum(ax0*pv*ax0,0)
    a12 = numpy.nansum(ax0*pv*ax1,0)
    a22 = numpy.nansum(ax1*pv*ax1,0)

    det = a11*a22 -  a12*a12

    ai11 = a22/det
    ai12 = -a12/det
    ai22 = a11/det

    vx0 = numpy.nansum(ax0*pv*lv,0)
    vx1 = numpy.nansum(ax1*pv*lv,0)

    a0 = ai11*vx0 + ai12*vx1
    a1 = ai12*vx0 + ai22*vx1

    # Kovarianzmatrix der Verbesserungen

    

    return a0,a1

def fitl_vektor(lv,av0,av1,pv):

    a11 = numpy.nansum(av0*pv*av0)
    a12 = numpy.nansum(av0*pv*av1)
    a22 = numpy.nansum(av1*pv*av1)

    det = a11*a22 -  a12*a12

    ai11 = a22/det
    ai12 = -a12/det
    ai22 = a11/det

    vx0 = numpy.nansum(av0*pv*lv)
    vx1 = numpy.nansum(av1*pv*lv)

    b0 = ai11*vx0 + ai12*vx1
    b1 = ai12*vx0 + ai22*vx1

    return b0,b1


# ==================================================================================
#                               OUTLIAR DETECTION SECTION 
# ==================================================================================


def robust_median(v,m=2.):
    print "Robust Method: \t\t\t Median-Distanz-Test"
    median = stats.nanmedian(v)
    dist_median = numpy.absolute(v - stats.nanmedian(v))    # absolute distanz der messdaten zum median
    ratio_dist_median = dist_median/median                       # verhaeltnis zwischen distanz und median, je groesser diese
                                                                # distanz umso wahrscheinlicher is der messwert ein aussreisser
    v=numpy.where(ratio_dist_median>=m,numpy.nan,v) if median else 0           # setzte alle aussreisser auf nan

    return v,ratio_dist_median



def robust_grubb(v):
    #print "Robust Method: \t\t\t Grubbs Test"

    #initialize parameters
    G=10000
    z_alpha = 100
    
    n = len(v)
    alpha = 0.05
    z_alpha = (n-1.)/numpy.sqrt(n) * ( numpy.sqrt(  stats.t.ppf(alpha/(2*n),n-2)**2 / (n-2+stats.t.ppf(alpha/(2*n),n-2)**2)))   # 
    while G>z_alpha:


        mue = numpy.nanmean(v)
        std = numpy.nanstd(v)
        

        dG = abs(v-mue)             # differenzen zu messwerten
        aktPG = numpy.nanmax(dG)    # aktueller maximaler wert
        G = aktPG/std               # Pruefgroesse
        
        if G> z_alpha:
            #print "Outlier gefunden - Robust Method: \t\t\t Grubbs Test"
            v[dG==aktPG]=numpy.nan      # setzten des ausreissers auf nan
            
        else:
            #print "kein aussreisser mehr gefunden"
            break
    

        
    
    return v
        
# Aggregate to no res
# ==========================

def agg_mod(data,ref_info_mo,aggFakt):



    cols_mo = data.shape[1]
    rows_mo = data.shape[0]
    res = ref_info_mo[1][1]
    cols_agg_mo = int(numpy.ceil((res*cols_mo)/(res*aggFakt)))
    rows_agg_mo = int(numpy.ceil((res*rows_mo)/(res*aggFakt)))

    moDataAgg = numpy.zeros([rows_agg_mo,cols_agg_mo],dtype=float)

    cou_c_mo = 0
    cou_r_mo = 0

    mask = numpy.ones([aggFakt,aggFakt])
    
    #data = numpy.nan_to_num(data).astype(numpy.uint16)      # funktioniert leider nur mit integer werten
    #data = mean(data,mask)

    def focalMean(x):
        return numpy.nanmean(x).astype(numpy.int)
    
    data = ndimage.filters.generic_filter(data,focalMean,size=3,footprint=mask,cval=numpy.nan,mode='constant')

    for r in range(0,rows_mo,aggFakt):
        cou_c_mo = 0
        if r + aggFakt>=rows_mo:
            rblock = abs(rows_mo - r)
        else:
            rblock = aggFakt

        for c in range(0,cols_mo,aggFakt):

            if c+aggFakt >= cols_mo:
                cblock = abs(cols_mo - c)
            else:
                cblock = aggFakt

            dBlock = data[r:r+rblock,c:c+cblock]
            dmean = numpy.nanmean(dBlock)

            if numpy.isnan(dmean):
                moDataAgg[cou_r_mo,cou_c_mo] = 0

            else:
                moDataAgg[cou_r_mo,cou_c_mo] = dmean

            cou_c_mo+=1
        cou_r_mo+=1
    
    moDataAgg[moDataAgg<=0.001]=numpy.nan
    return moDataAgg
    
    
    
    




    
    
    

    
    






















