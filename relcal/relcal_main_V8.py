# -*- coding: cp1252 -*-
# Reproject/Resample and Aggregate absolut calibrated/raw Landsatdata
# --> choose bewteen hdf or tif
# Paul Arzberger
# 06.03.2013 - V1.1
# 11.04.2013 - V2.0 --> all steps are written out as tif to compare results
# 19.04.2013 - V2.1 --> only output is written as tif , rest of production remains in RAM
# 25.04.2013 - V2.2 --> outsourcing via functions
# 15.05.2013 - V3.0 --> implement os.walk instead of loops to walk throug directories
# 13.05.2013 - V3.1 --> implement dataarray+
# 06.08.2013 - V6.0 --> Results are looking good :)
# 07.08.2014 - V6.1 --> LC8 implemented, MODIS Band 5 no calibration cause no equvivalent bands in Landsat systems
# 14.08.2014 - V6.2 --> statistical values for regression quality implemented
# 28.08.2014 - V7.0 --> Plots verschoenert



import os, time, glob, numpy, relcal_func_v6
from osgeo import gdal
from osgeo.gdal import *
from osgeo.gdalconst import *
from matplotlib import pyplot as plt

from scipy import stats
from scipy import ndimage
from scipy import odr

start = time.clock()

# Paul Home Extern Platte
### Paul Home

modis_fit_type_info = dict()
modis_fit_type_info["dft_1"] = {"dir": "dftelements_3100_001_001_001", "short_desc": ".dft_1", "min_epoch_year": "2001", "min_epoch_doy": "001"}
modis_fit_type_info["dft_2"] = {"dir": "dftelements_3100_050_025_001", "short_desc": ".dft_2", "min_epoch_year": "2001", "min_epoch_doy": "001"}
modis_fit_type_info["fft"] = {"dir": "fft", "short_desc":".fft" , "min_epoch_year": "2001", "min_epoch_doy": "1"}
modis_fit_type_info["poly_1"] = {"dir": "poly_lin_win15weights1_001_001_001", "short_desc": ".poly_1", "min_epoch_year": "2000", "min_epoch_doy": "113"}
modis_fit_type_info["poly_2"] = {"dir": "poly_lin_win15weights1_05_025_001", "short_desc": ".poly_2", "min_epoch_year": "2000", "min_epoch_doy": "113"}

calibrate_that = "dft_1"

modis_fit_type = modis_fit_type_info[calibrate_that]["short_desc"]
modis_stack_dir = modis_fit_type_info[calibrate_that]["dir"]
modis_stack_year_doy = modis_fit_type_info[calibrate_that]["min_epoch_year"] + modis_fit_type_info[calibrate_that]["min_epoch_doy"]

listCaliDataType = [".relcal_500.V6",".abscal.OLS.V6",".numpy_relcal_500.V6",".scipy_relcal_500.V6", ".scipy_odr_500.V6"]
listCaliDataType = [calitype + modis_fit_type for calitype in listCaliDataType]


in_dir   = r"E:\LANDSAT_Data\S2BIOM\relcal_195026"
out_dir  = r"E:\LANDSAT_STACK"

mod_dir  = os.path.join(r"E:\MODIS_Data\v6\stacked\h18v04", modis_stack_dir)
demo_mcd = os.path.join(mod_dir,  r"MCD43A4.A2004001.h18v04.006." + modis_fit_type_info[calibrate_that]["dir"] + ".tif")
# Create List with Modis DOY


# create list with all modis images
os.chdir(os.path.join(mod_dir))
inhalt = sorted(glob.glob('*.tif'))     # contains all *.tifs from Landsat directory


# create list with all modis doy info   
time_mod = numpy.ones([len(inhalt),1],dtype=int)
time_src = numpy.ones([len(inhalt),1],dtype=object)

for i in range(0,len(time_mod),1):
    time_mod[i]=int(inhalt[i].split('.')[1][1:])      # contains [year+doy]
    time_src[i]= os.path.join(mod_dir, inhalt[i])                        # contains modis paths



# Extract Reference Information    
ref_info_mo = {}

# open ref Modis Ras, to get Projection 
ras_mos = gdal.Open(demo_mcd, GA_ReadOnly)
ref_info_mo[0] = ras_mos.GetProjection()
ref_info_mo[1] = ras_mos.GetGeoTransform()

# ==== Calibration Extension =====
# ==== CHANGE HERE !!!!
ext = '*.tif'
res = 30.88751              # manuelle berechnung der aufloesung fur landsat damit ein vielfaches von modis res
win = 1                     # anzahl der vorher/nachherszenen um die landsatszene
med = range(0,2*win+1,1)    # sitz der landsatszene in einer zeitreihe
block_agg = 15*3

# ================================

print("os.indir exists: ", os.path.exists(in_dir))
cou = 0

# walk through Landsatdirectories and search for content
for roots, dirs, files in os.walk(in_dir):
    
    print(roots)
    os.chdir(roots)
    inhalt  = glob.glob('L*.tif')

    if inhalt:
        in_path  = roots                        # path to Landsat directory
        sat_date = roots.split("\\")[-1][9:16]   # date of akquisitin of the raw landsat szene

        # V0 relcal/abscal ohne Ausreisser mit ausgleichsgerade
        # V1 relcal/abscal ohne ausreisser mit scipy lin regress
        # V2 relcal/abscal mit ausreisser mit scipy lin regress

        #Define "Raw"-Data

        CaliDataType = listCaliDataType[4]
        plots_dir = roots+"\\"+"plots_"+CaliDataType[1:]

        if os.path.isdir(plots_dir):
           print(plots_dir," exists\n")
        else:
           os.makedirs(plots_dir)
           
        # create file for statistics
        
        if CaliDataType == ".abscal.OLS.V6":
            ls_ms    = glob.glob('ls*.msp.tif')     # absolute calibrated LandsatSzene
            #sat_raw = ls_ms[0]
            if ls_ms:
                sat_raw = ls_ms[0]
            else:
                sat_raw = roots.split("\\")[-1]+'.tif'
        else:
            ls_ms    = glob.glob('ls*.msp.tif')
            sat_raw = ls_ms[0]
            
        ziel = open(os.path.join(plots_dir,sat_raw[:-9]+"_Stats"+CaliDataType+".txt"),"w")

        
        print "\n\n====================================================================================================="
        print "Start Processing with Satellite:\t", sat_raw
        
        
        mod_bands = [3,4,1,2,6,7]            # Modis Bands
        # Define bands for Satellite and MODIS
        if sat_raw[:3]=='LE7':
            sat_bands = [1,2,3,4,5,8]           # Landsatbands      -- 8 is the "7th" Band-MIR-Band
            sat_bands_beschriftung = [1,2,3,4,5,7]
            ls_cm    = glob.glob('ls*.msk.tif')     # Landsat Cloudmask from absolute calibration
        elif sat_raw[:3]=='LT5':
            sat_bands = [1,2,3,4,5,7]
            sat_bands_beschriftung = [1,2,3,4,5,7]
            ls_cm    = glob.glob('ls*.msk.tif')     # Landsat Cloudmask from absolute calibration
        elif sat_raw[:2]=='ls':
            sat_bands = [1,2,3,4,5,6]
            sat_bands_beschriftung = [1,2,3,4,5,7]
            ls_cm    = glob.glob('ls*.msk.tif')     # Landsat Cloudmask from absolute calibration
        elif sat_raw[:3]=='LC8':
            sat_bands = [2,3,4,5,6,7,9]
            mod_bands = [3,4,1,2,6,7,5]            # Modis Bands
            sat_bands_beschriftung = [1,2,3,4,5,7,9]
            ls_cm    = glob.glob('lc*.msk.tif')     # Landsat Cloudmask from absolute calibration
    
        
        print "Quality information at :\t\t", ls_cm[0]
        print "-------------------------------"
        
        # Check if Fitted timeseries has a useable starting point
        sat_data_year_doy = ls_cm[0].split(".")[2]
        sat_data_year = int(sat_data_year_doy[:4])
        sat_data_doy = int(sat_data_year_doy[4:])



        if int(sat_data_year_doy) < int(modis_stack_year_doy):
            print("Landsat date is older then oldes MODIS doy : ",int(sat_data_year_doy) < int(modis_stack_year_doy))
            print("Landsat date: ", sat_data_year_doy)
            print("MODIS   date: ", modis_stack_year_doy)
            continue
        else:
            print("Landsat date: ", sat_data_year_doy)
            print("MODIS   date: ", modis_stack_year_doy)
            # Reproject Satellite and Quality Data
            # ====================================

            # out_stack is an object to the reprojected Landsatstack
            # out_al is an object to the reprojected Qualitymask
            # driver_tif and driver_mem are driver to create rasters in memory or on harddrive
            # satBandsNu is an array with the actual Bandindizes of the reprojected Landsatstack
            #out_stack,out_qal,driver_tif,driver_mem = relcal_func_v3.reprostack(sat_raw,ls_cm,sat_bands,ref_info_mo)

            out_stack,out_qal,driver_tif,driver_mem = relcal_func_v6.reprostack(sat_raw, ls_cm, sat_bands, ref_info_mo, in_path, CaliDataType)

    ##        out_stack = gdal.Open(r"LE71950262004140ASN01.rep.relcal_testarea.tif", GA_ReadOnly)
    ##        out_qal = gdal.Open(r"ls7.195026.2004140.msk.rep.relcal_testarea.tif", GA_ReadOnly)
    ##        driver_mem = gdal.GetDriverByName("MEM")
    ##        driver_tif = out_stack.GetDriver()

            # Aggregate Satellite Data to MODIS Resolution
            # ============================================
            agg_stack, DMPx_UL,qal_data = relcal_func_v6.agg(in_path, out_stack, out_qal, ref_info_mo, len(sat_bands), driver_tif, sat_raw, CaliDataType)

            # Create Path Array with Needed SatData
            # =====================================

            # src_data is a dictionary with the modis_epochs surounding the Landsatscene , depending on windowsize
            # pos is the position of the landsat szene in the src_data array

            src_data,pos = relcal_func_v6.dataarray(sat_raw, sat_date, in_path, time_mod, time_src, win, CaliDataType)

            # relative calibration Satellite Data to MODIS pixelvalue
            # =======================================================

            print "Start relative calibration\n..."

            ref_info_agg={}
            ref_info_agg[0] = agg_stack.GetProjection()
            ref_info_agg[1] = agg_stack.GetGeoTransform()

            # calculate the Index Position of the Data under the aggregated Landsatszene
            # Berechnet wie viele Pixel vom UL Landsatcorener man weitergehen muss, damit man eine nahezu volle abdeckung fuer den
            # ersten block / ersten MODIS pixel hat
            dcols = int(round(abs((ref_info_agg[1][0]-ref_info_mo[1][0])/ref_info_agg[1][1])))
            drows = int(round(abs((ref_info_agg[1][3]-ref_info_mo[1][3])/ref_info_agg[1][1])))

            agg_cols = agg_stack.RasterXSize
            agg_rows = agg_stack.RasterYSize

            block = 64


            # temp create tif for aggregated quality info
    ##        agg_tmp = relcal_func_v6.createRas(driver_tif,sat_raw[:-4]+".gain"+CaliDataType+".tif",agg_cols,agg_rows,1,gdalconst.GDT_Float32,ref_info_agg)
    ##        relcal_func_v6.writeOnBand(agg_tmp,1,qal_data,0)
    ##        del agg_tmp


            # Load Sat-Data
            # =================================================================

            # Determine the closesd epoch to the landsat scene
            print("src data: ", src_data)
            if sat_raw[:2]=='ls':
                if int(src_data[1][0].split("\\")[-1].split(".")[2][:4])<2000:
                    break
                else:
                    pre_doy = int(src_data[0][0].split("\\")[-1].split(".")[1][1:])
                    post_doy = int(src_data[2][0].split("\\")[-1].split(".")[1][1:])
                    ls_doy = int(src_data[1][0].split("\\")[-1].split(".")[2])

                    dpre = abs(pre_doy - ls_doy)
                    dpost = abs(post_doy - ls_doy)

                    if dpre < dpost:
                       sat = 0
                    else:
                       sat = 2
            else:
                if int(sat_raw[9:13])<2000:
                    break
                else:
                    pre_doy = int(src_data[0][0].split("\\")[-1].split(".")[1][1:])
                    post_doy = int(src_data[2][0].split("\\")[-1].split(".")[1][1:])
                    ls_doy = int(src_data[1][0].split("\\")[-1].split("_")[0][9:16])

                    dpre = abs(pre_doy - ls_doy)
                    dpost = abs(post_doy - ls_doy)

                    if dpre < dpost:
                        sat = 0
                    else:
                        sat = 2





            sat_data = numpy.empty([agg_rows,agg_cols,2],dtype=float)

            sat_poi_ls = gdal.Open(src_data[1][0], GA_ReadOnly)
            sat_poi_mo = gdal.Open(src_data[sat][0], GA_ReadOnly)


            # Calibrate Bands
            # ==========================================================

            # Open original Landsat Szene
            ls_orig = gdal.Open(sat_raw, GA_ReadOnly)

            # Create Output Raster for aggregated Calibrated LandsatSzene
            # Create Output Raster for original resolution LandsatSzene
            ls_orig_refinfo = {}
            ls_orig_refinfo[0] = ls_orig.GetProjection()
            ls_orig_refinfo[1] = ls_orig.GetGeoTransform()

            # Raster for Relative Calibrated Satellite Szene in Original Resolution and Reference System
            if CaliDataType == ".abscal.OLS.V6":
                NameRelCalOrigLS = sat_raw[:-4]+".calibrated"+CaliDataType+".tif"
            else:
                NameRelCalOrigLS = sat_raw[:-4]+".calibrated"+CaliDataType+".tif"

            Ls_RelOrigRes = relcal_func_v6.createRas(driver_tif, NameRelCalOrigLS, ls_orig.RasterXSize, ls_orig.RasterYSize, len(sat_bands), gdalconst.GDT_Int16, ls_orig_refinfo)


            # Create Object for Visualisation
            # -------------------------------
            fig = plt.figure(figsize=(30, 45))
            fig2 = plt.figure(figsize=(30, 45))
            fig3 = plt.figure(figsize=(30, 45))
            figDif = plt.figure(figsize=(30, 45))


            for band in range(0,len(sat_bands),1):

                print "Calibrate LS_band nr: \t\t", sat_bands[band], " to MODIS_band nr: \t",mod_bands[band]

                print "load data from Landsat"
                # Landsat Stack Original
                satOrigBand = ls_orig.GetRasterBand(sat_bands[band])
                satDataOrigtmp = satOrigBand.ReadAsArray()
                origNoData = satOrigBand.GetNoDataValue()
                satDataOrig = numpy.where(satDataOrigtmp==origNoData,numpy.nan, satDataOrigtmp)
                del satDataOrigtmp

                # Landsat Stack Aggregated
                sat_band = sat_poi_ls.GetRasterBand(band+1)
                sat_datatmp = sat_band.ReadAsArray()
                lsNoData = sat_band.GetNoDataValue()
                sat_data[:,:,1] = numpy.where(sat_datatmp==lsNoData,numpy.nan,sat_datatmp)
                sat_data[:,:,1] = numpy.where(sat_datatmp==32767,numpy.nan,sat_datatmp)
                del sat_band, sat_datatmp

                print "Load data from MODIS"
                # MODIS Szene
                sat_band = sat_poi_mo.GetRasterBand(mod_bands[band])
                modNoData = sat_band.GetNoDataValue()
                #sat_data[:,:,0] = sat_band.ReadAsArray(dcols,drows+1,agg_cols,agg_rows)
                sat_data[:,:,0] = sat_band.ReadAsArray(dcols,drows,agg_cols,agg_rows)
                sat_data[:,:,0] = numpy.where(sat_data[:,:,0]==modNoData,numpy.nan, sat_data[:,:,0])
                del sat_band

                # Calculate Focal Mean on MODIS SAT DATA
                # ======================================

                mask = numpy.ones([3,3])
                moDataFoc  = ndimage.filters.generic_filter(sat_data[:,:,0], relcal_func_v6.FocalMean, size=3, footprint=mask, cval=numpy.nan, mode='constant') # Calculates a Focal Mean over the MODIS coutout
                sat_data[:,:,1][qal_data<1]=numpy.nan
                qal_data[qal_data<1]=0

                satBandsCostum = [1,2,3,4,5,6]      # Landsat Agg Bands
                satDataAgg = sat_data[:,:,1]        # aggregated focalmeaned satdata
                qalDataAgg = ndimage.filters.generic_filter(qal_data, relcal_func_v6.FocalMean, size=3, footprint=mask, cval=numpy.nan, mode='constant')         # calculates Focal Mean Over Qualitydata
                ref500mo = [DMPx_UL[1][0],ref_info_agg[1][1],ref_info_agg[1][2],DMPx_UL[1][1], ref_info_agg[1][4], ref_info_agg[1][5]]

                if band == 0:

                    # Create Focal Quality Raster
                    lsQualiFoc = relcal_func_v6.createRas(driver_tif, sat_raw[:-4] + ".gain.FocalMean" + CaliDataType + ".tif", agg_cols, agg_rows, 1, gdalconst.GDT_Float32, ref_info_agg)
                    relcal_func_v6.writeOnBand(lsQualiFoc, 1, qalDataAgg, 0)



                    #m raster for focal mean agg calibrated landsat
                    ls_500_cal = driver_tif.Create(sat_raw[:-4]+".agg.calibrated"+CaliDataType+".tif",int(satDataAgg.shape[1]),int(satDataAgg.shape[0]),len(sat_bands),gdalconst.GDT_Float32, options=['COMPRESS=LZW'])
                    ls_500_cal.SetProjection(ref_info_agg[0])
                    ls_500_cal.SetGeoTransform(ref_info_agg[1])
                    #ls_500_cal.SetGeoTransform(ref500mo)

                    # MODIS Extent for processed Landsatscene
                    mo_500 = driver_tif.Create("MODIS_EPOCH"+".FocalMean"+CaliDataType+".tif",int(moDataFoc.shape[1]),int(moDataFoc.shape[0]),len(sat_bands),gdalconst.GDT_Float32, options=['COMPRESS=LZW'])
                    mo_500.SetProjection(ref_info_mo[0])
                    mo_500.SetGeoTransform(ref500mo)

                    modBandsCostum = [3,4,1,2,5,6]      # da ja nur 6 baender calibriert werden und auch der modisstack mo_500 nur 6 baender hat wird von [3,4,1,2,6,7] umgeschrieben damit die richtigen baender gezogen werden [3,4,1,2,5,6]
                mo_500B = mo_500.GetRasterBand(modBandsCostum[band])

                mo_500B.SetNoDataValue(modNoData)
                modtmpFoc = numpy.nan_to_num(moDataFoc)
                modtmpFoc[modtmpFoc==0]=modNoData
                mo_500B.WriteArray(modtmpFoc)
                mo_500B.FlushCache()
                del modtmpFoc

                print "First MODIS V6 Pixel UL: \t",sat_data[0,0,0]

                # Ausgleich
                # =================================================================

                cou_r = 0
                cou_c = 0

                xMat = numpy.ones([agg_rows,agg_cols,win*2+1], dtype=float)
                recalc = numpy.empty([agg_rows,agg_cols],dtype = float)


                # hier einsetzen - Depot!

                # reshape matrizzes to vectors
                lv = moDataFoc.reshape(moDataFoc.shape[0]*moDataFoc.shape[1])               # Modisdaten - beobachtungsvektor
                av0 = numpy.ones(moDataFoc.shape[0]*moDataFoc.shape[1])                     # erste Spalte A-Matrix
                av1 = satDataAgg.reshape(satDataAgg.shape[0]*satDataAgg.shape[1])           # zweite Spalte A-Matrix (Landsatdaten)
                pv = qalDataAgg.reshape(qalDataAgg.shape[0]*qalDataAgg.shape[1])            # gewichtsvektor

                av1[av1<=0]=numpy.nan
                lv[lv<=0]=numpy.nan
                pv[pv<=0]=0
                pv[pv<1]=0




                # mask_out modisdata with quality file
                # Updaten der gewichtsmatrix
                lv=numpy.where(numpy.isnan(av1),numpy.nan,lv)
                vec = numpy.nan_to_num(av1+1000)
                pv = numpy.where(vec==0,0,pv)

                # terminate NODATA Elements in arrays

                pv = pv[numpy.isnan(av1)==False]
                av0 = av0[numpy.isnan(av1)==False]
                lv = lv[numpy.isnan(av1)==False]
                av1 = av1[numpy.isnan(av1)==False]

                # terminate nodata elements in arrays,
                pv = pv[numpy.isnan(lv)==False]
                av0 = av0[numpy.isnan(lv)==False]
                av1 = av1[numpy.isnan(lv)==False]
                lv = lv[numpy.isnan(lv)==False]





                # Outliar-Detection
                av1 = relcal_func_v6.robust_grubb(av1)  # returns data with nodatavalues

                # terminate no data values
                pv = pv[numpy.isnan(av1)==False]
                av0 = av0[numpy.isnan(av1)==False]
                lv = lv[numpy.isnan(av1)==False]
                av1 = av1[numpy.isnan(av1)==False]


                # Ausgleich
                # =========================



                #[b0,b1] = relcal_func_v6.fitl_vektor(lv,av0,av1,pv)         # ausgleich nach parametern OLS

                # Augleich mit Numpy
                # ------------------

    ##            params = numpy.polyfit(av1,lv,1,cov=True)
    ##            b0 = params[1]
    ##            b1 = params[0]

                # Ausgleich mit Scipy Linregress
                # ------------------------------
                print "Scipy OLS Ausgleich\n..."
                b1,b0,r_val,tmp,p_val, = stats.linregress(av1,lv)

                # Ausgleich mit ODR - orthogonal Distanz regression
                # -------------------------------------------------
                print "Scipy ODR Ausgleich\n..."
                linearOdr = odr.Model(relcal_func_v6.linOdrFit)
                linearData = odr.Data(av1,lv)
                fitODR = odr.ODR(linearData,linearOdr,beta0=[b0,b1])
                paramsODR = fitODR.run()
                [b0,b1]=paramsODR.beta
                cov_mat = paramsODR.cov_beta




                satDataOrig[satDataOrig==0]=numpy.nan
                fitAgg = numpy.round(b0+b1*av1)                             # aggregated data !! has size of av1!!!
                fitSatDataOrig = numpy.round(b0+b1*satDataOrig)             # erneute berechnung original data
                fitSatDataOrig = numpy.nan_to_num(fitSatDataOrig)           # nodata values auf null setzten
                fitSatDataOrig[fitSatDataOrig<0]=0
                fitSatDataOrig[fitSatDataOrig>=10000]=10000
                #recalc = fitSatDataOrig.reshape(satDataOrig.shape[0],satDataOrig.shape[1])  # calibrierte landsatdaten (original) in matrix form bringen

                print "Writing Stuff Out\n..."
                # Landsat Szene Original-Res -  UTM
                BandLs_RelOrigRes = Ls_RelOrigRes.GetRasterBand(band+1)
                BandLs_RelOrigRes.SetNoDataValue(0)
                BandLs_RelOrigRes.WriteArray(fitSatDataOrig)
                BandLs_RelOrigRes.FlushCache()


                # Landsatszebe Aggregaded 500 Sin
                ls_500_calB = ls_500_cal.GetRasterBand(band+1)
                ls_500_calB.SetNoDataValue(0)
                fitSatDataAgg=b0+b1*satDataAgg
                fitSatDataAgg[fitSatDataAgg<0]=0
                fitSatDataAgg=numpy.round(fitSatDataAgg).astype(numpy.int16)
                ls_500_calB.WriteArray(fitSatDataAgg)
                ls_500_calB.FlushCache()


                # Recalculate Aggregated LandsatSzene
                fit_agg = fitSatDataAgg
                del fitSatDataAgg


                #Create Difference Rasters For actual MODIS band and aggregated calibrated LandsatBand
                diffRasName = os.path.join(plots_dir,"Diff.V6.LS_Band_"+str(sat_bands[band])+"."+sat_raw[:-4]+".tif")
                diffRas = relcal_func_v6.createRas(driver_tif, diffRasName, agg_cols, agg_rows, 1, gdalconst.GDT_Float32, ref_info_agg)
                diffRasData = moDataFoc-fit_agg
                relcal_func_v6.writeOnBand(diffRas, 1, diffRasData, -9999)

                spd = figDif.add_subplot(3,2,band+1)
                difim = spd.imshow(diffRasData)
                titleDif = "Differenzen MOD V6 Band "+str(mod_bands[band])+" | LS Band "+str(sat_bands_beschriftung[band])
                spd.set_title(titleDif)
                plt.suptitle(" - "+sat_raw[:-9]+" -\n- Differenzraster MODIS V6 zu Landsat - ",fontsize=20)
                plt.close()


                fitv_agg = fit_agg.reshape(fit_agg.shape[0]*fit_agg.shape[1])
                #fitv_agg = fitv_agg[numpy.isnan(fitv_agg)==False]

                # Qualitaetsmasse
                # ============================================================

                # Korrealtionskoeffizient
                meanLv = numpy.nanmean(lv)
                meanAv = numpy.nanmean(av1)
                corrCoef = (((av1-meanAv)*(lv-meanLv)).sum())/numpy.sqrt(((av1-meanAv)**2).sum()*((lv-meanLv)**2).sum())

                # R2 Bestimmtheitsmass
                R2= corrCoef**2


                # absolute Fehler
                absFehl = numpy.absolute(fitAgg-lv)             # laut bartsch,s509
                # mittlere Absolute Fehler
                meanAbsFehl=numpy.nanmean(absFehl)

                # relative Fehler
                relFehl = numpy.absolute(absFehl)/lv
                # mittlerer Relativer Fehler
                meanRelFehl = numpy.nanmean(relFehl)

                #Variationskoeffizient
                v_fit= numpy.std(fitAgg)/numpy.mean(fitAgg)
                v_lv = numpy.std(lv)/numpy.mean(lv)


                # Matplotlib - Creating Plot for relative calibrated band, compare to MODIS Band and write them into pdf File
                # ===========================================================================================================


                # caluculate regression lines
                fit_fn = numpy.poly1d(numpy.array([b1,b0]))
                xv = numpy.arange(0,numpy.nanmax(av1),1)
                ziel.write("\n\n------------------------------------------------\n")
                ziel.write("\t"+ CaliDataType + "\n - "+ "MOD V6 Band "+str(mod_bands[band])+" | LS Band "+str(sat_bands_beschriftung[band]) + "\n b0: %5.4f "%b0+"|  b1: %5.4f"%b1+"\n")
                if CaliDataType=='.scipy_odr_500':
                    # Residuen
                    v = paramsODR.delta

                    # durchschnittl quadrierte schaetzfehler --> entspricht der schaetzfehlerVarianz
                    resQuadMean = (v**2).sum()/v.shape

                    resVar = paramsODR.res_var


                    n=int(lv.shape[0])
                    erklVar  = paramsODR.sum_square_delta        # Varianz der schaetzung    RSS    skript uni duisburg
                    errorVar = paramsODR.sum_square_eps          # Varianz der Residuen      ESS
                    gesamtVar = paramsODR.sum_square     # varianz der beobachtungen TSS

                    ziel.write("Korrelationskoeffizient:\t"+str(corrCoef)+"\n")
                    ziel.write("Korrelationskoeffizient^2\t"+str(R2)+"\n")
                    ziel.write("durchsch. quad. Fehler :\t"+str(resQuadMean[0])+"\n")
                    ziel.write("Varianz der Residuen :\t\t"+str(resVar)+"\n")
                    ziel.write("Std der Residuen : \t\t"+str(numpy.sqrt(resVar))+"\n")
                    ziel.write("Erwartungswert Residuen :\t"+str(v.mean())+"\n")
                    ziel.write("mittlerer absolute Fehler :\t"+str(meanAbsFehl)+"\n")
                    #ziel.write("mittlerer relative Fehler :\t"+str(meanRelFehl)+"\n")
                    #ziel.write("mittlerer relFehler ODR:\t"+str(paramsODR.rel_error)+"\n")
                    ziel.write("Variationskoeffizient fit : \t"+str(v_fit)+"\n")
                    ziel.write("Variationskoeffizient lv :\t"+str(v_lv)+"\n")
                    #ziel.write("Erwartungswert Beobachtungen:\t"+str(lv.mean())+"\n")
                    #ziel.write("Erwartungswert Schaetzungen:\t"+str(fitAgg.mean())+"\n")
                    ziel.write("STD der Beobachtungen :\t\t"+str(lv.std())+"\n")
                    ziel.write("erklaerende Varianz : \t\t"+str(erklVar)+"\n")
                    ziel.write("nichterklaerende Varianz:\t"+str(errorVar)+"\n")
                    ziel.write("gesamt Varianz :\t\t"+str(gesamtVar)+"\n")

                else:
                    # Residuen
                    v = fitAgg-lv

                    # durchschnittl quadrierte schaetzfehler --> entspricht der schaetzfehlerVarianz
                    resQuadMean = (v**2).sum()/v.shape
                    resVar = (v**2).sum()/(v.shape[0]-2)


                    n=int(lv.shape[0])
                    erklVar  = numpy.sum((fitAgg-numpy.mean(lv))**2)/n  # Varianz der schaetzung    RSS    skript uni duisburg
                    errorVar = numpy.sum((lv-fitAgg)**2)/n              # Varianz der Residuen      ESS
                    gesamtVar = numpy.sum((lv-numpy.mean(lv))**2)/n     # varianz der beobachtungen TSS
                    ziel.write("Korrelationskoeffizient:\t"+str(corrCoef)+"\n")
                    ziel.write("Korrelationskoeffizient^2\t"+str(R2)+"\n")
                    ziel.write("durchsch. quad. Fehler :\t"+str(resQuadMean[0])+"\n")
                    ziel.write("Varianz der Residuen :\t\t"+str(resVar)+"\n")
                    ziel.write("Std der Residuen : \t\t"+str(numpy.sqrt(resVar))+"\n")
                    ziel.write("Erwartungswert Residuen :\t"+str(v.mean())+"\n")
                    ziel.write("mittlerer absolute Fehler :\t"+str(meanAbsFehl)+"\n")
                    #ziel.write("mittlerer relative Fehler :\t"+str(meanRelFehl)+"\n")
                    ziel.write("Variationskoeffizient fit : \t"+str(v_fit)+"\n")
                    ziel.write("Variationskoeffizient lv :\t"+str(v_lv)+"\n")
                    #ziel.write("Erwartungswert Beobachtungen:\t"+str(lv.mean())+"\n")
                    #ziel.write("Erwartungswert Schaetzungen:\t"+str(fitAgg.mean())+"\n")
                    ziel.write("STD der Beobachtungen :\t\t"+str(lv.std())+"\n")
                    ziel.write("erklaerende Varianz : \t\t"+str(erklVar)+"\n")
                    ziel.write("nichterklaerende Varianz:\t"+str(errorVar)+"\n")
                    ziel.write("gesamt Varianz :\t\t"+str(gesamtVar)+"\n")

                # Create Plots for calibrated Data
                sp1 = fig.add_subplot(3,2,band+1)        # add subplot to defined figure
                pw1 = sp1.plot(av1,lv,'.k',markersize=0.1,label='Punktwolke Daten')
                rg1 = sp1.plot(xv,fit_fn(xv),'--r',markersize=0.1,label='Regressionsgerade')
                titleNu ="    "+ CaliDataType + "\n - "+ "MOD V6 Band "+str(mod_bands[band])+" | LS Band "+str(sat_bands_beschriftung[band]) + "\n b0: %5.4f "%b0+"|  b1: %5.4f"%b1
                sp1.set_title(titleNu)
                plt.suptitle(" - "+sat_raw[:-9]+" - \nRegression",fontsize=20)
                if CaliDataType==".abscal.OLS.V6":
                    sp1.set_xlabel('Landsat [Reflectance]')

                else:
                    sp1.set_xlabel('Landsat [DN]')

                sp1.set_ylabel('MODIS [Reflectance]')
                sp1.legend(bbox_to_anchor=(0.30, 1), loc=1)
                plt.close()

                # Create plot for Residuals over MODIS
                erw_fn = numpy.poly1d(numpy.array([0,v.mean()]))
                ev = numpy.arange(0,int(lv.max()),1)

                sp2 = fig2.add_subplot(3,2,band+1)
                pw2 = sp2.plot(lv,v,'.k',markersize=0.25,label='Punktwolke Residuen')
                rg2 = sp2.plot(ev,erw_fn(ev),'--r',markersize=0.25,label='E(X) Residuen')
                title2 = "    "+CaliDataType+"\n   Residuen ueber MODIS V6 Band" + str(mod_bands[band]) + "\nErwartungswert Residuen : %5.4f"%v.mean()
                sp2.set_title(title2)
                plt.suptitle(" - "+sat_raw[:-9]+" - \n - Residuen ueber MODIS V6 - ",fontsize=20)
                sp2.set_xlabel('MODIS Reflectance [%]')
                sp2.set_ylabel('Residuum [%]')
                sp2.legend(bbox_to_anchor=(0.30, 1), loc=1)
                plt.close()

                # Create plot for Residuals over Predictions
                sp3 = fig3.add_subplot(3,2,band+1)
                pw3 = sp3.plot(fitAgg,v,'.k',markersize=0.25,label='Punktwolke Residuen')
                ew3 = sp3.plot(ev,erw_fn(ev),'--r',markersize=0.25,label='E(X) Residuen')
                title3 = "    "+CaliDataType+"\n   Schaetzung ueber MODIS V6 Band" + str(mod_bands[band]) + "\nErwartungswert Residuen : %5.4f"%v.mean()
                sp3.set_title(title3)
                plt.suptitle(" - "+sat_raw[:-9]+" - \n - Residuen ueber Schaetzungen - ",fontsize=20)
                sp3.set_xlabel('Schaetzung-Reflexion [%]')
                sp3.set_ylabel('Residuum [%]')
                sp3.legend(bbox_to_anchor=(0.30, 1), loc=1)
                plt.close()


            figDif.subplots_adjust(right=0.8)
            cbar_ax = figDif.add_axes([0.85, 0.15, 0.05, 0.7])
            figDif.colorbar(difim, cbar_ax)


            figName  = sat_raw[:-9]+" Kalibrierung Landsat zu MODIS V6 "    + CaliDataType+".png"
            fig.savefig(os.path.join(plots_dir,figName),dpi=300)

            figName2 = sat_raw[:-9] + " Residuen ueber MODIS V6 " +CaliDataType[1:]+".png"
            fig2.savefig(os.path.join(plots_dir,figName2),dpi=300)


            figName2 = sat_raw[:-9]+" Residuen ueber Schaetzwerte "+CaliDataType[1:]+".png"
            fig3.savefig(os.path.join(plots_dir,figName2),dpi=300)

            figNameDif = sat_raw[:-9]+" Differenz MODIS V6 zu agg Landsat "+CaliDataType[1:]+".png"
            figDif.savefig(os.path.join(plots_dir,figNameDif),dpi=300)




            del lv, av1, av0, pv,
            del BandLs_RelOrigRes,Ls_RelOrigRes,satOrigBand,ls_orig,sat_poi_ls,mo_500,mo_500B,ls_500_calB,ls_500_cal,diffRas,agg_stack,satDataOrig
        
       
    else:
        print "Kein Inhalt"
        continue
    
    if ziel:
        ziel.close()
    else:
        print "Kein Statsfile angelegt"
    
ende = time.clock()
print "Programm ENDE"
print "Elapsed time : \t", (ende-start)/60, "[min]"
