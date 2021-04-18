# -*- coding: cp1252 -*-
"""
 Source Name:   FITATA_V4_Four2.py
 Version:       4.1
 Author:        Paul Arzberger
 Date:          27 November 2012
                
 #Usage:        Fits Satellite Images With Fourier Development
 Arguments:     mod_dir                     - the path to the input files
                myd_dir
                out_dir                      - the path to the output images
                output extension             - defines the output format / extension
                                               if not specified, arcinfo grid will be used
                theme_file                   - specifies the themes to be stacked
                                               to change a specified theme you have to change
                                               the index in row 33 "theme_file = list_mod[0].split(".")"
                                               list_mod[0] == EVI
                                               list_mod[1] == MIR
                                               list_mod[2] == NIR
                                               list_mod[3] == blue
                                               list_mod[4] == composite
                                               list_mod[5] == pixel_reliability
                                               list_mod[6] == red
"""


# ============ IMPORTS, Variabledeklaration ====================
import os, osgeo, sys, traceback, numpy, time, glob
import matplotlib.pyplot as plt
from numpy import dot, pi, cos, sin
from osgeo import gdal, gdalconst
from osgeo.gdal import *

start = time.clock()

# Directories 

mod_dir = r"E:\data_modis_tiff\h20v03\MOD13Q1"
myd_dir = r"E:\data_modis_tiff\h20v03\MYD13Q1"

out_dir = r"E:\OutDir\fitted\Fourier\h20v03"


# Variables
ref_info = {}

# =============== Create Input List=============================
# ==============================================================
ws = os.chdir(mod_dir)
list_mod = glob.glob('*.tif')
list_mod.sort()



theme_name = list_mod[2].split('.')[5]          # these are costumized INDIZES!!!! cause Data is not complete , EVI etc in folder is missing!!!
qal_name   = list_mod[5].split('.')[5]

tile = list_mod[2].split('.')[2]

ws = os.chdir(myd_dir)
list_myd = glob.glob('*.tif')
list_myd.sort()

list_data = numpy.empty(552,dtype='object')
list_qal  = numpy.empty(552,dtype='object')
ind = 0
cou = 0

for i in range(0,len(list_mod),1):
    s_tt = list_mod[i].find(theme_name)
    s_ta = list_myd[i].find(theme_name)
    s_qt = list_mod[i].find(qal_name)
    s_qa = list_myd[i].find(qal_name)

    if s_tt>0 and s_ta>0:
        list_data[cou] = os.path.join(mod_dir, list_mod[i])
        cou = cou + 1
        list_data[cou] = os.path.join(myd_dir, list_myd[i])
        cou = cou + 1
    elif s_qt>0 and s_qa>0:
        list_qal[ind] = os.path.join(mod_dir, list_mod[i])
        ind = ind + 1
        list_qal[ind] = os.path.join(myd_dir, list_myd[i])
        ind = ind + 1
    else:
        continue


# ================================== FIT SAT DATA ==============================
# ==============================================================================


block = 150
T=year = 46
u = 7     # number of fourier elements

epochs = range(0,46,1)
names_ras = numpy.empty([year,1], dtype='object')
names_qal = numpy.empty([year,1], dtype='object')
out_ras = {}
ras_data = numpy.empty([year,block,block], dtype=float)
qal_data = numpy.empty([year,block,block], dtype=float)
epochs_ras = range(0,len(list_data),year)
epochs_qal = range(0,len(list_qal),year)

A = numpy.ones([year,u*2+1],dtype=float)
At_m = numpy.ones([block*block,u*2+1,year], dtype=float)
A_m = numpy.ones([block*block,year,u*2+1], dtype=float)
row = numpy.empty([u,2], dtype=float)
A_mclear = numpy.ones([block*block,year,u*2+1], dtype=float)        # to set Designmatrix ready for next loop at end of loop 
At_mclear = numpy.ones([block*block,u*2+1,year], dtype=float)
ld = numpy.empty((block*block,year,1),dtype=float)

for t in range(1,year+1,1):
    for n in range(1,u+1,1):

        da = cos((2*pi/T)*(n)*t)
        db = sin((2*pi/T)*(n)*t)
        row[n-1,0]=da
        row[n-1,1]=db
        
    A[t-1,1:]=row.reshape(1,u*2)

# Get Projection and Bitdepth Info from one raster

info_ras = gdal.Open(list_data[0], GA_ReadOnly)
driver = info_ras.GetDriver()
ref_info[0] = info_ras.GetProjection()
ref_info[1] = info_ras.GetGeoTransform()


for ts in range(1,12,1):         # goes from 1 to 12
    
    print "Starting Fitting Process...\nYear Nr :",ts, "of", len(epochs_ras)
    print "Resolution :", 250, "m"
    names_ras = list_data[ts*year:(ts+1)*year]
    names_qal = list_qal[ts*year:(ts+1)*year]

    for c in range(0,year,1):
        # create output files where fitted data is written onto
        
        fit_dir = os.path.join(out_dir, names_ras[c].split("\\")[4][:-4]+".DFT.fitted.tif")

        or_temp = driver.Create(fit_dir, 4800,4800,1,gdal.GDT_Int16)
        or_temp.SetProjection(ref_info[0])
        or_temp.SetGeoTransform(ref_info[1])

        out_ras[c] = or_temp
    
    for rows in range(0,4800,block):
        print rows, " rows from 4800 of tile : ", tile, " have been processed" 
        for cols in range(0,4800,block):
            print cols
            for t in epochs:


                
                # open and read sat data in blocks
                ras = gdal.Open(names_ras[t], GA_ReadOnly)
                ras_data[t,:,:] = ras.ReadAsArray(cols,rows,block,block)
                
                qal = gdal.Open(names_qal[t], GA_ReadOnly)
                qal_data[t,:,:] = qal.ReadAsArray(cols,rows,block,block)
                

            # ==========  FITTING DISTURBED SAT DATA =======================
            # ==============================================================
            
            lv = ras_data.transpose().reshape(year*block*block,1)
            pv = qal_data.transpose().reshape(year*block*block,1)
            Av = numpy.kron(numpy.ones((block*block,1)),A)        # repeat matrix n times
                                                                  # dim Av e.g.: [5625,46,15]
                                       
            
            # weight sat data
            pv = numpy.where(pv==0,0.99,pv)
            pv = numpy.where(pv==1,0.5 ,pv)
            pv = numpy.where(pv==2,0.01, pv)
            pv = numpy.where(pv==3,0.01, pv)
            pv = numpy.where(pv==-1,0.0001,pv)
            

            print "Start ATPA..."
           
            # ==== multidim data preperation =====
            
            A_m = A_m*A
            
            At_m = At_m*A.transpose()

            pvm = pv.reshape(block*block,1,year)        # dim [5625,1,46]
            lvm = lv.reshape(block*block,year,1)        # 
            ATP  = At_m * pvm
            
            
            for i in range(0,len(ATP),1):
                ATPA = dot(ATP[i],A_m[i])

                if numpy.linalg.det(ATPA)!=0:
                    ATPA = numpy.linalg.inv(ATPA)
                    ATPL = dot(ATP[i],lvm[i])

                    xd = dot(ATPA,ATPL)
                    ld[i] = dot(A_m[i],xd)
                else:
                    if i < len(lvm)-2:
                        
                        ld[i] = (lvm[i-3]+lvm[i-2]+lvm[i-1]+lvm[i]+lvm[i+1]+lvm[i+2]+lvm[i+3])/7
                    elif i == len(lvm)-2:
                        ld[i] = (lvm[i-2]+lvm[i-1]+lvm[i]+lvm[i+1]+lvm[i+2])/5
                    elif i == len(lvm)-1:
                        ld[i] = (lvm[i-1]+lvm[i]+lvm[i+1])/3
                    elif i == len(lvm):
                        ld[i] = lvm[i]
                    print "Not inv! in Epcoh NR: ", i
                 
            # Rest Desgignmatrix!!!!!!!!!!!   
            A_m = A_mclear       
            At_m = At_mclear 
                
            fit_data = ld.transpose().reshape(year,block,block)

            # write fitted data into
            print "start writing into files"
            for i in range(0,year,1):
                fitBand_temp= out_ras[i]
                fitBand = fitBand_temp.GetRasterBand(1)
                fitBand.SetNoDataValue(32767)
                fitBand.WriteArray(fit_data[i].transpose(),cols,rows)
                fitBand.FlushCache()
 
del out_ras
print "Programm ENDE"
ende = time.clock()
print "Ellapsed time :", ende-start
