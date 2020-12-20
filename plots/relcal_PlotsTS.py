import osgeo , os, glob, matplotlib, numpy
from matplotlib import pyplot as plt
from osgeo import gdal, ogr


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
        MskName = 'ls7.'+pr+'.'+year+doy+'.msk.tif'
    elif sat == 'LT5':
        satAbsName = 'ls5.'+pr+'.'+year+doy+'.msp'
        PifName = 'ls5_'+pr+'_'+year+doy+'_msp_NoComp_PIF.tif'
        RbaName = 'ls5_'+pr+'_'+year+doy+'_msp_rc.tif'
        MskName = 'ls7.'+pr+'.'+year+doy+'.msk.tif'
        
    AbsRas = satAbsName+'.tif'
    RelRas = satAbsName+srchPattern[1:]
    

    

    # find Pifs
    

    return os.path.join(roots,AbsRas),os.path.join(roots,RelRas),os.path.join(roots,PifName),os.path.join(roots,RbaName),os.path.join(roots,MskName),int(year),int(doy)



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
                print "Shp y Coordinate not in SatSzene Extent"
                rasData=0
                
        else:
            print "Shp x Coordinate not in SatSzene Extent"
            rasData=0
    return rasData
    


inWalkDir = [r"L:\LANDSAT_STACK\S2BIOM",r"L:\test\relcal",r"L:\Diplom_Daten\test\relcal",r"L:\LANDSAT_STACK\S2BIOM"]
WalkDir = inWalkDir[3]
shpPOI = r"L:\LANDSAT_STACK\S2BIOM\shp\POIProfileUTM.shp"
shpDirLine = r"L:\LANDSAT_STACK\S2BIOM\shp\profileUTM.shp"
plotsDir = r"L:\LANDSAT_STACK\TSPlots\Points"
wrkDir = r"L:\LANDSAT_STACK\S2BIOM\wrkDir"


##shpDirLine = r"G:\LANDSAT_STACK\S2BIOM\shp\profile.shp"
##shpPOI = r"G:\LANDSAT_STACK\S2BIOM\shp\POI.shp"
##rasDir = r"G:\Diplom_Daten\test\relcal\p195_027\LE71950272005174EDC01\ls7_195027_2005174_msp.tif"
##wrkDir = r"G:\LANDSAT_STACK\S2BIOM\wrkDir"
##inWalkDir = [r"G:\LANDSAT_STACK\S2BIOM",r"G:\test\relcal",r"G:\Diplom_Daten\test\relcal",r"G:\LANDSAT_STACK\S2BIOM"]
##plotsDir = r"G:\LANDSAT_STACK\TSPlots"
##WalkDir = inWalkDir[-1]

# UserParameters
# ==============
UserYear = 2007
DOYB = 152       # DOY Beginn
DOYE = 265      # DOY End
srchPattern = '*.calibrated.abscal.OLS.tif'
shpPoiKoords = []
absEpochs = []
relEpochs = []
pifEpochs = []
rbaEpochs = []
mskEpochs = []
VisBand = 3
# ==============

shpDrv = ogr.GetDriverByName('ESRI Shapefile')
dsPoi = shpDrv.Open(shpPOI)
lyr = dsPoi.GetLayer()
numPoi = lyr.GetFeatureCount()



if os.path.exists(os.path.join(plotsDir,str(UserYear))):
    print "plotsdir exists"
else:
    print "Creating plotsdir subfolder : ", os.path.join(plotsDir,str(UserYear))
    os.makedirs(os.path.join(plotsDir,str(UserYear)))
    
    plotsDir = os.path.join(plotsDir,str(UserYear))
    


#iterate through Features
for ft in range(0,numPoi,1):
    actPoi = lyr.GetFeature(ft)
    shpPoiGeom = actPoi.GetGeometryRef()
    xPoi = shpPoiGeom.GetX()
    yPoi = shpPoiGeom.GetY()

    shpPoiKoords.append([xPoi,yPoi])


    
    

for roots, dirs, files in os.walk(WalkDir):
    
    os.chdir(roots)
    srchCalFile = glob.glob(srchPattern)
    
    if srchCalFile:
        AbsRas,RelRas,PifRas,RbaRas,MskRas,yearEp,doyEp = GetRasterNames(srchPattern,roots)
        
        if yearEp == UserYear:
            if doyEp>=DOYB and doyEp<=DOYE:
                #print "SatEpoche innerhalb Userinput:\t",yearEp,doyEp
                absEpochs.append(AbsRas)
                relEpochs.append(RelRas)
                pifEpochs.append(PifRas)
                rbaEpochs.append(RbaRas)
                mskEpochs.append(MskRas)
                 
            else:
                continue            
        else:
            continue


for ep in range(0,len(absEpochs),1):
    if os.path.exists(pifEpochs[ep]):
        print "Exists:\t", pifEpochs[ep]
    else: pifEpochs[ep] = None

    if os.path.exists(rbaEpochs[ep]):
        print "Exists:\t", rbaEpochs[ep]
    else: rbaEpochs[ep] = None

    if os.path.exists(absEpochs[ep]):
        print "Exists:\t", absEpochs[ep]
    else: absEpochs[ep] = None

    if os.path.exists(relEpochs[ep]):
        print "Exists:\t", relEpochs[ep]
    else: relEpochs[ep] = None

    if os.path.exists(mskEpochs[ep]):
        print "Exists:\t", mskEpochs[ep]
    else:
        mskEpochs[ep]=None
        

# ReadOut Data on The Shapefile Coordinates

TS = []

satDataSortedEpochs = numpy.empty([len(absEpochs),1,5],dtype=object)
satDataSortedEpochs[:,0,1]=absEpochs
satDataSortedEpochs[:,0,2]=relEpochs
satDataSortedEpochs[:,0,3]=pifEpochs
satDataSortedEpochs[:,0,4]=rbaEpochs




for t in range(0,satDataSortedEpochs.shape[0],1):
    satDataSortedEpochs[t,0,0]=satDataSortedEpochs[t,0,1].split("\\")[-1].split(".")[2]

sortedEps = numpy.unique(numpy.sort(satDataSortedEpochs,0)[:,0,0])
#satDataBase = numpy.zeros([len(absEpochs),4,len(shpPoiKoords)],dtype=int)
satDataBase = numpy.zeros([len(absEpochs),5,len(shpPoiKoords)],dtype=int)

# Create Figure Objects
# ---------------------
fig = plt.figure(figsize=(45, 300))




for shp in range(0,len(shpPoiKoords),1):
    print "Proccessing POI Nr :",shp+1," Koords : ",shpPoiKoords[shp]
    shp_x = shpPoiKoords[shp][0]
    shp_y = shpPoiKoords[shp][1]

    cou_ep = 0
    for e in range(0,len(sortedEps),1):
        dataProcess = satDataSortedEpochs[satDataSortedEpochs[:,0,:]==sortedEps[e]]
        
        for d in range(0,len(dataProcess),1):
            dataArray = numpy.empty([5,1],dtype=object)
            
            for l in range(1,len(dataProcess[d]),1):
                dataProduct =  dataProcess[d][l]
                satDataBase[cou_ep,l,shp]=ReadOutSatData(dataProcess[d][l], shp_x, shp_y)
            satDataBase[cou_ep,0,shp]=dataProcess[d][0]
            cou_ep+=1
          
    dataArray =  satDataBase[:,:,shp][satDataBase[:,1,shp]!=0]

    # Matplotlib - create Plots for each Feature in Shapefile
    # -------------------------------------------------------
    xTickList = []
    for i in range(0,dataArray.shape[0],1):
        xTickList.append(str(dataArray[i,0]))    

    xAxeArr = numpy.arange(1,dataArray.shape[0]+1,1)
    sp = fig.add_subplot(numPoi,2,shp+1)
    #sp = plt.subplot(1,numPoi,shp+1)
    absfig = sp.plot(xAxeArr,dataArray[:,1],'r',markersize=0.5,label='absolut Kal.')
    relfig = sp.plot(xAxeArr,dataArray[:,2],'g',markersize=0.5,label='relativ Kal.')
    piffig = sp.plot(xAxeArr,dataArray[:,3],'*b',markersize=0.5,label='PIF Kal.')
    rbafig = sp.plot(xAxeArr,dataArray[:,4],'*k',markersize=0.5,label='RBA Kal.')
    sp.set_ylabel('Landsat Reflectence [%]')
    sp.set_xlabel('Year - DOY []')
    sp.set_ylim([-100,dataArray[:,1].max()+500])
    sp.set_xticks(numpy.arange(xAxeArr.min(),xAxeArr.max()+1,1.0))
    sp.set_xticklabels(xTickList,rotation=45)
    #sp.set_xticks(xTickList)
    sp.legend(bbox_to_anchor=(0.30, 1), loc=1)
    sp.set_title('plot with shp Fid: '+str(shp)+'\nLSBand: '+str(VisBand))
    sp.grid()
    plt.close()

figName = os.path.join(plotsDir,'S2BIOM_LSBand_'+str(VisBand)+'_'+str(UserYear)+'_TS_plot.png')
fig.savefig(figName,dpi=109)
    

#for i in range(0,satDataBase.shape[0],1):
    

    
                           
                           

        
    

        
        

    

    
            

            
        
    



print "Programm ENDE"
