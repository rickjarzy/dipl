# Paul Arzberger
# November 2014


import arcpy, os, glob, osgeo, numpy, itertools,matplotlib
from osgeo import ogr
from arcpy.sa import *
from arcpy import env
from matplotlib import pyplot as plt
arcpy.CheckOutExtension('spatial')

def ReadOutSatData(absEpochs,VisBand,shp_x,shp_y):
    
    if absEpochs==None:
        rasData = 0
        
    else:

        if dataProcess[d][l].split("\\")[-1].split('.')[-2] == 'msk':
            ras = gdal.Open(absEpochs, gdal.GA_ReadOnly)
            rasBand = ras.GetRasterBand(1)
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
                    elif rasData == 2:
                        rasData = 0

                else:
                    print "Shp y Coordinate not in SatSzene Extent"
                    rasData=0
                
                
                
            else:
                print "Shp x Coordinate not in SatSzene Extent"
                rasData=0
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
                    elif rasData>=10000:
                        rasData = 0

                else:
                    print "Shp y Coordinate not in SatSzene Extent"
                    rasData=0
                    
            else:
                print "Shp x Coordinate not in SatSzene Extent"
                rasData=0
    return rasData

def GetRasterNames(srchPattern,roots):
    
    epoch = roots.split("\\")[-1]
    sat     = epoch[:3]         # Satname
    pr      = epoch[3:9]        # Path and Row
    year    = epoch[9:13]       # Year
    doy     = epoch[13:16]      # DOY

    if sat == 'LE7':
        satAbsName = 'ls7_'+pr+'_'+year+doy+'_msp'
        PifName = 'ls7_'+pr+'_'+year+doy+'_msp_NoComp_PIF.tif'
        RbaName = 'ls7_'+pr+'_'+year+doy+'_msp_rc.tif'
        MskName = 'ls7_'+pr+'_'+year+doy+'_msk.tif'
        ODRName = 'ls7.'+pr+'.'+year+doy+'.scipy_odr_500k.tif'
    elif sat == 'LT5':
        satAbsName = 'ls5_'+pr+'_'+year+doy+'_msp'
        PifName = 'ls5_'+pr+'_'+year+doy+'_msp_NoComp_PIF.tif'
        RbaName = 'ls5_'+pr+'_'+year+doy+'_msp_rc.tif'
        MskName = 'ls5_'+pr+'_'+year+doy+'_msk.tif'
        ODRName = 'ls5.'+pr+'.'+year+doy+'.scipy_odr_500.tif'
        
    AbsRas = satAbsName+'.tif'
    RelRas = satAbsName+srchPattern[1:]



    # find Pifs
    

    return os.path.join(roots,AbsRas),os.path.join(roots,RelRas),os.path.join(roots,PifName),os.path.join(roots,RbaName),os.path.join(roots,MskName),os.path.join(roots,ODRName),int(year),int(doy)



def CreateTempLineShp(LinePoints,shpDrv,wrkDir):
    tmpShpName = os.path.join(wrkDir,"tmpEBM.shp")
    if os.path.exists(tmpShpName):
        shpDrv.DeleteDataSource(tmpShpName)
        
    ds  = shpDrv.CreateDataSource(tmpShpName)
    lineLyr = ds.CreateLayer('line;', geom_type=ogr.wkbLineString)
    line = ogr.Geometry(ogr.wkbLineString)
    line.AddPoint(LinePoints[0][0],LinePoints[0][1])
    line.AddPoint(LinePoints[1][0],LinePoints[1][1])

    nuFeat = lineLyr.GetLayerDefn()
    fpos = ogr.Feature(nuFeat)
    fpos.SetGeometry(line)
    fpos.SetField('LINE',1)
    lineLyr.CreateFeature(fpos)
    ds.Destroy()
    return tmpShpName
    

shpDirLine = r"L:\LANDSAT_STACK\S2BIOM\shp\profileUTM2.shp"
rasDir = r"L:\Diplom_Daten\test\relcal\p195_027\LE71950272005174EDC01\ls7_195027_2005174_msp.tif"
wrkDir = r"L:\LANDSAT_STACK\S2BIOM\wrkDir"
inWalkDir = [r"L:\LANDSAT_STACK\S2BIOM",r"L:\test\relcal",r"L:\Diplom_Daten\test\relcal",\
             r"L:\LANDSAT_STACK\S2BIOM",r"L:\LANDSAT_STACK\S2BIOM\relcal_195027\done_195027"]
plotsDir = r"L:\LANDSAT_STACK\TSPlots\Profiles\ODR"

##shpDirLine = r"G:\LANDSAT_STACK\S2BIOM\shp\profileUTM.shp"
##rasDir = r"G:\Diplom_Daten\test\relcal\p195_027\LE71950272005174EDC01\ls7_195027_2005174_msp.tif"
##wrkDir = r"G:\LANDSAT_STACK\S2BIOM\wrkDir"
##inWalkDir = [r"G:\LANDSAT_STACK\S2BIOM",r"G:\test\relcal",r"G:\Diplom_Daten\test\relcal",\
##             r"G:\LANDSAT_STACK\S2BIOM",r"G:\LANDSAT_STACK\S2BIOM\relcal_195027\done_195027"]
##plotsDir = r"G:\LANDSAT_STACK\TSPlots"



# UserParameters
# ==============
UserYear = 2010
DOYB = 152       # DOY Beginn
DOYE = 265      # DOY End
srchPattern = '*_calibrated_abscal_OLS.tif'
shpPoiKoords = []
absEpochs = []
relEpochs = []
pifEpochs = []
rbaEpochs = []
mskEpochs = []
odrEpochs = []
VisBand = 4
LsBand = "\\Band_4"
WalkDir = inWalkDir[-1]
matMarkerColor = ['r','g','c','k','b']
matMarkerLabel = ['absolut Kal.','relativ Kal.','PIF Kal.','RBA Kal.','ODR Kal']


# ==============

if os.path.exists(os.path.join(plotsDir,str(UserYear))):
    print "plotsdir exists"
    plotsDir = os.path.join(plotsDir,str(UserYear))
else:
    print "Creating plotsdir subfolder : ", os.path.join(plotsDir,str(UserYear))
    os.makedirs(os.path.join(plotsDir,str(UserYear)))
    
    plotsDir = os.path.join(plotsDir,str(UserYear))
    
    
                  


# creates array with any possible epochs
for roots, dirs, files in os.walk(WalkDir):
    
    os.chdir(roots)
    srchCalFile = glob.glob(srchPattern)
    
    if srchCalFile:
        AbsRas,RelRas,PifRas,RbaRas,MskRas,ODRRas,yearEp,doyEp = GetRasterNames(srchPattern,roots)
        
        if yearEp == UserYear:
            if doyEp>=DOYB and doyEp<=DOYE:
                #print "SatEpoche innerhalb Userinput:\t",yearEp,doyEp
                absEpochs.append(AbsRas)
                relEpochs.append(RelRas)
                pifEpochs.append(PifRas)
                rbaEpochs.append(RbaRas)
                mskEpochs.append(MskRas)
                odrEpochs.append(ODRRas)
 
            else:
                continue            
        else:
            continue



# looks if epochs are real
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
    else: mskEpochs[ep] = None

    if os.path.exists(odrEpochs[ep]):
        print "Exists:\t", odrEpochs[ep]
    else: odrEpochs[ep] = None


satDataSortedEpochs = numpy.empty([len(absEpochs),1,7],dtype=object)
satDataSortedEpochs[:,0,1]=absEpochs
satDataSortedEpochs[:,0,2]=relEpochs
satDataSortedEpochs[:,0,3]=pifEpochs
satDataSortedEpochs[:,0,4]=rbaEpochs    
satDataSortedEpochs[:,0,5]=mskEpochs
satDataSortedEpochs[:,0,6]=odrEpochs

for t in range(0,satDataSortedEpochs.shape[0],1):
    satDataSortedEpochs[t,0,0]=satDataSortedEpochs[t,0,1].split("\\")[-1].split("_")[2]

sortedEps = numpy.unique(numpy.sort(satDataSortedEpochs,0)[:,0,0])
#satDataBase = numpy.zeros([len(absEpochs),4,len(shpPoiKoords)],dtype=int)
satDataProfile = {}


# OGR
# ====

shpDrv = ogr.GetDriverByName('ESRI Shapefile')
dsL = shpDrv.Open(shpDirLine)
lyrL = dsL.GetLayer()
numLines = lyrL.GetFeatureCount()

print numLines

#iterate through Features

for ft in range(0,numLines,1):

    print "Feature NR:\t", ft+1
    actLine = lyrL.GetFeature(ft)
    shpLineGeom = actLine.GetGeometryRef()
    actLineType = actLine.GetField('Id')


    # Create a Temp Line Shpfile for ExtractByMask
    LinePoints = shpLineGeom.GetPoints()
    
    dx = numpy.array(LinePoints[0][0])-numpy.array(LinePoints[1][0])
    dy = numpy.array(LinePoints[0][1])-numpy.array(LinePoints[1][1])

    tmpShpName = CreateTempLineShp(LinePoints,shpDrv,wrkDir)
    print LinePoints
    
    fig = plt.figure(figsize=(45,30))
    
    for profile in range(0,len(sortedEps),1):
        print satDataSortedEpochs[profile,0,0]

        
        sp = fig#fig.add_subplot(len(sortedEps),1,profile+1)
        for epoch in range(1,satDataSortedEpochs.shape[-1],1):
            actualEpochName =  satDataSortedEpochs[profile,0,epoch]
            
            if actualEpochName:     # if actualEpochName is not 'None'
                # Extract By Mask
                if epoch == 1:
                    print "Extracting Line and cloud mask\n..."
                    mskBand = satDataSortedEpochs[profile,0,5]
                    rasBand = actualEpochName+LsBand
                    lineCM = ExtractByMask(mskBand,tmpShpName)
                    linMatCM = arcpy.RasterToNumPyArray(lineCM)

                    line = ExtractByMask(rasBand,tmpShpName)
                    linMat = arcpy.RasterToNumPyArray(line)
                elif epoch == 5:
                    continue
                else:
                    rasBand = actualEpochName+LsBand
                    #print actualEpochName
                    print "Extracting Line\n..."
                    
                    #env.extent = tmpShpName
                    line = ExtractByMask(rasBand,tmpShpName)
                    linMat = arcpy.RasterToNumPyArray(line)
                
                profileData = []
                cloudData = []
                tr = []
                if dx>=0 and dy>=0:  # P1 is Ur Corner and P2 is LL Corner
                    for row in range(linMat.shape[0]-1,-1,-1):
                        profileData.append(list(linMat[row,:][linMat[row,:]!=-32768]))
                        cloudData.append( list(linMatCM[row,:][linMatCM[row,:]!=0]))
                        
                elif dx>=0 and dy<=0: # P1 in LR Corner and P2 in UL Corner
                    for row in range(0,linMat.shape[0],1):
                        profileData.append(list(linMat[row,:][linMat[row,:]!=-32768]))
                        cloudData.append( list(linMatCM[row,:][linMatCM[row,:]!=0]))
                        
                elif dx<=0 and dy<=0: # P1 in LL Corner and P2 in UR Corner
                    for row in range(linMat.shape[0]-1,-1,-1):
                        profileData.append(list(linMat[row,:][linMat[row,:]!=-32768]))
                        cloudData.append( list(linMatCM[row,:][linMatCM[row,:]!=0]))
                        
                elif dx<=0 and dy >=0: # P1 in UL Corner and P2 in LR Corner
                    for row in range(0,linMat.shape[0],1):
                        profileData.append(list(linMat[row,:][linMat[row,:]!=-32768]))
                        cloudData.append( list(linMatCM[row,:][linMatCM[row,:]!=0]))
                        

                arrPD = numpy.array(sum( profileData,[]))    # concatenate to one list and turn into array
                arrCm = numpy.array(sum( cloudData,[]))
                arrCm[arrCm==2]=0
                satDataProfile[satDataSortedEpochs[profile,0,0]]=arrPD
                xAxeArr = numpy.arange(1,len(arrPD)+1,1)
                dataFig = plt.plot(xAxeArr,arrPD*arrCm,matMarkerColor[epoch-1],markersize=0.5,label=matMarkerLabel[epoch-1])
                plt.ylabel('Reflectance [%]')
                plt.xlabel('ProfileArray []')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=1)
                plt.title('Profile of shp Fid: '+str(ft)+'\nTile: '+satDataSortedEpochs[profile,0,1].split("\\")[3]+'Year: '+str(UserYear)+'\nLSBand: '+str(VisBand)+'\nLineType: '+str(actLineType))
                plt.grid(True)
                plt.hold(True)
                
                
            else:
                continue

    plt.close()
    
    figName = os.path.join(plotsDir,'S2BIOM_'+satDataSortedEpochs[profile,0,1].split("\\")[-1][:-7]+'_LSBand_'+str(VisBand)+'_'+str(UserYear)+'_Profil_Fid_'+str(ft)+'.png')
    fig.savefig(figName,dpi=300)
    
##            
        
    
     
        





print "Programm ENDE"
