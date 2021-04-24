# -*- coding: cp1252 -*-
"""
 Source Name:   MODIS_Poly_FIT.py
 Version:       Python 2.7.3 x64
 Author:        Paul Arzberger
 Date:          06 Frebruary 2013
 #Usage:        MODIS_select_EVI_max <input folder> <output folder> <output extension>
 Arguments:     in_dir_mod                   - the path to the input files
                in_dir_myd
                out_dir                      - the path to the output images
                output extension             - defines the output format / extension
                                               if not specified, arcinfo grid will be used
                theme_file                   - specifies the themes to be stacked
                                               to change a specified theme you have to change
                                               the index in row 33 "theme_file = list_mod[0].split(".")"
                                               list_mod[0] == RED
                                               list_mod[1] == NIR
                                               list_mod[2] == ???
                                               list_mod[3] == ???
                                               list_mod[4] == ???
                                               list_mod[5] == ???
                                               list_mod[6] == ???
                theme_qal                      list_qal[0] == BRDF_Albedo_Ancillary
                                               list_qal[1] == BRDF_Albedo_Band_Quality
                                               list_qal[2] == BRDF_Albedo_Quality
                                               list_qal[3] == Snow_BRDF_Albedo
"""
# INPUT GENERELL Gleich grosze bloecke mit nodata aufgefuellt mit gleicher aufloesung und gleichem
# zeitstempel am beginn des filenamens T2008239_B01_originalname  ... mit jahr und doy
# Optional Qualityfile dazu Q2008239_originalname
# Zeitbezug wenn erste szene T2001100 folgt differenz = 7 jahre und 139 Tage, da Schaltjahre dazwischen
# time funktion mit differenz der Tage also ca. 7*365=2555 + 1 schaltjahr + 139 = 2695 fuer xv vektor
# ==============================================================
# ============ Functionlisting =================================
def fitq(lv,pv,xv):
    # Quadratischer Fit Input Matrix in Spalten Pixelwerte in Zeilen die Zeitinformation
    # lv ... Beobachtungsvektor = Grauwerte bei MODIS in Prozent z.B. (15, 12 ....)
    # pv ... Gewichtsvektor mit  p = 1 fuer MCD43A2 = 0 0.2 bei MCD43A2=1 (bei MODIS) in erster
    # Iteration, der bei den weiteren Iterationen entsprechend ueberschrieben wird.
    # xv ... Zeit in day of year. Damit die Integerwerte bei Quadrierung nicht zu groß werden anstatt
    # direkte doy's die Differenz zu Beginn, also beginnend mit 1 doy's
    # A [ax0, ax1, ax2] Designmatrix
    # Formeln aus 
    
    ax0 = xv**0    # Vektor Laenge = 15 alle Elemente = 1 aber nur derzeit so bei Aufruf, spaeter bei z.B.
    # Fit von Landsat Aufnahmen doy Vektor z.B. [220, 780, 820, 1600 ...]
    ax1 = xv**1    # Vektor Laenge = 15 [1, 2 , 3 , 4 ... 15]
    ax2 = xv**2    #                [ 1 , 4 , 9 ... 225]

    # ATPA Normalgleichungsmatrix
    a11 = numpy.nansum(ax0*pv*ax0,0)
    a12 = numpy.nansum(ax0*pv*ax1,0)
    a13 = numpy.nansum(ax0*pv*ax2,0)
    
    a22 = numpy.nansum(ax1*pv*ax1,0)
    a23 = numpy.nansum(ax1*pv*ax2,0)
    a33 = numpy.nansum(ax2*pv*ax2,0)

    # Determinante (ATPA)
    det = a11*a22*a33 + a12*a23*a13 \
        + a13*a12*a23 - a13*a22*a13 \
        - a12*a12*a33 - a11*a23*a23 \
    
    # Invertierung (ATPA) mit: Quelle xxx mit Zitat
    # da die Inverse von A symmetrisch ueber die Hauptdiagonale ist, entspricht ai12 = ai21
    # (ATPA)-1
    ai11 = (a22*a33 - a23*a23)/det
    ai12 = (a13*a23 - a12*a33)/det
    ai13 = (a12*a23 - a13*a22)/det    
    ai22 = (a11*a33 - a13*a13)/det
    ai23 = (a13*a12 - a11*a23)/det       
    ai33 = (a11*a22 - a12*a12)/det

    # ATPL mit Bezeichnung vx0 fueer Vektor x0 nansum ... fuer nodata-summe
    vx0 =  numpy.nansum(ax0*pv*lv,0)
    vx1 =  numpy.nansum(ax1*pv*lv,0)
    vx2 =  numpy.nansum(ax2*pv*lv,0)
    
    # Quotienten der quadratischen Gleichung ... bzw. Ergebnis dieser Funktion
    a0 = ai11*vx0 + ai12*vx1 + ai13*vx2
    a1 = ai12*vx0 + ai22*vx1 + ai23*vx2
    a2 = ai13*vx0 + ai23*vx1 + ai33*vx2

    
    return a0,a1,a2

# Linearer Fit, wenn zu wenige Beobachtungen im Zeitfenster vorliegen nach def. Kriterien
# Bezeichnungen wie bei fitq
def fitl(lv,pv,xv):
    ax0 = xv**0  # schneller im vergleich zu funktion ones da kein gesonderter funktionsaufruf
    ax1 = xv

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

    return a0,a1


# ==============================================================
# ============ IMPORTS, Variabledeklaration ====================
import os, osgeo, numpy, time, glob, math
# os    ... sowieso in Python daben (operatingsystem uabhaengigkeit)
# osgeo ... extra download = GDAL MODUL sowie Vektordaten etc. inkludiert
# numpy ... extra download der Standart-Mathematik Library
# time  ... sowieso in Python dabei. hier fuer Laufzeitabfrage des Programms
# glob  ... sowieso in Python dabei. fuer Namenssuchfilter z.B. alle tifs mit Wildcard etc.
# math  ... sowieso in Python dabei (evt. nicht notwendig hier, da ohnehin numby)

from numpy import dot, nan   # damit die Funktionsschreibweise kuerzer ist z.b. anstatt numpy.nan nur nan
from osgeo import gdal, gdalconst # gdalconst ... evt. loeschen
from osgeo.gdal import *   # sollte man eigentlich nicht machen, da die Namen evt. in anderen Modulen gleich

start = time.clock()

# Directories 

MCD_dir = r"D:\DIPLOM\data\tiff\MCD43A4"  # mcd fuer mcd43a4 daten, wenn moeglich mcd_dir als Argument uebergeben
QAL_dir = r"D:\DIPLOM\data\tiff\MCD43A2"  # wenn moeglich als argument uebergeben

OUT_dir = r"D:\DIPLOM\data\fitted\poly\MCD43A4"  # wenn moeglich als argument

tileDirs = os.listdir(MCD_dir)

#for tile in range(0,len(tileDirs),1):
for tile in range(0,1,1):
    
    try:
        print "\n==========================================="
        print " - Processing Tile: \t", tileDirs[tile]
        

        mcd_dir = os.path.join(MCD_dir,tileDirs[tile]) # mcd fuer mcd43a4 daten, wenn moeglich mcd_dir als Argument uebergeben
        qal_dir = os.path.join(QAL_dir,tileDirs[tile])  # wenn moeglich als argument uebergeben

        out_dir = os.path.join(OUT_dir, tileDirs[tile])
        

        # =============== Create Input Lists =============================
        # ================================================================


        ws = os.chdir(mcd_dir)          # workspace bildverzeichnis
        list_mcd = glob.glob('*.tif')   # Fileliste mit wildcart und tifs
        list_mcd.sort()                 # sortieren, muss evt. bei anderen Daten angepasst werden, z.B. wenn
        # Landsat und Spot Daten gemischt etc. jedenfalls ergebnis der liste muss zeitlich aufsteigend sein
        # Zeit nur ueber Index beruecksichtigt und nicht ueber doy. Daher evt. umschreiben wenn fuer Landsat etc.
        theme_list = list_mcd[0:7]
        

    
        for theme in range(0,len(theme_list),1):
            try:
                
                theme_name = list_mcd[theme].split('.')[5]  # nimmt ersten Eintrag und Teilt den Namen bei jedem Punkt und nimmt das
                print "\n\nProcessing Theme:\t", theme_name
                print "===========================================\n"
                # 6. Element da Python von 0 wegzaehlt. Bsp. Band1.  als Schleife umsetzen damit ohne edits anwendbar
                

                ws = os.chdir(qal_dir)              # workspace qualtiyverzeichnis
                list_qalt = glob.glob('*.tif')
                list_qalt.sort()

                theme_qal  = list_qalt[2].split('.')[5]  #  fix den 2. bzw. BRDF_ALBEDO_QUALITY s.o.

                list_data = numpy.empty(numpy.ceil(len(list_mcd)/7.),dtype='object')   # Liste aus allen Bilddaten durch 7 da sieben Baender
                # dtype ... object, also Mischung von String und Zahlen moeglich, daher leichter zum handeln
                list_qal  = numpy.empty(numpy.ceil(len(list_qalt)/4.),dtype='object')
                ind = 0  # index der fuer die Speicherung der gefitteten Daten verwendet wird
                cou = 0  # counter ....

                # create LIST MCD
                # in 1er Schritten alle Files der Liste mit Band1 etc. durchsuchen
                for i in range(0,len(list_mcd),1):        
                    s_mcd = list_mcd[i].find(theme_name)  # list_mcd[i] ... Liste der Namen der Bilder, Finde Band1 etc.

                    if s_mcd>0:                           # wenn keines gefunden, dann s_mcd oben auf -1 gesetzt, ansonsten pos.
                        list_data[cou] = os.path.join(mcd_dir, list_mcd[i])  # Liste mit allen z.B. Band1 Daten 
                        cou = cou + 1
                    else:
                        continue

                # Create LIST QAL
                # gleich wie MCD Liste oben
##                cou = 0
##                for i in range(0,len(list_qalt),1):
##                    s_qal = list_qalt[i].find(theme_qal)
##
##                    if s_qal>0:
##                        list_qal[cou] = os.path.join(qal_dir,list_qalt[i])
##                        cou = cou +1
                os.chdir(qal_dir)
                list_qaltmp = glob.glob('*.'+theme_qal+'.tif')
                cou = 0
                
                for i in range(0,len(list_qaltmp),1):
                    list_qal[cou] = os.path.join(qal_dir,list_qaltmp[i])
                    cou += 1
                    

                # defines out_dir as workspace
                os.chdir(out_dir)

                # ================================== FIT SAT DATA ==============================
                # ==============================================================================


                # Windowsize/Blocksize/Weight
                # ===========================
                window = 15   # Zeitfenster mit 15 Stueck
                blocky = 150  # Block y ... 150 x 150 je Block (Teiler von 2400 x 2400 Kachelgroesze)
                blockx = 2400 # ABFRAGE DER BLOCKGROESZE und ... mit floor oder aehnliches auf gemeinsamen Teiler
                ref_info = {} 
                weight = .2

                # variabledeklarations
                cou = 0  # Zaehler
                ind = 0  # Index
                win_arr = range(0,window,1)     # zahlenreihe von 0 bis 15 
                fit_nr = numpy.median(win_arr)  # index fuer den gefitteten layer z.b. 8 bei 15 windowsize

                fit_rni = ".Polyfit.NanSum.W"+str(window)+".tif"  # rastername und zusatz mit windowsize

                # definiere multidim. array mit window in 3. dim., also hier 15x2400x2400
                ras_data = numpy.empty([window,blockx,blockx],dtype=float)  # definiere multidim array mit der groeße einer Satszene auf den die Rasterdaten ausgelesen werden
                qal_data = numpy.empty([window,blockx,blockx],dtype=float)
                pv = numpy.empty([window,blocky*blocky], dtype=float)

                lv = numpy.empty([window,blocky*blocky], dtype=float)       # multidim array in bestimmte form gebracht auf den die reshapeten Rasterbloecke gelesen werden
                A  = numpy.ones([window,3],dtype=float)                     # allokiere A-matrix
                fit = numpy.ones([window,blocky*blocky], dtype=float)       # allokiere matrix fuer gefittete daten
                ras_fit = numpy.empty([2400,2400], dtype=float)             # ^
                                                                            # |
                d_lv = numpy.ones([window,blocky,blocky], dtype=float)      # |
                sigm = numpy.ones([window,blocky*blocky], dtype=float)      # = Array allokationen
                nu_pv = numpy.ones([window,blocky,blocky], dtype=float)     # |
                n0 = numpy.ones([window,blocky*blocky], dtype=float)        # |
                NaN = numpy.nan                                             # v

                center = round(window/2)                                    # berechne den mittleresten layerindex der die gefittet information beinhaltet
                half = numpy.floor(window/2)

                # create designmatrix

                for i in range(0,window,1):
                    A[i,:]=[(i+1)**0,(i+1)**1,(i+1)**2]

                xv = A[:,1].reshape(window,1)



                # Info-Raster                               # hier wird ein Rasterfile geoeffnet um die Projektionsinfomrmation auszulesen
                ras = gdal.Open(list_data[0], GA_ReadOnly)  # oeffnen
                ref_info[0]=ras.GetProjection()             # verspeichern der info in einer liste
                ref_info[1]=ras.GetGeoTransform()

                driver = ras.GetDriver()

                # Initialise datablock

                for t in range(0,window,1):                             # zubeginn werden die ersten Raster (Anzahl abhaengig von der windowsize) komplett auf einen multidim
                    ras = gdal.Open(list_data[t], GA_ReadOnly)          # Array gelesen
                    ras_data[t,:,:] = ras.ReadAsArray()

                    qal = gdal.Open(list_qal[t], GA_ReadOnly)           # dies gilt auch fuer die Qualitaetsfiles
                    qal_data[t,:,:] = qal.ReadAsArray()

                
                btmp = ras.GetRasterBand(1)
                noDataValue = btmp.GetNoDataValue()
                del btmp

                # Walk through epochs
                for ts in range(0,len(list_data),1):                    # dies hier ist jene schleife die durch alle Epochen iteriert
                #for ts in range(0,2,1):
                    print "Calculate Polyfit epoch nr :", ts ," of ",len(list_data)         # Bildschirmausgabe welche Epoche gerade Prozessiert wird

                    
                        
                    # Create Outputfile                                                     
                    out_ras_name = list_data[fit_nr+ts].split("\\")[6][:-4]                 # hier wird der Name des  Rasterfile erstellt in welches die gefitteten daten geschrieben werden
                    out_ras_dir = os.path.join(out_dir,out_ras_name+fit_rni)                # Namensgebung - verknuepfung des aktuellen Rasternamen mit der Output directory
                    print "Create Outputrasterfile : ",out_ras_dir                          # bildschirmausgabe - pfad des Out-Files
                    out_ras = driver.Create(out_ras_dir,2400,2400,1,gdalconst.GDT_Int16, options=[ 'COMPRESS=LZW' ])         # Hier wird das Rasterfile nun tatsaechlich erstellt welches die gefitten daten beinhalten soll
                    out_ras.SetProjection(ref_info[0])                                      # Projektion des neuen Rasterfiles mit der selben Projektionsinfo wie die ausgangsdaten
                    out_ras.SetGeoTransform(ref_info[1])


                    print "Sucessfull!"
                    cou = 0     # Counter der die Spalten erhoeht : benoetigt fuer Blockauslese und Blockschreiben
                    ind = 0     # Index der die zeilen Erhoeht : benoetigt fuer Blockauslese und Blockschreiben
                    if ts==0:
                    
                        for r in range(0,blockx,blocky):        # Iteration die die Zeilen durchlaeuft - Blockauslese/ Verarbeitung / Verspeicherung
                            ind=0
                            print "Start processing row :", r
                            for c in range(0,blockx,blocky):    # Iteration die die Spalten durchlaeuft - Blockauslese/ Verarbeitung / Verspeicherung

                            
                                
                                n0 = numpy.ones([window,blocky*blocky], dtype=float)        # matrix welche verwendet wird um die Anzahl der verwendbaren beobachtungen festzustellen, hat selbe dimension wie der datenblock!
                                lv = ras_data[:,(cou*blocky):(cou+1)*blocky,(ind*blocky):(ind+1)*blocky].reshape(window,blocky*blocky)  # lese hier einen datenblock aus und reshape ihn um ihn spaeter prozessieren zu koennen
                                
                                lv = numpy.where(lv==32767,NaN,lv)                          # filtere den Datenblock, alle NoData-Werte sollen auf nan gesetzt werden
                                n0=numpy.where(numpy.isnan(lv),NaN,1)                       # schreiben fuer jedes nan im lv-datenblock ein nan auch in meinen n0 - array - habe somit fuer jede brauchbare beobachtung eine 1, und fuer jede schlechte beobachtung ein nan im array stehen
                                
                                pv = qal_data[:,(cou*blocky):(cou+1)*blocky,(ind*blocky):(ind+1)*blocky].reshape(window,blocky*blocky)  # lese die qualitaetsinformation blockweise aus und shape sie in der gleichen dimension wie lv
                                pv = numpy.where(pv==1,0.1,pv)                              # gewichte die qualitaetsinfo, qualitaetsinfo 1 wird auf 0.1 gesetzt
                                pv = numpy.where(pv==0,1,pv)                                # gewichte die qualitaetsinfo, qualitaetsinfo 0 wird auf 1 gesetzt
                                pv = numpy.where(pv==255,NaN,pv)                            # Nodata werte werden auf nan gesetzt
                                
                                obs_check = numpy.where(pv>0,1,0)                           # zaehle hier die verwendeten beobachtungen, 
                                obs_check = numpy.nansum(obs_check,0).reshape(blocky,blocky)# summiere meine matrix und erhalte die tatsaechliche anzahl an beobachtungen , hier in block und nicht in array dim
                                lv_fit_nr = lv[fit_nr].reshape(blocky,blocky)               # waehle hier noch mal die originaldaten der zu fittenden epoche aus, wird spaeter bei datenluecken verwendet
                                
                                    
                                #iv = numpy.where(pv>1)[0]
                                 
                                yma = numpy.ones([window,blocky**2], dtype=float)*numpy.nanmax(lv,0)        # schreibe hier die maximalwerte meiner window-size-epoche auf eine matrix
                                ymi = numpy.ones([window,blocky**2], dtype=float)*numpy.nanmin(lv,0)        # schreibe hier die minimalwerte meiner window-size-epoche auf eine matrix

                                nl = numpy.nansum(n0[0:center,:],0)                                         # erhalte hier die anzahl der beobachtungen linksseitig der mittelepoche
                                nr = numpy.nansum(n0[center+1:,:],0)                                        # erhalte hier die anzahl der beobachtungen rechtsseitig der mittelepoche
                                nc = numpy.nansum(n0[center-1:center+1,:],0)                                # erhalte hier die anzahl der beobachtungen um die zentralepoche
                                n  = numpy.nansum(n0,0)                                                     # erhalte hier die gesamtanzahl der verwendbaren beobachtungen
                               
                                # First Fit
                                [a0,a1,a2] = fitq(lv,pv,xv)         #schicke meine beobachtungen, meinen gewichtsmatrix(vektor), und meine zeitinformation in die funktion zur ermittlung der parameter des quadratischen ppolynoms
                                                                     
                                fit = a0 + a1*xv + a2*(xv**2)       # berechne mir meine gefitteten daten
                                
                                # calculate nu weight               # berechne mir nun ein neues aus :
                                d_lv = abs(fit-lv)                  # der differenz zu ursprünglichen beobachtungen und gefitteten werten
                                d_lv[d_lv<1]=1                      # differenz werte die kleiner eins sind werden gleich auf 1 gesetzt
                                sig = numpy.nansum(d_lv,0)          # summiere meine deltawerte 
                                sigm = sigm*sig                     # gewichte alle epochen sig[window, bloxcky**2]

                                pv_nu = sigm/d_lv                   # neues gewicht 

                                # Second Fit
                                [a0,a1,a2] = fitq(fit,pv_nu,xv)     # ermittle nun die parameter fuer eine quad polynom erneut, mit den gefitteten daten, dem neuen gewicht
                                
                                fit = numpy.round(a0 + a1*xv + a2*(xv**2)) # berechne die gefitten daten erneut

                                # linear Fit                            ermittle hier an welchen positionen linksseitige oder rechtsseitige datenluecken auftreten oder ob gar zu wenig daten vorhanden sind
                                                                
                                ids = numpy.concatenate(( numpy.where(nl<=3),numpy.where(nr<=3), numpy.where(nc<=0), numpy.where(n<=half)),axis=1 )     # dann wird an den auftretenden stellen ein linerarer fit durchgeführt
                                iv = numpy.unique(ids)
                                [a0,a1] = fitl(fit[:,iv],pv_nu[:,iv],xv)

                                fit[:,iv] = numpy.round(a0 + a1*xv)             # linearer fit an den datenluecken

                                
                                fit = numpy.where(fit>yma,yma,fit)              # falls der gefittete wert hoeher als der vorkommende maximalwert ist, wird gleich der maximalwert herangezogen
                                fit = numpy.where(fit<ymi,ymi,fit)              # das gleiche fuer den minimalwert
                                
                                fit = fit.reshape(window, blocky,blocky)        # hier wird dern "datenvektor" wieder auf mulltidimensionale matrizenform gebracht

                                

                                fit_layer = fit[fit_nr]                         # waehle meine gefittete epoche aus 
                                fit_layer = numpy.where(obs_check<=2,yma[center].reshape(blocky,blocky),fit_layer)      # schau nochmal wenn die beobachtungsanzahl zu gering ist, nehme ich den maximal wert her
                                ras_fit[(cou*blocky):(cou+1)*blocky,(ind*blocky):(ind+1)*blocky] = fit_layer            # verspeichere meinen gefitteten daten auf 
                                
                                sigm = sigm**0              # sigmamatrix fuer das neue gewicht auf null setzten
                                ind += 1                    # spaltenindex erhoehen

                            cou += 1                        # zeilenindex erhoehen

                        del ras, qal
                    else:
                        # Update ras_data
                        print "Get IN"

                        ras_data[0:-1] = ras_data[1:]           # updaten des gesamten Rastermaterials um eine neue epoche, entfernen der aeltesten epoche
                        ras = gdal.Open(list_data[t+ts], GA_ReadOnly)
                        ras_data[window-1,:,:] = ras.ReadAsArray()
                        print "weiter"
                        qal_data[0:-1] = qal_data[1:]           # updaten der gesamten qualityinfo um eine neue epoche, entfernen der aeltesten epoche
                        qal = gdal.Open(list_qal[t+ts], GA_ReadOnly)
                        qal_data[window-1,:,:] = qal.ReadAsArray()
                        cou = 0
                        
                        for r in range(0,blockx,blocky):            # hier beginnt wieder alles von vorne, gleiche vorgehensweise wie oben beschrieben
                            print "Start processing row :", r
                            ind = 0
                            for c in range(0,blockx,blocky):
                        

                                n0 = numpy.ones([window,blocky*blocky], dtype=float)
                                lv = ras_data[:,(cou*blocky):(cou+1)*blocky,(ind*blocky):(ind+1)*blocky].reshape(window,blocky*blocky)
                                
                                lv = numpy.where(lv==32767,NaN,lv)
                                n0=numpy.where(numpy.isnan(lv),NaN,1)
                                
                                pv = qal_data[:,(cou*blocky):(cou+1)*blocky,(ind*blocky):(ind+1)*blocky].reshape(window,blocky*blocky)
                                pv = numpy.where(pv==1,0.1,pv)
                                pv = numpy.where(pv==0,1,pv)
                                pv = numpy.where(pv==255,NaN,pv)
                                
                                obs_check = numpy.where(pv>0,1,0)
                                obs_check = numpy.nansum(obs_check,0).reshape(blocky,blocky)
                                lv_fit_nr = lv[fit_nr].reshape(blocky,blocky)
                                
                                    
                                #iv = numpy.where(pv>1)[0]
                                 
                                yma = numpy.ones([window,blocky**2], dtype=float)*numpy.nanmax(lv,0)
                                ymi = numpy.ones([window,blocky**2], dtype=float)*numpy.nanmin(lv,0)

                                nl = numpy.nansum(n0[0:center,:],0)
                                nr = numpy.nansum(n0[center+1:,:],0)
                                nc = numpy.nansum(n0[center-1:center+1,:],0)
                                n  = numpy.nansum(n0,0)
                               
                                # First Fit
                                [a0,a1,a2] = fitq(lv,pv,xv)
                 
                                fit = a0 + a1*xv + a2*(xv**2)
                                
                                
                                d_lv = abs(fit-lv)
                                d_lv[d_lv<1]=1
                                sig = numpy.nansum(d_lv,0)
                                sigm = sigm*sig

                                pv_nu = sigm/d_lv

                                # Second Fit
                                [a0,a1,a2] = fitq(fit,pv_nu,xv)
                                
                                fit = numpy.round(a0 + a1*xv + a2*(xv**2))

                                # linear Fit
                                ids = numpy.concatenate(( numpy.where(nl<=3),numpy.where(nr<=3), numpy.where(nc<=0), numpy.where(n<=half)),axis=1 )
                                iv = numpy.unique(ids)
                                [a0,a1] = fitl(fit[:,iv],pv_nu[:,iv],xv)

                                fit[:,iv] = numpy.round(a0 + a1*xv)

                                
                                fit = numpy.where(fit>yma,yma,fit)
                                fit = numpy.where(fit<ymi,ymi,fit)
                                
                                fit = fit.reshape(window, blocky,blocky)
                                
                                
                                

                                fit_layer = fit[fit_nr]
                                fit_layer = numpy.where(obs_check<=2,yma[center].reshape(blocky,blocky),fit_layer)
                                ras_fit[(cou*blocky):(cou+1)*blocky,(ind*blocky):(ind+1)*blocky] = fit_layer
                                
                                sigm = sigm**0
                                ind += 1
                            cou += 1        
                        
                        



                    
                    ras_fit_band = out_ras.GetRasterBand(1)         # um die gefitteten daten in mein erstelltes file schreiben zu koennen muss ich zuerst ein Band ansprechen auf welches dies geschrieben werden soll
                    #ras_fit[ras_fit>=10000]=NaN                       # werte die groesser als 10000 sind werden als unrealistisch angesehen (optional)
                    ras_fit_band.WriteArray((ras_fit))              #+ schrebe meine gesamte gefittete matrix in mein neues rasterfile
                    ras_fit_band.SetNoDataValue(NaN)                # lasse nan als NoDataValue interpretieren
                    ras_fit_band.FlushCache()                       # lere den speicher auf der variable des bandes

                    epoche_t = time.clock()                         # zeiterfassung - fit eines Bildes
                    
                    print "Time elapsed :", epoche_t-start          # zeitausgaben am bildschirm
                del out_ras, ras,qal, out_count_ras
            except:
                
                print "Windowende für Band: \t",theme_name,"erreicht!"
                
            
        
    
    except:
        print "Fit-Windowende erreicht!!"
        continue
ende = time.clock()
print "Time elapsed : ", ende-start
print "Programm ENDE"


