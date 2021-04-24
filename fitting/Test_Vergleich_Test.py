# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:43:21 2020

@author: Thomi Hahn
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib as mpl
import scipy.stats as stats
import numpy
import time
import math


summe = 0
start = time.time()
start_proc = time.process_time()

if __name__ == "__main__":
       
    
##########################################################################################################################################
#####################################          Funktion Ausreissertest    ################################################################
##########################################################################################################################################  

    def robust_grubb(v, einseitig=True):

        # v sind die daten ( numpy array, pandas dataframe oder was auch immer das es zu überprüfen gibt
    
        #initialize parameters
        G=10000
        z_alpha = 100
        
        n = len(v)
        #alpha = 0.05    #95.0%
        alpha = 0.003    #99.7%
        z_alpha = (n-1.)/numpy.sqrt(n) * ( numpy.sqrt(  stats.t.ppf(alpha/(2*n),n-2)**2 / (n-2+stats.t.ppf(alpha/(2*n),n-2)**2)))
        while G>z_alpha:    
    
            mue = numpy.nanmean(v)

            if einseitig:
                std = numpy.nanstd(v, ddof=0)#ddof=0 bedeutet: 1/(n)
            else:
                std = numpy.nanstd(v, ddof=1)#ddof=1 bedeutet: 1/(n-1)
    
            dG = abs(v-mue)             # differenzen zu messwerten
            aktPG = numpy.nanmax(dG)    # aktueller maximaler wert
            G = aktPG/std               # Pruefgroesse
            
            if G> z_alpha:
                #print "Outlier gefunden - Robust Method: \t\t\t Grubbs Test"
                v[dG==aktPG]=numpy.nan      # Ausresser auf NaN legen
                
            else:
                #print "kein aussreisser mehr gefunden"
                break
        return v, mue, std
    
##########################################################################################################################################
#####################################  Daten einlesen - DJI Absolutdistanz    ################################################

    input_txt_DJI = r"Punktwolke_DJI_Profil_2.txt"
    data_frame_DJI = pd.read_csv(input_txt_DJI, delimiter=";")
    
    abs_dist_DJI = data_frame_DJI["C2C absolute distances[Least Square Plane][r=0.393101]"]
    abs_dist_z_DJI = data_frame_DJI["C2C absolute distances[Least Square Plane][r=0.393101] (Z)"]
    abs_dist_x_DJI = data_frame_DJI["C2C absolute distances[Least Square Plane][r=0.393101] (X)"]
    abs_dist_y_DJI = data_frame_DJI["C2C absolute distances[Least Square Plane][r=0.393101] (Y)"]    

##########################################################################################################################################
#####################################  Funktion Ausreissertest für DJI Absolutdistanz    ################################################

    v_abs_dist, mue_abs_dist, std_abs_dist = robust_grubb(abs_dist_DJI, einseitig=True)
    fit = stats.norm.pdf(v_abs_dist, mue_abs_dist, std_abs_dist)  #this is a fitting indeed
    
    print('\n---------------------------------------------')
    print('Ergebnisse für DJI Absolutdistanz')
    print('---------------------------------------------')
    print('Maximaler Wert: {:.7} [m]'.format(numpy.max(v_abs_dist)))
    print('Minimaler Wert: {:.7} [m]'.format(numpy.min(v_abs_dist)))
    print('Mittelwert: {:.7} [m]'.format(mue_abs_dist))
    print('Standardabweichung: {:.7} [m]'.format(std_abs_dist))
    print('Linke Schranke bei 3 Sigma: {:.7} [m]'.format(mue_abs_dist-(3*std_abs_dist)))
    print('Rechte Schranke bei 3 Sigma: {:.7} [m]'.format(mue_abs_dist+(3*std_abs_dist)))
    print('---------------------------------------------')

##########################################################################################################################################
#####################################  Plot für DJI Absolutdistanz    ################################################
      
    fit = stats.norm.pdf(v_abs_dist, 0, std_abs_dist)  #this is a fitting indeed
    
    fig = plt.figure(1)
    ax = sns.distplot(v_abs_dist, hist=True, kde=False, 
              bins=int(180/4), color = 'darkblue', 
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 2}, norm_hist=True)
    
    ax.plot(v_abs_dist, fit, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    ax.grid(True, color='blue', linestyle='-', linewidth=0.1)
    
    ax.set_xlabel('Absolutdistanzen [m]')
    ax.set_ylabel('Absolute Häufigkeit')
    ax.set_title('Verteilung aus dem DJI Phantom 4 RTK Vergleich')
    
    #plt.legend()
    fig.legend(labels=['Dichtefunktion','Häufigkeit'], loc = 7, bbox_to_anchor=(0.87,0.8),frameon=True)
    
    ax.annotate(r'$\mu={mue}\,[mm]$''\n''$\sigma={std}\,[mm]$'.format(mue=round(mue_abs_dist*1000,2),std=round(0,2)),
                    xy=(0.65, 0.6), xycoords='axes fraction', fontsize=10,
                    #xytext=(0, 100), textcoords='data',
                    ha='left', va='bottom', bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="red", lw=1))

    fig.savefig('Distanz_Absolut_DJI_Vergleich_Test.png', dpi=500, quality=1000, facecolor='w', edgecolor='w', orientation='portrait')
    
#########################################################################################################################################
####################################  Funktion Ausreissertest für DJI X Distanz    #####################################################
    
    v_abs_dist_x, mue_abs_dist_x, std_abs_dist_x = robust_grubb(abs_dist_x_DJI, einseitig=False)
    

    print('\n---------------------------------------------')
    print('Ergebnisse für DJI Distanz in X-Richtung')
    print('---------------------------------------------')
    print('Maximaler Wert: {:.7} [m]'.format(numpy.max(v_abs_dist_x)))
    print('Minimaler Wert: {:.7} [m]'.format(numpy.min(v_abs_dist_x)))
    print('Mittelwert: {:.7} [m]'.format(mue_abs_dist_x))
    print('Standardabweichung: {:.7} [m]'.format(std_abs_dist_x))
    print('Linke Schranke bei 3 Sigma: {:.7} [m]'.format(mue_abs_dist_x-(3*std_abs_dist_x)))
    print('Rechte Schranke bei 3 Sigma: {:.7} [m]'.format(mue_abs_dist_x+(3*std_abs_dist_x)))
    print('---------------------------------------------')
    
##########################################################################################################################################
#####################################  Plot für DJI Distanzen in X Richtung    ##########################################################

    fit_x = stats.norm.pdf(v_abs_dist_x, mue_abs_dist_x, std_abs_dist_x)  #this is a fitting indeed
    
    fig = plt.figure(2)
    ax = sns.distplot(v_abs_dist_x, hist=True, kde=False, 
              bins=int(180/4), color = 'darkblue', 
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 2}, norm_hist=True)
    
    ax.plot(v_abs_dist_x, fit_x, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    ax.grid(True, color='blue', linestyle='-', linewidth=0.1)
    plt.xlim(-0.04,0.04)
    
    ax.set_xlabel('Distanz in X-Richtung [m]')
    ax.set_ylabel('Absolute\nHäufigkeit')
    ax.set_title('Verteilung aus dem DJI Phantom 4 RTK Vergleich')
    
    #plt.legend()
    fig.legend(labels=['Dichtefunktion','Häufigkeit'], loc = 7, bbox_to_anchor=(0.87,0.8),frameon=True)
    
    ax.annotate(r'$\mu={mue}\,[mm]$''\n''$\sigma={std}\,[mm]$'.format(mue=round(mue_abs_dist_x*1000,2),std=round(std_abs_dist_x*1000,2)),
                    xy=(0.65, 0.6), xycoords='axes fraction', fontsize=10,
                    #xytext=(0, 100), textcoords='data',
                    ha='left', va='bottom', bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="red", lw=1))

    fig.savefig('Distanz_X_DJI_Vergleich_Test.png', dpi=500, quality=1000, facecolor='w', edgecolor='w', orientation='portrait')   
  
#########################################################################################################################################
####################################  Funktion Ausreissertest für DJI Y Distanz    #####################################################
    
    v_abs_dist_y, mue_abs_dist_y, std_abs_dist_y = robust_grubb(abs_dist_z_DJI, einseitig=False)
    
    print('\n---------------------------------------------')
    print('Ergebnisse für DJI Distanz in Y-Richtung')
    print('---------------------------------------------')
    print('Maximaler Wert: {:.7} [m]'.format(numpy.max(v_abs_dist_y)))
    print('Minimaler Wert: {:.7} [m]'.format(numpy.min(v_abs_dist_y)))
    print('Mittelwert: {:.7} [m]'.format(mue_abs_dist_y))
    print('Standardabweichung: {:.7} [m]'.format(std_abs_dist_y))
    print('Linke Schranke bei 3 Sigma: {:.7} [m]'.format(mue_abs_dist_y-(3*std_abs_dist_y)))
    print('Rechte Schranke bei 3 Sigma: {:.7} [m]'.format(mue_abs_dist_y+(3*std_abs_dist_y)))
    print('---------------------------------------------')
    
##########################################################################################################################################
#####################################  Plot für DJI Distanzen in Y Richtung    ##########################################################
  
    fit_y = stats.norm.pdf(v_abs_dist_y, mue_abs_dist_y, std_abs_dist_y)  #this is a fitting indeed
    
    fig = plt.figure(3)
    ax = sns.distplot(v_abs_dist_y, hist=True, kde=False, 
              bins=int(180/4), color = 'darkblue', 
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 2}, norm_hist=True)
    
    ax.plot(v_abs_dist_y, fit_y, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    ax.grid(True, color='blue', linestyle='-', linewidth=0.1)
    plt.xlim(-0.02,0.02)
    
    ax.set_xlabel('Distanz in Y-Richtung [m]')
    ax.set_ylabel('Absolute Häufigkeit')
    ax.set_title('Verteilung aus dem DJI Phantom 4 RTK Vergleich')
    
    #plt.legend()
    fig.legend(labels=['Dichtefunktion','Häufigkeit'], loc = 7, bbox_to_anchor=(0.87,0.8),frameon=True)
    
    ax.annotate(r'$\mu={mue}\,[mm]$''\n''$\sigma={std}\,[mm]$'.format(mue=round(mue_abs_dist_y*1000,2),std=round(std_abs_dist_y*1000,2)),
                    xy=(0.65, 0.6), xycoords='axes fraction', fontsize=10,
                    #xytext=(0, 100), textcoords='data',
                    ha='left', va='bottom', bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="red", lw=1))

    fig.savefig('Distanz_Y_DJI_Vergleich_Test.png', dpi=500, quality=1000, facecolor='w', edgecolor='w', orientation='portrait')   
  
##########################################################################################################################################
#####################################  Funktion Ausreissertest für DJI Z Distanz    #####################################################
    
    v_abs_dist_z, mue_abs_dist_z, std_abs_dist_z = robust_grubb(abs_dist_z_DJI, einseitig=False)
    
    print('\n---------------------------------------------')
    print('Ergebnisse für DJI Distanz in Z-Richtung')
    print('---------------------------------------------')
    print('Maximaler Wert: {:.7} [m]'.format(numpy.max(v_abs_dist_z)))
    print('Minimaler Wert: {:.7} [m]'.format(numpy.min(v_abs_dist_z)))
    print('Mittelwert: {:.7} [m]'.format(mue_abs_dist_z))
    print('Standardabweichung: {:.7} [m]'.format(std_abs_dist_z))
    print('Linke Schranke bei 3 Sigma: {:.7} [m]'.format(mue_abs_dist_z-(3*std_abs_dist_z)))
    print('Rechte Schranke bei 3 Sigma: {:.7} [m]'.format(mue_abs_dist_z+(3*std_abs_dist_z)))
    print('---------------------------------------------')
    
##########################################################################################################################################
#####################################  Plot für DJI Distanzen in Z Richtung    ##########################################################

    fit_z = stats.norm.pdf(v_abs_dist_z, mue_abs_dist_z, std_abs_dist_z)  #this is a fitting indeed
    
    fig = plt.figure(4)
    ax = sns.distplot(v_abs_dist_z, hist=True, kde=False, 
              bins=int(180/4), color = 'darkblue', 
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 2}, norm_hist=True)
    
    ax.plot(v_abs_dist_z, fit_z, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    ax.grid(True, color='blue', linestyle='-', linewidth=0.1)
    plt.xlim(-0.15,0.15)
    
    ax.set_xlabel('Distanz in Z-Richtung [m]')
    ax.set_ylabel('Absolute Häufigkeit')
    ax.set_title('Verteilung aus dem DJI Phantom 4 RTK Vergleich')
    
    #plt.legend()
    fig.legend(labels=['Dichtefunktion','Häufigkeit'], loc = 7, bbox_to_anchor=(0.87,0.8),frameon=True)
    
    ax.annotate(r'$\mu={mue}\,[mm]$''\n''$\sigma={std}\,[mm]$'.format(mue=round(mue_abs_dist_z*1000,2),std=round(std_abs_dist_z*1000,2)),
                    xy=(0.65, 0.6), xycoords='axes fraction', fontsize=10,
                    #xytext=(0, 100), textcoords='data',
                    ha='left', va='bottom', bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="red", lw=1))

    fig.savefig('Distanz_Z_DJI_Vergleich_Test.png', dpi=500, quality=1000, facecolor='w', edgecolor='w', orientation='portrait')
    
    
    
##########################################################################################################################################
#####################################  Plot Vergleich#####################################################################################


    plt.subplot(3,1,1)
    plt.plot(v_abs_dist_x, fit_x, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    plt.title('Vergleich der Distanzen')
    plt.xlabel('Distanz in X-Richtung [m]')
    plt.ylabel('Absolute\nHäufigkeit')
    
    plt.subplot(3,1,2)
    plt.plot(v_abs_dist_y, fit_y, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    plt.xlabel('Distanz in Y-Richtung [m]')
    plt.ylabel('Absolute\nHäufigkeit')
    
    plt.subplot(3,1,3)
    plt.plot(v_abs_dist_z, fit_z, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    plt.xlabel('Distanz in Z-Richtung [m]')
    plt.ylabel('Absolute\nHäufigkeit')    
    
    plt.show()
    
    
    
    
    
    fig.suptitle('Vergleich der Distanzen')

    ax1 = sns.distplot(v_abs_dist_x, hist=True, kde=False, 
              bins=int(180/4), color = 'darkblue', 
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 2}, norm_hist=True)    
    ax1.plot(v_abs_dist_x, fit_x, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    ax1.set_xlabel('Distanz in X-Richtung [m]')
    ax1.set_ylabel('Absolute Häufigkeit')

    ax2 = sns.distplot(v_abs_dist_y, hist=True, kde=False, 
              bins=int(180/4), color = 'darkblue', 
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 2}, norm_hist=True)    
    ax2.plot(v_abs_dist_y, fit_y, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    ax2.set_xlabel('Distanz in Y-Richtung [m]')
    ax2.set_ylabel('Absolute Häufigkeit')
    
    ax3 = sns.distplot(v_abs_dist_z, hist=True, kde=False, 
              bins=int(180/4), color = 'darkblue', 
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 2}, norm_hist=True)    
    ax3.plot(v_abs_dist_z, fit_z, color = 'red', marker='.', ms=0.5, linewidth=0, label = 'Dichtefunktion')
    ax3.set_xlabel('Distanz in Z-Richtung [m]')
    ax3.set_ylabel('Absolute Häufigkeit')

    fig.savefig('XYZ_Vergleich_Plot.png', dpi=500, quality=1000, facecolor='w', edgecolor='w', orientation='portrait')
    
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################    

    ende = time.time()
    ende_proc = time.process_time()
    print('######################################################')
    print('Gesamtzeit: {:.0f} Minuten oder {:.1f} Sekunden'.format(int((ende-start)/60),(ende-start)))
    print('Systemzeit: {:.0f} Minuten oder {:.1f} Sekunden'.format(int((ende_proc-start_proc)/60),(ende-start)))
    
    
    print("Programm ENDE")