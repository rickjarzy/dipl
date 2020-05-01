import os, ftplib, re

# file Stream


out_dir = r"F:\DIPLOM\data\hdf\MCD43A4"
#sat = "MYD13Q1"
sat = "MCD43A4"

listOutDir = os.listdir(out_dir)

cou=0
list_outdir = {}
for i in range(0,len(listOutDir),1):
    if os.path.isdir(os.path.join(out_dir,listOutDir[i])):
        list_outdir[cou]=listOutDir[i]
        cou +=1
        
                     






print "Programm ENDE"
