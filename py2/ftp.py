# ============== FTP DOWNLOAD =====================
import os, ftplib, re

# file Stream

ziel = open("D:\Diplomarbeit\Daten\downloadlist\download_list_for_total_commander\MCD43A4\h18v05_MCD.txt", "w")




ftp_str = "ftp://e4ftl01.cr.usgs.gov"

ftp_adr = "e4ftl01.cr.usgs.gov"
ftp = ftplib.FTP(ftp_adr)
if ftp:
    print "true"
log = ftp.login()
data = []
subdir_data = []
list_data = {}
cou = 0
# main dir
dir_path = "/MOTA/MCD43A4.005"



# ============ Get Subdirectories ======================
maindir=ftp.cwd(dir_path)
ftp.dir(data.append)      # creates list with directories



for w in range(0,len(data),1):
    # =========== Search in Subdirectories =================

    # Searchpattern
    
    pattern = 'h18v05.005.*.hdf'


    subdir_path = str(dir_path + "/" + data[w+1][37:])
    print "Processing Dir:", subdir_path
    subdir = ftp.cwd(subdir_path)
    ftp.dir(subdir_data.append)
    
    for i in range(0,len(subdir_data),1):
        #epoch_search = subdir_data[i].find(pattern)
        epoch_search = re.search(pattern,subdir_data[i])


        if epoch_search >= 0:
            list_data[cou] = subdir_data[i]
            cou = cou + 1
            ziel_str = "" + ftp_str + dir_path + "/" + data[w+1][37:] + "/" + subdir_data[i][38:]+"\n"
            print "Writing :", ziel_str ,"...\n"
            ziel.write(ziel_str)
            
            break
    subdir_data = []
        
        
ftp.quit()
ziel.close()


   
