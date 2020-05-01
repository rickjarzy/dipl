# Paul Arzberger



# ============== FTP DOWNLOAD =====================
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
        
                     



ftp_str = "ftp://e4ftl01.cr.usgs.gov"

ftp_adr = "e4ftl01.cr.usgs.gov"

con=0
vers = 0
print "Tryn to connect to : ",ftp_str,"\n..."
while(con==0):
    try:
        vers+=1
        print "Versuch Nr: ",vers
        ftp = ftplib.FTP(ftp_adr)

        if ftp:
            print "Connection successful...\n"
            con = 1
    except:
        continue


    
log = ftp.login()
data = []
subdir_data = []
list_data = {}
cou = 0
# main dir
# ftp_path = "/MOLA/MYD13Q1.005" # 250 Meter Dirs
ftp_path = "/MOTA/MCD43A4.005"




# ============ Get Subdirectories ======================
maindir=ftp.cwd(ftp_path)
ftp.dir(data.append)      # creates list with directories

total = len(data)
delta = len(data[-28:])     # epochs since 2012

diff_2012 = total- delta - 1

##for o in range(12,len(list_outdir),1):
##    
##    dir_pat = list_outdir[o]
##    print dir_pat
##
##    # Searchpattern
##    pattern = sat + '.*.'+list_outdir[o]+'.005.*.hdf'
##    print pattern
##
##    # Create Output Path
##    hdf_dir = out_dir +"\\"+dir_pat+"\\"+sat
##    print hdf_dir
##    for w in range(diff_2012,len(data),1):
##        
##        # =========== Search in Subdirectories =================
##
##
##
##
##        subdir_path = str(ftp_path + "/" + data[w][37:])
##        print "\nProcessing Dir:", subdir_path
##        subdir = ftp.cwd(subdir_path)
##        ftp.dir(subdir_data.append)
##        #for i in range(0,len(subdir_data),1):
##        for i in range(0,len(subdir_data),1):
##            #epoch_search = subdir_data[i].find(pattern)
##            epoch_search = re.search(pattern,subdir_data[i])
##
##
##            if epoch_search >= 0:
##                
##
##
##        
##                list_data[cou] = subdir_data[i]
##                cou = cou + 1
##                ziel_str = "" + ftp_str +"/"+ ftp_path + "//" + data[w][37:] + "//" + subdir_data[i][40:]+"\n"
##                print "Writing :", ziel_str ,"...\n"
##                
##                print "Start Downloading file into dir: ", hdf_dir
##                print "Filename : ", ziel_str.split('/')[9][:-1]
##                file_download = open(hdf_dir + "\\"+ziel_str.split('/')[9][:-1],"wb")
##                ftp.retrbinary("RETR "+ziel_str.split('/')[9][:-1],file_download.write)
##
##        
##        subdir_data = [] 
##            
ftp.quit()


print "Erfolg - Finished Downloading! "


