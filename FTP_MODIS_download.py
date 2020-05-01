import ftplib
import os
import glob
import threading



if __name__ == "__main__":
    ftp_server = "https://e4ftl01.cr.usgs.gov/MOTA/MCD43A2.006/"
    ftp_dir_A2 = r"https://e4ftl01.cr.usgs.gov/MOTA/MCD43A2.006/"
    ftp_dir_A4 = r"https://e4ftl01.cr.usgs.gov/MOTA/MCD43A4.006/"

    # hdf dir
    out_dir = r"M:\modis\v6\hdf\h18v03"


    print("PROGRAMM ENDE")