# ========================================
# meta data to read out
# every 8th day we have a satellite image
#
# Day of Year for every 8 day epoch
# 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
# 209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305, 313, 321, 329, 337, 345, 353, 361
#
# one year has 46 epochs
#
# TS raw starts with 2000057
# TS fit starts with 2000113
# Doy 2000 057 - index: 7
# Doy 2000 113 - index: 14
#
# Doy 2001 001 - index fit: 2000113 + (46-14)
# -
# Indizes to plot
# ---------------
# TS raw - index for 2001001 = 38
# TS
# ========================================


import os
import glob
import time
import socket
import numpy

def main():

    print("Start Processing Plots for yearwise timeseries")

if __name__ == "__main__":

    # 2T USB3 Festplatte und Home Rechner
    if socket.gethostname() in ['XH-AT-NB-108', 'Paul-PC']:
        in_dir_qs = r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
        in_dir_tf = r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
        out_dir_fit = r"E:\MODIS_Data\v6\fitted"

    else:
        print("Check Data Drive Letter !!!!!")

    if os.name == "posix" and socket.gethostname() == "paul-buero":

        in_dir_qs = r"/media/paul/Daten_Diplomarbeit2/MODIS_Data/v6/tiff_single/MCD43A2"
        in_dir_tf = r"/media/paul/Daten_Diplomarbeit2/MODIS_Data/v6/tiff_single/MCD43A4"
        out_dir_fit = r"/media/paul/Daten_Diplomarbeit2/MODIS_Data/v6/fitted"

    elif os.name == "posix" and socket.gethostname() in ["iotmaster", "iotslave1", "iotslave2"]:
        in_dir_qs = r"/home/iot/scripts/dev/projects/timeseries/data/v6/tiff_single/MCD43A2"
        in_dir_tf = r"/home/iot/scripts/dev/projects/timeseries/data/v6/tiff_single/MCD43A4"
        out_dir_fit = r"/home/iot/scripts/dev/projects/timeseries/data/v6/fitted"
    else:
        in_dir_qs = r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
        in_dir_tf = r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
        out_dir_fit = r"E:\MODIS_Data\v6\fitted"


    tile = "h18v04"

    sat_raw_indir = in_dir_tf
    sat_qual_indir = in_dir_qs
    sat_fitted_indir = out_dir_fit

    doy = [doy for doy in range(1,365,8)]
    print("doy: ", doy)


    user_year = 2004

    # the fitted time series starts with 2000113
    epoch_num_begin_raw_index = 7 #2000057 - index
    epoch_num_begin_fit = 0 #2000113

    os.chdir(os.path.join(out_dir_fit, tile))

    fit_tif_list = glob.glob("*.tif")

    print("len of fit_tif_list: ", len(fit_tif_list))

    print("Programm ENDE")