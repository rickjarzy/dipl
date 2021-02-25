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

# todo: select a year by index
# todo: select different calculations from fit
# todo: select corresponding raw data for same time periode
# todo: create first plot with

import os
import glob
import time
import socket
import numpy

from fit_information import fit_info_all, fit_info_poly, fit_info_fft



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


    doy_full = [doy for doy in range(1,365,8)]
    doy_57 = [doy for doy in range(57,365,8)]
    doy_113 = [doy for doy in range(113,365,8)]

    nr_msing_doy_indizes = doy_full[0:min(doy_57)]

    print("doy_full:\t", len(doy_full))
    print("doy_57:\t\t", len(doy_57))
    print("doy_113:\t", len(doy_113))
    print("doy_msing:\t", len(nr_msing_doy_indizes))

    user_year = 2004

    ts_raw_base_index = len(doy_57)           # this is the index where the year 2001 epoch starts in the data_lists for the bands
    ts_raw_end_index = ts_raw_base_index + len(doy_full)

    ts_fit_base_indes = len(doy_113)*
    ts_fit_end_index = ts_fit_base_indes + len(doy_full)

    # get into raw dir and select year001
    os.chdir(os.path.join(in_dir_tf, tile))
    raw_data_list_band1 = sorted(glob.glob("*.band_1.tif"))

    # get into  fit dir and select year001
    os.chdir(os.path.join(out_dir_fit,tile))


    for fit_product in fit_info_fft.keys():
        print("Fit Product: ", fit_product)
        fit_info_fft[fit_product]["files_list"] = sorted(glob.glob("*.band_1.sg_15_fft_poly.tif"))

    print("len band_1 ras list: ", len(raw_data_list_band1))
    print("Raster RAW 2001001: ", raw_data_list_band1[ts_raw_base_index])
    print("Raster Fit 2001001: ", fit_data_list_band1[ts_fit_base_indes] )

    print("Programm ENDE")