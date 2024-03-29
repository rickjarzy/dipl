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

# todo: select a year by index - done
# todo: select different calculations from fit - done
# todo: select corresponding raw data for same time periode
# todo: create first plot with

import os
import glob
import time
import socket
import numpy
import copy

from osgeo import gdal, ogr
from matplotlib import pyplot as plt
from fit_information import fit_info_all, fit_info_poly, fit_info_fft, fit_info_best, doy_factors, fit_info_best_poly, fit_info_best_fou
from create_date_for_plots import get_dates_from_doy
from dataclasses import dataclass

@dataclass
class PlotDirInstance():
    figure_path : str
    figure_filename_post_fix : str
    figure_file_name_desc: str
    fitted_raster_band_name : str
    fit_year : str
    fit_doy : str
    band : str
    extension: str

def interp_raw_data(input_raw_data, input_qual_data):
    """This algo inertpolates linear on indizes that hold nan values

    Args:
        input_raw_data ([numpy array]): [raw data array, that holds NaN Values]
        input_qual_data ([numpy array]): [description]

    Returns:
        [numpy array]: [interpolated data array]
    """


    data_to_interp = numpy.copy(input_raw_data)
    data_to_interp = numpy.where(data_to_interp == 32767, numpy.nan, data_to_interp )
    print("\ninput_raw_data: ", input_raw_data)
    print("nan data: ", data_to_interp)
    data_mat_v_nan = numpy.isfinite(data_to_interp)
    data_mat_v_t = numpy.arange(0, len(data_mat_v_nan), 1)
    data_mat_v_interp = numpy.round(numpy.interp(data_mat_v_t, data_mat_v_t[data_mat_v_nan], data_to_interp[data_mat_v_nan]))
    data_to_interp = data_mat_v_interp

    print("Data inpert: ", data_to_interp)
    return data_to_interp
    

def convert_koords_to_indizes(coords, master_raster_info):

    shp_x = coords[0]
    shp_y = coords[1]

    # print("raster bands nr: ", ras.RasterCount)
    #print("ras_geo_ref: ", ras_geo_ref)

    ras_xorigin = master_raster_info[0][0]
    ras_yorigin = master_raster_info[0][3]
    ras_res = master_raster_info[0][1]
    ras_cols = 2400
    ras_rows = 2400

    ras_x_ul = ras_xorigin
    ras_y_ul = ras_yorigin

    #calcuate the satellite image extend
    ras_x_lr = ras_xorigin + ras_res * ras_cols
    ras_y_lr = ras_yorigin + master_raster_info[0][5] * ras_rows          # negative ras_resolution (0.0, 463.31271652750013, 0.0, 5559752.597460333, 0.0, -463.31271652750013)

    #print("Sat Extend UL: ",(ras_x_ul, ras_y_ul))
    #print("Sat Extend LR: ", (ras_x_lr, ras_y_lr))

    if shp_x >= ras_x_ul or shp_x <= ras_x_lr:
        if shp_y <= ras_y_ul and shp_y >= ras_y_lr:
            #print("Shp Coords in SatExtend")

            # calculate the pixel position in  the image matrix that was marked with the shape geometry
            ras_x_ind_BildMatrix = int(abs((abs(ras_xorigin) - abs(shp_x)) / ras_res))
            ras_y_ind_BildMatrix = int(abs((abs(ras_yorigin) - abs(shp_y)) / ras_res))

        else:
            print("Shp y Coordinate not ins SatSzene Extent")
            
    else:
        print("Shp x Coordinate not ins SatSzene Extent")
        

    return ras_x_ind_BildMatrix, ras_y_ind_BildMatrix


def read_out_modis_values(epochs_path_list, shp_koords_list, root_dir=False):
    """
    Extracts the rastervalue of the MODIS image according to its position. the x and y shape koordinates are taken an
    with the resolution and extend coordinates of the image, the corresponding row and cols for the underlying pixel value
    are calculated
    Parameters
    ----------
    epochs_path_list - list - with the epochs file names
    shp_koords_list  - list touple -  holding the x and y koords of the shape point
    root_dir         - string - if you need to fully reconstruct the path to the data

    Returns          - list - with all pixel data from the sat epochs from the epochs_path_list
    -------

    """

    print("\nREAD OUT SAT DATA\n====================")
   # print("shp koords: ", shp_koords_list)
    #print("Epoch to read out: ", epochs_path_list)

    shp_x = shp_koords_list[0]
    shp_y = shp_koords_list[1]

    return_ras_data_list = []

    for epoch in epochs_path_list:
        if root_dir:
            epoch_path = os.path.join(root_dir, epoch)
        else:
            epoch_path = epoch

        #print("Processing epoch: ", epoch_path)

        # read out the reference system infos
        ras = gdal.Open(epoch_path, gdal.GA_ReadOnly)
        ras_band = ras.GetRasterBand(1)                 # interesstingly, here one has to count starting from one, also if only one raster band is available
        ras_geo_ref = ras.GetGeoTransform()

        # print("raster bands nr: ", ras.RasterCount)
        #print("ras_geo_ref: ", ras_geo_ref)

        ras_xorigin = ras_geo_ref[0]
        ras_yorigin = ras_geo_ref[3]
        ras_res = ras_geo_ref[1]
        ras_cols = ras.RasterXSize
        ras_rows = ras.RasterYSize

        ras_x_ul = ras_xorigin
        ras_y_ul = ras_yorigin

        #calcuate the satellite image extend
        ras_x_lr = ras_xorigin + ras_res * ras_cols
        ras_y_lr = ras_yorigin + ras_geo_ref[5] * ras_rows          # negative ras_resolution (0.0, 463.31271652750013, 0.0, 5559752.597460333, 0.0, -463.31271652750013)

        #print("Sat Extend UL: ",(ras_x_ul, ras_y_ul))
        #print("Sat Extend LR: ", (ras_x_lr, ras_y_lr))

        if shp_x >= ras_x_ul or shp_x <= ras_x_lr:
            if shp_y <= ras_y_ul and shp_y >= ras_y_lr:
                #print("Shp Coords in SatExtend")

                # calculate the pixel position in  the image matrix that was marked with the shape geometry
                ras_x_ind_BildMatrix = int(abs((abs(ras_xorigin) - abs(shp_x)) / ras_res))
                ras_y_ind_BildMatrix = int(abs((abs(ras_yorigin) - abs(shp_y)) / ras_res))

                # reading out the sat data has a return like --> array([[370]], dtype=int16) so we have to access that nested array with indizes [0][0]
                ras_data = ras_band.ReadAsArray(ras_x_ind_BildMatrix, ras_y_ind_BildMatrix, 1, 1)[0][0]

                if ras_data <= 0:
                    ras_data = 0


            else:
                print("Shp y Coordinate not ins SatSzene Extent")
                ras_data = 0
        else:
            print("Shp x Coordinate not ins SatSzene Extent")
            ras_data = 0

        # appending a single value to the return list
        return_ras_data_list.append(ras_data)

    return return_ras_data_list


def read_out_shp_koord(shp_path):
    """
    Read out the X and Y Koords from each Point Shape Polygon from the handed Shape Path
    Parameters
    ----------
    shp_path - string - holds the path to the shape file

    Returns - dictionary - holding key value pairs for  "koords":[Xkoordinate of the shape, Ykoordinate of the shape]
                                                        "desc": Where is the shape point location - woods, see, beach etc
    -------

    """
    print("Extract X/Y Koords from Shape file")

    shp_poi_koords = []
    return_dict = {}
    shpDrv = ogr.GetDriverByName('ESRI Shapefile')
    dsPoi = shpDrv.Open(shp_path)
    lyr = dsPoi.GetLayer()
    numPoi = lyr.GetFeatureCount()
    print("number of points: ", numPoi)
    # iterate through Features
    for ft in range(0, numPoi, 1):
        actPoi = lyr.GetFeature(ft)
        shpPoiGeom = actPoi.GetGeometryRef()
        xPoi = shpPoiGeom.GetX()
        yPoi = shpPoiGeom.GetY()

        shp_poi_koords.append([xPoi, yPoi])
        return_dict[ft] = {"koords":[xPoi, yPoi], "desc":actPoi.GetField("Desc")}

    return return_dict


def plot_ts_with_shape(input_dict, input_shp_date_info_dict, input_band, input_year, fig_path, plotting_theme):

    x_axe_data = numpy.array(range(0,46,1))

    print("Start plotting ...")

    for shape_id in input_dict.keys():

        plotted_year = input_dict[shape_id]["year"]

        print("Create Plot for %d and  shape_desc %s: "%(plotted_year, input_dict[shape_id]["desc"]))
        print("ID Desc : ", input_dict[shape_id].keys())

        location_desc = input_dict[shape_id]["desc"]
        fitting_method = input_dict[shape_id]["desc"]
        print("Fit Products: ", input_dict[shape_id]["fit_products"].keys())

        fig, ax = plt.subplots()
        plt.rcParams["figure.figsize"] = (15, 9)



        
        # iterate through the fit products and create for each shape id a plot and show it

        best_qual = numpy.array(input_dict[shape_id]["quality_%d"%shape_id])
        qual_factor = 100
        # turn the coding around so 0 is best quality but it should be higher in the plot so 0 turns 3
        good_qual = numpy.where(best_qual == 0, 3, numpy.nan) * qual_factor
        okay_qual = numpy.where(best_qual == 1, 2, numpy.nan) * qual_factor
        bad_qual = numpy.where(best_qual == 2, 1, numpy.nan) * qual_factor
        really_bad_qual = numpy.where(best_qual == 3, 0.5, numpy.nan) * qual_factor
        fill_value = numpy.where(best_qual == 4, 0.1, numpy.nan) * qual_factor
        nan_values = numpy.where(best_qual == 255,  0, numpy.nan)



        for fit_product in input_dict[shape_id]["fit_products"].keys():


            print("processing fit product: ", fit_product)

            data_array = input_dict[shape_id]["fit_products"][fit_product]["fit_data_%d"%shape_id]
            print("data: ", data_array)
            ax.plot(x_axe_data,data_array, label=input_dict[shape_id]["fit_products"][fit_product]["label"])


        ax.set_title("Result Comparison Different Weights %s - %s - Year %d - Band %s" % (input_dict[shape_id]["fit_products"][fit_product]["fit_method_name"], location_desc, input_year, input_band.split("_")[1]))
        
        #ax.set_title("Fitting Method Comparison - %s - Year %d - Band %s" % (location_desc, input_year, input_band.split("_")[1]))

        print("raw_data: ", input_dict[shape_id]["raw_data_%d"%shape_id])
        ax.plot(x_axe_data, input_dict[shape_id]["raw_data_%d"%shape_id], color='c', LineWidth=0, marker="*",markersize=15, label="raw data")
        ax.plot(x_axe_data, input_dict[shape_id]["raw_data_interp_%d"%shape_id], color='k', LineWidth=1, linestyle='--', label='linear interpolated raw data')
        ax.plot(x_axe_data, good_qual, 'go', label="Best Quality")
        ax.plot(x_axe_data, okay_qual, 'yo', label="Good Quality ")
        ax.plot(x_axe_data, bad_qual, 'o', color='orange', label="Magnitude Inversion obs >= 7")
        ax.plot(x_axe_data, really_bad_qual, 'ro', label="Magnitude Inversion obs 2>= X < 7")
        ax.plot(x_axe_data, fill_value, 'bo', label="Fill Value")
        ax.plot(x_axe_data, nan_values, 'ko', label="NaN Values")

        ax.set_xlabel("Day Of Year From Year %s" % plotted_year)
        ax.set_ylabel("Reflexion [%]")
        ax.set_xticks(list(range(0,46,1)))
        #ax.set_xticklabels([str(i) for i in range(1,365,8)])
        ax.set_xticklabels(input_shp_date_info_dict["plot_dates"]["dates"], rotation=45)
        
        if input_band == "band_1":
            y_limits = [0, 5000]
        elif input_band == "band_2":
            y_limits = [0, 6000]

        ax.set_ylim(y_limits)
        plt.legend(loc ="upper right")
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.subplots_adjust(left=0.059, right=0.715, top=0.96, bottom=0.123)
        #plt.show()

        plt.savefig(os.path.join(fig_path, plotting_theme +"_" + location_desc+ "_%d"%input_year), dpi=150)
        del fig, ax



def main():

    print("Start Processing Plots for yearwise timeseries")
    # 2T USB3 Festplatte und Home Rechner
    if socket.gethostname() in ['XH-AT-NB-108', 'Paul-PC', 'DESKTOP-8K4K22S']:
        in_dir_qs = r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
        in_dir_tf = r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
        out_dir_fit = r"E:\MODIS_Data\v6\fitted"

    else:
        print("Check Data Drive Letter !!!!!")

    if os.name == "posix" and socket.gethostname() == "paul-buero":

        in_dir_qs =   r"/media/paul/Daten_Diplomarbeit2/MODIS_Data/v6/tiff_single/MCD43A2"
        in_dir_tf =   r"/media/paul/Daten_Diplomarbeit2/MODIS_Data/v6/tiff_single/MCD43A4"
        out_dir_fit = r"/media/paul/Daten_Diplomarbeit2/MODIS_Data/v6/fitted"

    elif os.name == "posix" and socket.gethostname() in ["iotmaster", "iotslave1", "iotslave2"]:
        in_dir_qs =   r"/home/iot/scripts/dev/projects/timeseries/data/v6/tiff_single/MCD43A2"
        in_dir_tf =   r"/home/iot/scripts/dev/projects/timeseries/data/v6/tiff_single/MCD43A4"
        out_dir_fit = r"/home/iot/scripts/dev/projects/timeseries/data/v6/fitted"

    elif os.name == "nt" and socket.gethostname() == "DESKTOP-8K4K22S":
        in_dir_qs = r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
        in_dir_tf = r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
        out_dir_fit = r"E:\MODIS_Data\v6\fitted"
        shp_dir     = r"E:\MODIS_Data\shp\checkFitPlots"
        figure_path = r"E:\Diplomarbeit\Schriftlich\Diplomarbeit\LatexFiles\Versuch2\Versuch1\Grafiken\Fitting\Fitting_method_comparison\selected_comparisons_poly"

    elif os.name == "nt" and socket.gethostname() == "TX64-01540":
        in_dir_qs = r"D:\modis\v6\tiff_single\MCD43A2"
        in_dir_tf = r"D:\modis\v6\tiff_single\MCD43A4"
        out_dir_fit = r"D:\modis\v6\fitted"
        shp_dir     = r"C:\Users\parzberg\Documents\private\dev\python\dipl\shp\checkFitPlots"
        figure_path = r"C:\Users\parzberg\Documents\private\dev\latex\dipl\Grafiken\Fitting\Fitting_method_comparison"
    else:
        in_dir_qs =   r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
        in_dir_tf =   r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
        out_dir_fit = r"E:\MODIS_Data\v6\fitted"
        shp_dir =     r"E:\MODIS_Data\shp\checkFitPlots"
        figure_path = r"E:\Diplomarbeit\Schriftlich\Diplomarbeit\LatexFiles\Versuch2\Versuch1\Grafiken\Fitting\Fitting_method_comparison"
        figure_filename_post_fix = r"comparison_reflectance_values_for_"

    tile = "h18v04"

    shps = os.chdir(shp_dir)

    shp_file = glob.glob("checkFitProducts1804.shp")[0]
    shp_info = read_out_shp_koord(os.path.join(shp_dir,shp_file))

    print("Found ShapeFile: ", shp_file)
    print("SHP Info Dict: ", shp_info)

    sat_raw_indir = in_dir_tf
    sat_qual_indir = in_dir_qs
    sat_fitted_indir = out_dir_fit

    doy_full = [doy for doy in range(1, 365, 8)]
    doy_57 = [doy for doy in range(57, 365, 8)]
    doy_113 = [doy for doy in range(113, 365, 8)]

    nr_msing_doy_indizes = doy_full[0:min(doy_57)]

    print("doy_full:\t", len(doy_full))
    print("doy_57:\t\t", len(doy_57))
    print("doy_113:\t", len(doy_113))
    print("doy_msing:\t", len(nr_msing_doy_indizes))

    user_band = "band_2"

    user_year = 2002

    # calcute the starting point of the year defined by user_year
    ts_raw_base_index = len(doy_57) + len(doy_full) * doy_factors[user_year]["factor"]  # this is the index where the year 2001 epoch starts in the data_lists for the bands
    print("Raw based index: ", ts_raw_base_index)
    
    # calculate the end point of the year defined by the user_year
    ts_raw_end_index = ts_raw_base_index + len(doy_full)

    # get into raw dir and select year001
    os.chdir(os.path.join(in_dir_tf, tile))
    raw_data_list = sorted(glob.glob("*.%s.tif"%user_band))
    print(os.getcwd())
    print("Raw data list: ", len(raw_data_list))
    # get into qual dir and select year001
    os.chdir(os.path.join(in_dir_qs, tile))
    qual_data_list = sorted(glob.glob("*.%s.tif"%user_band))

    # get into  fit dir and select year001
    os.chdir(os.path.join(out_dir_fit, tile))

    print("Startung point raw raster epoch: ", raw_data_list[ts_raw_base_index])
    print("Ending point raw raster epoch:   ", raw_data_list[ts_raw_end_index])

    print(raw_data_list[ts_raw_base_index:ts_raw_end_index])

    # store the specific year data onto the fit_info_XXX Dict. It holds on the key files_list a list with tif epochs for the fit product of the
    # defined year . eg. 2005 --> 2005001 - 2005361

    #used_fit_info_dict = fit_info_all
    #used_fit_info_dict = fit_info_fft
    #used_fit_info_dict = fit_info_poly
    #used_fit_info_dict = fit_info_best
    used_fit_info_dict = fit_info_best_fou
    #used_fit_info_dict = fit_info_best_poly

    print("SHP_INFO: ", shp_info)
    shp_date_info = get_dates_from_doy(raw_data_list[ts_raw_base_index:ts_raw_end_index], {})
    print("after: ", shp_info)
    
    # write the epochs file names onto the dict
    for fit_product in used_fit_info_dict.keys():
        print("Fit Product: ", fit_product)
        print("""fit_product == "fft" or "dft.elements_3.1.00_0.50_0.25_0.01" """, fit_product == "fft" or "dft.elements_3.1.00_0.50_0.25_0.01")
        used_fit_info_dict[fit_product]["root_dir"] = os.path.join(out_dir_fit, tile)

        # fft files start with year 2001-001!!!!!
        if fit_product == "fft":
            ts_fit_base_index = len(doy_full) * doy_factors[user_year]["factor"]
            ts_fit_end_index = ts_fit_base_index + len(doy_full)

        elif fit_product == "dft.elements_3.1.00_0.50_0.25_0.01":
            ts_fit_base_index = len(doy_full) * doy_factors[user_year]["factor"]
            ts_fit_end_index = ts_fit_base_index + len(doy_full)

        elif fit_product == "dft.elements_3.1.00_0.01_0.01_0.01":
            ts_fit_base_index = len(doy_full) * doy_factors[user_year]["factor"]
            ts_fit_end_index = ts_fit_base_index + len(doy_full)
        else:
            ts_fit_base_index = len(doy_113) + len(doy_full) * doy_factors[user_year]["factor"]
            ts_fit_end_index = ts_fit_base_index + len(doy_full)

        print("ts_raw_base_index: ", ts_raw_base_index)
        print("ts_raw_end_index : ", ts_raw_end_index)
        print("ts_fit_base_index: ", ts_fit_base_index)
        print("ts_fit_end_index:  ", ts_fit_end_index)
        print()

        full_ts = sorted(glob.glob("*.%s.%s.tif" % (user_band, fit_product)))

        used_fit_info_dict[fit_product]["files_list"] = sorted(glob.glob("*.%s.%s.tif" % (user_band, fit_product)))[
                                                        ts_fit_base_index:ts_fit_end_index]
        print("search for: ", "*.%s.%s.tif" % (user_band, fit_product))
        print("Starting point fill TS: ", full_ts[0])
        print("Starting point fit raster epoch: ", used_fit_info_dict[fit_product]["files_list"][0])
        print("Ending point fit raster epoch : ", used_fit_info_dict[fit_product]["files_list"][-1])

    cou = 0
    # read out the satellite data on the specific shape koordinates and attach it to the dict
    for shp_index in shp_info.keys():
        print("\n### Processing SHP ID: ", shp_index)
        shp_info[shp_index].update({"fit_products":used_fit_info_dict})
        koords_to_process = shp_info[shp_index]["koords"]

        shp_info[shp_index].update({"year": user_year})

        raw_data = read_out_modis_values(raw_data_list[ts_raw_base_index:ts_raw_end_index],
                                                   koords_to_process,
                                                   root_dir=os.path.join(in_dir_tf, tile))


        qual_data = read_out_modis_values(qual_data_list[ts_raw_base_index:ts_raw_end_index],
                                                   koords_to_process,
                                                   root_dir=os.path.join(in_dir_qs, tile))

        raw_data_interp = interp_raw_data(raw_data, qual_data)

        shp_info[shp_index].update({"raw_data_%d"%shp_index: raw_data})

        shp_info[shp_index].update({"raw_data_interp_%d"%shp_index: raw_data_interp})

        shp_info[shp_index].update({"quality_%d"%shp_index: qual_data})

        for fit_product in shp_info[shp_index]["fit_products"].keys():

            print("koords: ", koords_to_process)
            print("")
            print(shp_info[shp_index]["fit_products"][fit_product].keys())

            data_extracted = read_out_modis_values(shp_info[shp_index]["fit_products"][fit_product]["files_list"],
                                                   koords_to_process,
                                                   root_dir=shp_info[shp_index]["fit_products"][fit_product]["root_dir"])
            print(data_extracted)

            shp_info[shp_index]["fit_products"][fit_product].update({"fit_data_%d"%shp_index: data_extracted})

            print(shp_info[shp_index]["fit_products"][fit_product].keys())

            print(shp_info[shp_index]["fit_products"][fit_product]["fit_data_%d"%shp_index])


    print("\n\n")
    # todo: raw data an das shp_info anhängen

    for ind in shp_info.keys():
        print()
        for indi in shp_info[ind]["fit_products"].keys():

            print("data: ", shp_info[ind]["fit_products"][indi]["fit_data_%d"%ind])

    # create a plot for the raster data TS

    plotting_theme = "Fit_comparison_fft_all"
    plot_ts_with_shape(shp_info, shp_date_info, user_band, user_year, figure_path, plotting_theme)



if __name__ == "__main__":


    main()



    print("Programm ENDE")