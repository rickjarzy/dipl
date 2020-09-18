from __future__ import print_function
import torch
import os
import glob
import time
import socket
import numpy
from osgeo import gdalconst
#import multiprocessing
from multi_fit_single_utils import *
from utils_numpy import additional_stat_info_raster_numpy, init_data_block_numpy, fitq_numpy, fitq



if __name__ == "__main__":

    start = time.time()

    if torch.cuda.is_available():
        #device = torch.device("cuda")
        #print("CUDA is available")
        device = torch.device("cpu")
        print("CUDA is available - but still using cpu due not enought GPU Mem")
    else:
        device = torch.device("cpu")
        print("CPU is available")


    # # 4TB USB Festplatte
    # in_dir_qs = r"F:\modis\v6\tiff_single\MCD43A2"
    # in_dir_tf = r"F:\modis\v6\tiff_single\MCD43A4"

    # 2T USB3 Festplatte und Home Rechner
    if socket.gethostname() in ['XH-AT-NB-108', 'Paul-PC']:
        in_dir_qs = r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
        in_dir_tf = r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
        out_dir_fit = r"E:\MODIS_Data\v6\fitted"

    else:
        print("Check Data Drive Letter !!!!!")

    if os.name == "posix" and socket.gethostname() == "paul-buero":

        in_dir_qs = r"/media/paul/Daten_Diplomarbeit/MODIS_Data/v6/tiff_single/MCD43A2"
        in_dir_tf = r"/media/paul/Daten_Diplomarbeit/MODIS_Data/v6/tiff_single/MCD43A4"
        out_dir_fit = r"/media/paul/Daten_Diplomarbeit/MODIS_Data/v6"

    elif os.name == "posix" and socket.gethostname() in ["iotmaster", "iotslave1", "iotslave2"]:
        in_dir_qs = r"/home/iot/scripts/dev/projects/timeseries/data/v6/tiff_single/MCD43A2"
        in_dir_tf = r"/home/iot/scripts/dev/projects/timeseries/data/v6/tiff_single/MCD43A4"
        out_dir_fit = r"/home/iot/scripts/dev/projects/timeseries/data/v6/fitted"
    else:
        in_dir_qs = r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
        in_dir_tf = r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
        out_dir_fit = r"E:\MODIS_Data\v6\fitted"


    # kacheln = ["h18v04", "h18v03", "h19v03", "h19v04"]
    tile = "h18v04"
    bands = list(range(1,8,1))
    print(bands)
    sg_window = 15
    window_arr = numpy.arange(0,sg_window,1)       # range from 0 to sg_window
    fit_nr = int(numpy.median(window_arr))               # just a tensor fomr 0 to sg_window
    center = int(numpy.median(window_arr))               # center index of the data stack
    sigm = numpy.ones((sg_window, 2400,2400))
    half_window = int(numpy.floor(sg_window/2))



    master_raster_info = get_master_raster_info(in_dir_tf, tile, "MCD43A4")

    for b in bands:
        os.chdir(os.path.join(in_dir_qs, tile))
        list_qual = sorted(glob.glob("MCD43A2.*.band_%d.tif" % b))

        os.chdir(os.path.join(in_dir_tf, tile))
        list_data = sorted(glob.glob("MCD43A4.*.band_%d.tif" % b))


        if int(len(list_qual)) != int(len(list_data)):
            print("Len list_qual: ", len(list_qual))
            print("Len list_data: ", len(list_data))
            print("\nBand %s cannot be processed!\n" % b)
            # print("len data %d != %d qual: \n" % ( len(list_data), len(list_qual)))

        else:

            # Initialize data fiting -load satellite data into data blocks
            # ============================================================

            data_block, qual_block = init_data_block_numpy(sg_window, b, in_dir_qs, in_dir_tf, tile, list_qual, list_data, device, master_raster_info)


            data_block_indizes = [[index for index in range(i, i+300, 1)] for i in range(0, 2400, 300)]

            A, data_block, qual_block, noup_c, noup_r, noup_l, iv = additional_stat_info_raster_numpy(data_block, qual_block, sg_window, device, half_window, center)

            print("iv.shape: ", iv.shape)
            print(iv[:200])

            #todo: überlegen ob man nicht für links und rechtsseitig der zentralen bildmatrix einen linearen fit machen will wenn zu wenige daten sind
            #todo: fit aus check für cuda und numpy implementieren dann geht die sache in produktion


            print("Start fitting ...")
            #[a0, a1, a2] = fitq_numpy(data_block, qual_block, A, sg_window)
            [fit, sig]= fitq(data_block, qual_block, A, sg_window)
            print("fit.shape: ", fit.shape)

            delta_lv = abs(fit - data_block)
            delta_lv = numpy.where(delta_lv<1, 1, delta_lv)
            print("delta_lv.shape: ", delta_lv.shape)
            sigm = sigm * sig

            qual_block_nu = sigm/delta_lv
            print("Type qual_block: ", type(qual_block_nu))
            [fit, sig] = fitq(fit, qual_block_nu, A, sg_window)

            #
            # # filtered epoch
            # fit_layer = torch.reshape(fit[fit_nr], (2400,2400)).numpy()
            # # write output raster
            # print("Fitlayer stats: ", fit_layer.shape)
            # out_ras_name = os.path.join(out_dir_fit, "firstfit.tif")
            # print("outdir: ", out_ras_name)
            # out_ras = master_raster_info[-1].Create(out_ras_name, 2400, 2400, 1, gdalconst.GDT_Int16)
            # out_band = out_ras.GetRasterBand(1)
            # out_band.WriteArray(fit_layer)
            # out_band.SetNoDataValue(32767)
            # out_ras.SetGeoTransform(master_raster_info[0])
            # out_ras.SetProjection(master_raster_info[1])
            # out_ras.FlushCache()
            # del out_ras

            break
            ##test = [data_block[:, i, :] for i in data_block_indizes]

            #return_test_liste = multi_fit_gpu(data_block, qual_block, data_block_indizes)

            # END of Initializing
            # ===============================================================

            # START FITTING
            # ===============================================================

            # numbers_of_data_epochs = len(list_data)
            #
            # for i in range(0,len(list_qual),1):
            #     window_end_index = i + sg_window
            #     if window_end_index < numbers_of_data_epochs:
            #
            #         names_of_interrest_window = list_data[i:i+sg_window]
            #         print("Process tile {} index from to: {} - {}".format(tile, i, window_end_index))
            #
            #
            #
            #
            #
            #     else:
            #         print("\nreached end of processing in tile {} - index {} - {}".format(tile, i, window_end_index))
            #         break


    print("elapsed time: ", time.time() - start , " [sec]")
    print("Programm ENDE")