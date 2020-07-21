from __future__ import print_function
import torch
import os
import glob
import time
import numpy
from osgeo import gdalconst
#import multiprocessing
from multi_fit_single_utils import *





if __name__ == "__main__":

    start = time.time()

    if torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA is available")
    else:
        device = torch.device("cpu")
        print("CPU is available")


    # # 4TB USB Festplatte
    # in_dir_qs = r"F:\modis\v6\tiff_single\MCD43A2"
    # in_dir_tf = r"F:\modis\v6\tiff_single\MCD43A4"

    # 2T USB3 Festplatte und Home Rechner
    in_dir_qs = r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
    in_dir_tf = r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
    out_dir_fit = r"E:\MODIS_Data\v6\fitted"

    if os.name == "posix":
        in_dir_qs = r"/media/paul/Daten_Diplomarbeit/MODIS_Data/v6/tiff_single/MCD43A2"
        in_dir_tf = r"/media/paul/Daten_Diplomarbeit/MODIS_Data/v6/tiff_single/MCD43A4"
        out_dir_fit = r"/media/paul/Daten_Diplomarbeit/MODIS_Data/v6"



    # kacheln = ["h18v04", "h18v03", "h19v03", "h19v04"]
    tile = "h18v04"
    bands = list(range(1,8,1))
    print(bands)
    sg_window = 15
    window_arr = torch.arange(0,sg_window,1)       # range from 0 to sg_window
    fit_nr = torch.median(window_arr)               # just a tensor fomr 0 to sg_window
    center = torch.median(window_arr)               # center index of the data stack
    sigm = torch.ones(sg_window, 2400*2400)


    list_of_bands_to_process = []

    master_raster_info = get_master_raster_info(in_dir_tf, tile, "MCD43A4")

    for b in bands:
        os.chdir(os.path.join(in_dir_qs, tile))
        list_qual = sorted(glob.glob("MCD43A2.*.band_%d.tif" % b))

        os.chdir(os.path.join(in_dir_tf, tile))
        list_data = sorted(glob.glob("MCD43A4.*.band_%d.tif" % b))

        list_of_bands_to_process.append({"band":b,
                                         "in_dir_qs": in_dir_qs,
                                         "in_dir_tf": in_dir_tf,
                                         "sg_window": sg_window,
                                         "out_dir": out_dir_fit,
                                         "device":device,
                                         "tile": tile})

        if int(len(list_qual)) != int(len(list_data)):
            print("\nBand %s cannot be processed!\n" % b)
            # print("len data %d != %d qual: \n" % ( len(list_data), len(list_qual)))

        else:

            # Initialize data fiting -load satellite data into data blocks
            # ============================================================

            data_block, qual_block = init_data_block(sg_window, b, in_dir_qs, in_dir_tf, tile, list_qual, list_data, device, master_raster_info)
            print("SHAPE OF DATA: ", data_block.shape)

            data_block_indizes = [[index for index in range(i, i+300, 1)] for i in range(0, 2400, 300)]
            data_block = torch.reshape(data_block, (sg_window, master_raster_info[2]*master_raster_info[3]))
            print("data values: ", qual_block[:, 0, -1])
            qual_block = torch.reshape(qual_block, (sg_window, master_raster_info[2]*master_raster_info[3]))

            qual_block[qual_block==0] = 1
            qual_block[qual_block==1] = 0.75
            qual_block[qual_block==2] = 0.25
            qual_block[qual_block==3] = 0.1

            qual_block[qual_block==255] = 0         # set to 0 so in the ausgleich the nan -> zero convertion is not needed
                                                    # nan will be replaced by zeros so this is a shortcut to avoid that transformation
            data_block[data_block==32767] = 0

            A = torch.ones(sg_window, 3).to(device=device)
            torch.arange(1, sg_window + 1, 1, out=A[:, 1])
            torch.arange(1, sg_window + 1, 1, out=A[:, 2])
            A[:, 2] = A[:, 2]**2

            print("reshaped data block: ", data_block.shape)
            print("reshaped qual block: ", qual_block.shape)

            print("A: \n", A)
            print("QM: ", qual_block[:,0])

            #todo: A-->xv, P --> pv, data_block --> lv in form bringen dass in die ausgleichsfunktion reinpasst

            print("Start fitting ...")
            [a0, a1, a2] = fitq_cuda(data_block, qual_block, A[:,1], sg_window)

            print("len a0: ", a0.shape)
            print("len a1: ", a1.shape)
            print("len a2: ", a2.shape)

            # ------------------------------------------
            # Logic for empty or not enought data spots
            # ymin, ymax etc
            # ------------------------------------------
            print("calc new raster matrix ... ")

            # fit the data
            # A.shape = [15,1]
            # a0.shape = [57600000]
            fit = a0 + a1 * torch.reshape(A[:,1], (sg_window, 1)) + a2 * torch.reshape(A[:,2], (sg_window, 1))      # fit.shape: [15,5_760_000]

            # calc new weights
            delta_lv = torch.abs(fit - data_block)          # delta_lv.shape: [15,5_760_000]
            delta_lv[delta_lv<1] = 1                        #
            sig = torch.sum(delta_lv,0)                     # sig.shape: [5_760_000]
            sigm = sigm * sig                               # sigm.shape: [15, 5_760_000]

            qual_updated = sigm/delta_lv                    # qual_updated.shape: [15, 5_760_000]

            print("delta_lv: ", delta_lv.shape)
            print("calc new raster matrix - 2nd iteration ...")
            [a0, a1, a2] = fitq_cuda(data_block, qual_updated, A[:, 1], sg_window)

            fit = a0 + a1 * torch.reshape(A[:,1], (sg_window, 1)) + a2 * torch.reshape(A[:,2], (sg_window, 1))
            fit_layer = torch.reshape(fit[fit_nr], (2400,2400)).numpy()
            # write output raster
            print("Fitlayer stats: ", fit_layer.shape)
            out_ras_name = os.path.join(out_dir_fit, "firstfit.tif")
            print("outdir: ", out_ras_name)
            out_ras = master_raster_info[-1].Create(out_ras_name, 2400, 2400, 1, gdalconst.GDT_Int16)
            out_band = out_ras.GetRasterBand(1)
            out_band.WriteArray(fit_layer)
            out_band.SetNoDataValue(32767)
            out_ras.SetGeoTransform(master_raster_info[0])
            out_ras.SetProjection(master_raster_info[1])
            out_ras.FlushCache()
            del out_ras

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