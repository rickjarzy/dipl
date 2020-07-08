from __future__ import print_function
import torch
import os
import glob
import time
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



            data_block, qual_block = init_data_block(sg_window, in_dir_qs, in_dir_tf, tile, list_qual, list_data, device)
            print("SHAPE OF DATA: ", data_block.shape)

            data_block_indizes = [[index for index in range(i, i+300, 1)] for i in range(0, 2400, 300)]
            data_block = torch.reshape(data_block, (sg_window, master_raster_info[2]*master_raster_info[3]))
            qual_block = torch.reshape(qual_block, (sg_window, master_raster_info[2]*master_raster_info[3]))

            A = torch.ones(sg_window, 3)
            torch.arange(1,15,1, out=A[:, 1])
            torch.arange(1,15,1, out=A[:, 2])
            A[:, 2] = A[:, 2]**2

            print("reshaped data block: ", data_block.shape)
            print("reshaped qual block: ", qual_block.shape)


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