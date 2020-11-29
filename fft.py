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
from utils_numpy import init_data_block_fft, perfom_fft,  update_data_block_numpy, write_fitted_raster_to_disk
import fit_config


if __name__ == "__main__":
    try:
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
            out_dir_fit = r"/media/paul/Daten_Diplomarbeit/MODIS_Data/v6/fitted"

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
        bands = fit_config.bands              #list(range(1,8,1))
        print(bands)

        sg_window = 92
        window_arr = numpy.arange(0,sg_window,1)       # range from 0 to sg_window
        fit_nr = int(numpy.median(window_arr))               # just a tensor fomr 0 to sg_window
        center = int(numpy.median(window_arr))               # center index of the data stack
        sigm = numpy.ones((sg_window, 2400,2400))
        half_window = int(numpy.floor(sg_window/2))

        A = numpy.ones([sg_window, 3])
        A[:, 1] = numpy.arange(1, sg_window + 1, 1)
        A[:, 2] = numpy.arange(1, sg_window + 1, 1)
        A[:, 2] = A[:, 2] ** 2

        weights = [1, 2, 3, 3]

        name_weights_addition = ".fft.%s.tif"
        calc_from_to = [177, 263]

        master_raster_info = get_master_raster_info(in_dir_tf, tile, "MCD43A4")

        for b in bands:
            os.chdir(os.path.join(in_dir_qs, tile))
            list_qual = sorted(glob.glob("MCD43A2.*.band_%d.tif" % b))[calc_from_to[0]:]

            os.chdir(os.path.join(in_dir_tf, tile))
            list_data = sorted(glob.glob("MCD43A4.*.band_%d.tif" % b))[calc_from_to[0]:]
            len_list_data = len(list_data)
            # check if qual and sat data have the same amount of files
            if int(len(list_qual)) != int(len(list_data)):
                print("Len list_qual: ", len(list_qual))
                print("Len list_data: ", len(list_data))
                print("\nBand %s cannot be processed!\n" % b)
                # print("len data %d != %d qual: \n" % ( len(list_data), len(list_qual)))

            else:

                for ts in range(calc_from_to[0],len_list_data,1):
                    epoch_start = time.time()
                    ref_ras_epoch = list(range(calc_from_to[0], len_list_data, 1))
                    if ts == ref_ras_epoch[0]:
                        #try:
                        # Initialize data fiting -load satellite data into data blocks
                        # ============================================================

                        #data_block, qual_block, fitted_raster_band_name = init_data_block_numpy(sg_window, b, in_dir_qs, in_dir_tf, tile, list_qual, list_data, device, master_raster_info, fit_nr, name_weights_addition)

                        data_block, fitted_raster_band_name = init_data_block_fft(sg_window, b, in_dir_tf, tile, list_data, master_raster_info, fit_nr, name_weights_addition)

                        #todo: überlegen ob man nicht für links und rechtsseitig der zentralen bildmatrix einen linearen fit machen will wenn zu wenige daten sind
                        #todo: fit aus check für cuda und numpy implementieren dann geht die sache in produktion


                        print("\nStart fitting %s - Nr %d out of %d \n-------------------------------------------" % (fitted_raster_band_name, ts+1, len_list_data))
                        print("DATABLOCK: \n", data_block[:,2200,1000])
                        print("SHape Datablock: ", data_block.shape)
                        # FFT Logic Here


                        perfom_fft(data_block[:,0,0])


                        break

                        # write output raster
                        #write_fitted_raster_to_disk(fit_layer, out_dir_fit, tile, fitted_raster_band_name, master_raster_info)

                        print("- FINISHED Fit after ", time.time() - epoch_start, " [sec]\n")
                        # except Exception as BrokenFirstIteration:
                        #     print("### ERROR - Something went wrong in the first iteration \n  - {}".format(BrokenFirstIteration))
                        #     break
                        # except KeyboardInterrupt:
                        #     print("### PROGRAMM ENDED BY USER")
                        #     break
                    elif ts == calc_from_to[1]:
                        break

                    else:
                        try:
                            # update data and qual information
                            data_block, qual_block, noup_array, fitted_raster_band_name, iv, l_max, l_min = update_data_block_numpy(data_block, qual_block, noup_array, in_dir_tf, in_dir_qs, tile, list_data, list_qual, sg_window, center, half_window, fit_nr, ts, weights, name_weights_addition)

                            #A, data_block, qual_block, iv, l_max, l_min = additional_stat_info_raster_numpy(data_block, qual_block, sg_window, device, half_window, center)

                            print("\nStart fitting %s - Nr %d out of %d \n-------------------------------------------" % (
                            fitted_raster_band_name, ts + 1, len_list_data))
                            print("DATABLOCK: \n", data_block[:, 0, 0])


                            ## FFT Logic Here

                            # write output raster
                            # write_fitted_raster_to_disk(fit_layer, out_dir_fit, tile, fitted_raster_band_name, master_raster_info)

                            print("- FINISHED Fit after ", time.time() - epoch_start, " [sec]\n")
                        except Exception as BrokenFurtherIteration:
                            print("### ERROR - Something went wrong in the following iterations \n  - {}".format(BrokenFurtherIteration))
                            break
                        # except KeyboardInterrupt:
                        #     print("### PROGRAMM ENDED BY USER")
                        #     break


        print("elapsed time: ", time.time() - start , " [sec]")
        print("Programm ENDE")
    except KeyboardInterrupt:
        print("### PROGRAMM ENDED BY USER")
