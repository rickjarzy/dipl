from __future__ import print_function
import torch
import os
import glob
import time
import socket
import numpy
from osgeo import gdalconst
import multiprocessing
from multi_fit_single_utils import *
from utils_numpy import (fitq, write_fitted_raster_to_disk, plot_raw_data)
from utils_mp import (init_data_block_mp, additional_stat_info_raster_mp,update_data_block_mp, multi_linear_interpolation)
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
            os.environ['PROJ_LIB'] = r"C:\Users\ArzbergerP\.conda\envs\py38\Library\share\proj"
            os.environ['GDAL_DATA'] = r"C:\Users\ArzbergerP\.conda\envs\py38\Library\share\gdal"

            in_dir_qs = r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
            in_dir_tf = r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
            out_dir_fit = r"E:\MODIS_Data\v6\fitted"


        # kacheln = ["h18v04", "h18v03", "h19v03", "h19v04"]
        tile = "h18v04"
        bands = fit_config.bands              #list(range(1,8,1))
        print(bands)
        sg_window = 15
        window_arr = numpy.arange(0,sg_window,1)       # range from 0 to sg_window
        fit_nr = int(numpy.median(window_arr))               # just a tensor fomr 0 to sg_window
        center = int(numpy.median(window_arr))               # center index of the data stack
        sigm = numpy.ones((sg_window, 2400,2400))
        half_window = int(numpy.floor(sg_window/2))

        A = numpy.ones([sg_window, 3])
        A[:, 1] = numpy.arange(1, sg_window + 1, 1)
        A[:, 2] = numpy.arange(1, sg_window + 1, 1)
        A[:, 2] = A[:, 2] ** 2

        weights = [1, 0.5, 0.01, 0.01]

        name_weights_addition = ".lin_poly_win%s.weights.{}_{}_{}_{}.tif".format(weights[0], weights[1], weights[2], weights[3])
        calc_from_to = [0, 263]

        master_raster_info = get_master_raster_info(in_dir_tf, tile, "MCD43A4")

        # multiprocessing constants
        num_of_pyhsical_cores = 4 - 1
        number_of_rows_data_part = master_raster_info[2] // num_of_pyhsical_cores
        num_of_buf_bytes = sg_window * master_raster_info[2] * master_raster_info[3] * 8


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

                for ts_epoch in range(calc_from_to[0], len_list_data, 1):
                    epoch_start = time.time()
                    ref_ras_epoch = list(range(calc_from_to[0], len_list_data, 1))

                    if ts_epoch == ref_ras_epoch[0]:
                        #try:
                        # Initialize data fiting -load satellite data into data blocks
                        # ============================================================


                        data_block, qual_block, fitted_raster_band_name, shm = init_data_block_mp(sg_window, b, in_dir_qs, in_dir_tf, tile, list_qual, list_data, num_of_buf_bytes, master_raster_info, fit_nr, name_weights_addition)

                        qual_block = additional_stat_info_raster_mp(qual_block, weights)


                        #todo: überlegen ob man nicht für links und rechtsseitig der zentralen bildmatrix einen linearen fit machen will wenn zu wenige daten sind
                        #todo: fit aus check für cuda und numpy implementieren dann geht die sache in produktion

                        print("\nStart fitting %s - Nr %d out of %d \n-------------------------------------------" % (fitted_raster_band_name, ts_epoch+1, len_list_data))

                        plot_indizess = [800,800]
                        start_interpl = time.time()
                        print("time start interpolation: ")
                        print("finished interpl: ", time.time() - start_interpl , " [sec]")
                        print("datablock type: ", type(data_block))

                        print(data_block[:,plot_indizess[0], plot_indizess[1]])

                        # interpolate linear on nan values
                        # - keep in mind - shit data will allways stay shit data


                        job_list_with_data_inidzes = []             # for mp pool
                        cou = 0
                        start_interp_time = time.time()

                        # create sections that should run in parallel
                        for part in range(0, master_raster_info[2], number_of_rows_data_part):
                            print(part)
                            info_dict = {"from": part, "to": part + number_of_rows_data_part, "shm": shm, "process_nr": cou,
                                         "dim":(sg_window, master_raster_info[2], master_raster_info[3]), "num_of_bytes": num_of_buf_bytes}
                            job_list_with_data_inidzes.append(info_dict)
                            cou += 1


                        multi_linear_interpolation(job_list_with_data_inidzes)
                        print("finished interpolation in ", time.time() - start_interp_time, " [sec] ")

                        # END linear interpolation
                        [fit, sig, delta_lv] = fitq(data_block, qual_block, A, sg_window)

                        print("- fit.shape: ", fit.shape)
                        print("- fit data: \n", fit[:,plot_indizess[0], plot_indizess[1]])

                        print("- delta_lv.shape: ", delta_lv.shape)

                        # plot_raw_data(data_block[:, plot_indizess[0], plot_indizess[1]],
                        #               qual_block[:, plot_indizess[0], plot_indizess[1]],
                        #               weights,
                        #               fit[:, plot_indizess[0], plot_indizess[1]])

                        sigm = sigm * sig
                        qual_block_nu = sigm/delta_lv

                        [fit, sig, delta_lv] = fitq(fit, qual_block_nu, A, sg_window)

                        # write output raster
                        write_fitted_raster_to_disk(fit[fit_nr], out_dir_fit, tile, fitted_raster_band_name, master_raster_info)

                        sigm = sigm ** 0            # set back to ones
                        del delta_lv, fit
                        print("- FINISHED Fit after ", time.time() - epoch_start, " [sec]\n")
                        # except Exception as BrokenFirstIteration:
                        #     print("### ERROR - Something went wrong in the first iteration \n  - {}".format(BrokenFirstIteration))
                        #     break
                        # except KeyboardInterrupt:
                        #     print("### PROGRAMM ENDED BY USER")
                        #     break

                    elif ts_epoch == calc_from_to[1]:
                        break

                    else:
                        try:

                            # update data and qual information
                            data_block, qual_block, fitted_raster_band_name = update_data_block_mp(data_block, qual_block, in_dir_tf, in_dir_qs, tile, list_data, list_qual, sg_window, fit_nr, ts_epoch, weights, name_weights_addition)

                            #A, data_block, qual_block, iv, l_max, l_min = additional_stat_info_raster_numpy(data_block, qual_block, sg_window, device, half_window, center)

                            print("\nStart fitting %s - Nr %d out of %d "
                                  "\n-------------------------------------------" % (fitted_raster_band_name, ts_epoch + 1, len_list_data))
                            print("DATABLOCK: \n", data_block[:, 0, 0])

                            job_list_with_data_inidzes = []  # for mp pool
                            cou = 0
                            start_interp_time = time.time()

                            # create sections that should run in parallel
                            for part in range(0, master_raster_info[2], number_of_rows_data_part):
                                print(part)
                                info_dict = {"from": part, "to": part + number_of_rows_data_part, "shm": shm,
                                             "process_nr": cou,
                                             "dim": (sg_window, master_raster_info[2], master_raster_info[3]),
                                             "num_of_bytes": num_of_buf_bytes}
                                job_list_with_data_inidzes.append(info_dict)
                                cou += 1

                            multi_linear_interpolation(job_list_with_data_inidzes)
                            print("finished interpolation in ", time.time() - start_interp_time, " [sec] ")


                            [fit, sig, delta_lv] = fitq(data_block, qual_block, A, sg_window)


                            sigm = sigm * sig
                            qual_block_nu = sigm / delta_lv

                            [fit, sig, delta_lv] = fitq(fit, qual_block_nu, A, sg_window)


                            fit_layer = fit[fit_nr]

                            # write output raster
                            write_fitted_raster_to_disk(fit_layer, out_dir_fit, tile, fitted_raster_band_name, master_raster_info)

                            sigm = sigm ** 0  # set back to ones
                            del delta_lv
                            print("- FINISHED Fit after ", time.time() - epoch_start, " [sec]\n")
                        except Exception as BrokenFurtherIteration:
                            print("### ERROR - Something went wrong in the following iterations \n  - {}".format(BrokenFurtherIteration))
                            break
                        # except KeyboardInterrupt:
                        #     print("### PROGRAMM ENDED BY USER")
                        #     break

            break
        print("elapsed time: ", time.time() - start , " [sec]")
        print("Programm ENDE")
    except KeyboardInterrupt:
        print("### PROGRAMM ENDED BY USER")
