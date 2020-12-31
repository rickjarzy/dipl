from __future__ import print_function
import torch
import os
import glob
import time
import socket
import numpy
from utils_numpy import write_fitted_raster_to_disk
from utils_fft import init_data_block_sg_fft, get_master_raster_info, multi_fft, update_data_block_sg_fft
import fit_config

"""

ATTENTION - THIS SOFTWARE FITS FOR AN ENTIRE YEAR!!!!!!


"""
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

        sg_window = 15                                       # fit 15 epochs
        window_arr = numpy.arange(0,sg_window,1)             # range from 0 to sg_window
        fit_nr = int(numpy.median(window_arr))               # just a tensor fomr 0 to sg_window
        center = int(numpy.median(window_arr))               # center index of the data stack
        sigm = numpy.ones((sg_window, 2400,2400))
        half_window = int(numpy.floor(sg_window/2))

        A = numpy.ones([sg_window, 3])
        A[:, 1] = numpy.arange(1, sg_window + 1, 1)
        A[:, 2] = numpy.arange(1, sg_window + 1, 1)
        A[:, 2] = A[:, 2] ** 2

        weights = [1, 2, 3, 3]

        name_weights_addition = ".single_fft_sg_%s.tif"

        calc_from_to = [0, 355]                #39 = 2000057 -  =

        master_raster_info = get_master_raster_info(in_dir_tf, tile, "MCD43A4")

        # multiprocessing constants
        num_of_pyhsical_cores = 4 - 1
        number_of_rows_data_part = master_raster_info[2] // num_of_pyhsical_cores
        num_of_buf_bytes = sg_window * master_raster_info[2] * master_raster_info[3] * 8


        for b in bands:
            os.chdir(os.path.join(in_dir_qs, tile))
            list_qual = sorted(glob.glob("MCD43A2.*.band_%d.tif" % b))[calc_from_to[0]:calc_from_to[1]]

            os.chdir(os.path.join(in_dir_tf, tile))
            list_data = sorted(glob.glob("MCD43A4.*.band_%d.tif" % b))[calc_from_to[0]:calc_from_to[1]]
            len_list_data = len(list_data)

            # check if qual and sat data have the same amount of files
            if int(len(list_qual)) != int(len(list_data)):
                print("Len list_qual: ", len(list_qual))
                print("Len list_data: ", len(list_data))
                print("\nBand %s cannot be processed!\n" % b)
                # print("len data %d != %d qual: \n" % ( len(list_data), len(list_qual)))

            else:

                for ts in range(0, len_list_data, 1):
                    epoch_start = time.time()


                    #list with file names to process
                    year_list_data = list_data[ts:ts + sg_window]
                    year_list_qual = list_qual[ts:ts + sg_window]

                    if ts == 0:
                        #try:
                        # Initialize data fitting -load satellite data into data blocks
                        # ============================================================

                        data_block, qual_block, shm, shm_qual, fitted_raster_band_name = init_data_block_sg_fft(sg_window, b, in_dir_qs, in_dir_tf,
                                                                                    tile, list_qual, list_data,
                                                                                    num_of_buf_bytes,fit_nr, name_weights_addition, master_raster_info)

                        print("\nStart fitting FFT SG block - Nr %d out of %d \n-------------------------------------------" % (ts, len_list_data))
                        print("Shape Datablock: ", data_block.shape)
                        # FFT Logic Here

                        job_list_with_data_inidzes = []  # for mp pool
                        cou = 0
                        start_interp_time = time.time()

                        # create sections that should run in parallel
                        for part in range(0, master_raster_info[2], number_of_rows_data_part):
                            print(part)
                            info_dict = {"from": part, "to": part + number_of_rows_data_part, "shm": shm,
                                         "process_nr": cou, "shm_qual": shm_qual, "weights": weights,
                                         "dim": (sg_window, master_raster_info[2], master_raster_info[3]),
                                         "num_of_bytes": num_of_buf_bytes}
                            job_list_with_data_inidzes.append(info_dict)
                            cou += 1

                        multi_fft(job_list_with_data_inidzes)
                        print("finished FFT in ", time.time() - start_interp_time, " [sec] ")

                        write_fitted_raster_to_disk(data_block[fit_nr], out_dir_fit, tile, fitted_raster_band_name, master_raster_info)
                        # except Exception as BrokenFirstIteration:
                        #     print("### ERROR - Something went wrong in the first iteration \n  - {}".format(BrokenFirstIteration))
                        #     shm.unlink()
                        #     shm_qual.unlink()
                        #     break

                    else:
                        try:

                            data_block, qual_block, fitted_raster_band_name = update_data_block_sg_fft(data_block, qual_block, in_dir_tf, in_dir_qs,
                                                                           tile, list_qual, list_data, sg_window, ts, fit_nr, name_weights_addition, weights)

                            print("\nStart fitting FFT year block - Nr %d out of %d \n-------------------------------------------" % ( ts, len_list_data/sg_window))
                            print("DATABLOCK: \n", data_block[:, 0, 0])

                            ## FFT Logic Here

                            job_list_with_data_inidzes = []  # for mp pool
                            cou = 0
                            start_interp_time = time.time()

                            # create sections that should run in parallel
                            for part in range(0, master_raster_info[2], number_of_rows_data_part):
                                print(part)
                                info_dict = {"from": part, "to": part + number_of_rows_data_part, "shm": shm,
                                             "process_nr": cou, "shm_qual": shm_qual, "weights": weights,
                                             "dim": (sg_window, master_raster_info[2], master_raster_info[3]),
                                             "num_of_bytes": num_of_buf_bytes}
                                job_list_with_data_inidzes.append(info_dict)
                                cou += 1

                            multi_fft(job_list_with_data_inidzes)
                            print("finished FFT in ", time.time() - start_interp_time, " [sec] ")

                            # write output raster
                            write_fitted_raster_to_disk(data_block[fit_nr], out_dir_fit, tile, fitted_raster_band_name, master_raster_info)

                            print("- FINISHED Fit after ", time.time() - epoch_start, " [sec]\n")

                        except Exception as BrokenFurtherIteration:
                            print("### ERROR - Something went wrong in the following iterations \n  - {}".format(BrokenFurtherIteration))
                            shm.close()
                            shm_qual.close()
                            shm.unlink()
                            shm_qual.unlink()
                            break
                        # except KeyboardInterrupt:
                        #     print("### PROGRAMM ENDED BY USER")
                        #     break


        print("elapsed time: ", time.time() - start , " [sec]")
        print("Programm ENDE")
        shm.close()
        shm_qual.close()
        shm.unlink()
        shm_qual.unlink()
    except KeyboardInterrupt:
        shm.close()
        shm_qual.close()
        shm.unlink()
        shm_qual.unlink()
        print("### PROGRAMM ENDED BY USER")
