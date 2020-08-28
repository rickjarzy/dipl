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
    half_window = numpy.floor(sg_window/2)



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

            data_block, qual_block = init_data_block(sg_window, b, in_dir_qs, in_dir_tf, tile, list_qual, list_data, device, master_raster_info)


            data_block_indizes = [[index for index in range(i, i+300, 1)] for i in range(0, 2400, 300)]

            #data_block = torch.reshape(data_block, (sg_window, master_raster_info[2]*master_raster_info[3]))                                            # contains instead of 32767 --> 0
            data_block = torch.reshape(data_block, (master_raster_info[2] * master_raster_info[3], sg_window, 1))  # contains instead of 32767 --> 0


            #qual_block = torch.reshape(qual_block, (sg_window, master_raster_info[2]*master_raster_info[3]))                                            # contains instead of 255 --> 0
            qual_block = torch.reshape(qual_block, (master_raster_info[2] * master_raster_info[3], sg_window, 1))  # contains instead of 255 --> 0
            print("SHAPE OF QUAL: ", qual_block.shape)
            print("SHAPE OF DATA: ", data_block.shape)
            print(data_block[0,:,:])
            sigm = torch.ones(sg_window, 2400**2)

            qual_block[qual_block==0] = 1
            qual_block[qual_block==1] = 0.75
            qual_block[qual_block==2] = 0.1
            qual_block[qual_block==3] = 0.01

            noup_zero = torch.zeros(2400**2, sg_window, 1)         # noup = number of used pixels
            noup_ones = torch.ones(2400**2, sg_window, 1)
            noup_tensor = torch.where(qual_block == 255, noup_zero, noup_ones)
            noup_tensor[noup_tensor != 0]=1

            qual_block[qual_block==255] = 0                     # set to 0 so in the ausgleich the nan -> zero convertion is not needed
                                                                # nan will be replaced by zeros so this is a shortcut to avoid that transformation
            data_block[data_block==32767] = 0

            A = torch.ones(sg_window, 3).to(device)
            torch.arange(1, sg_window + 1, 1, out=A[:, 1])
            torch.arange(1, sg_window + 1, 1, out=A[:, 2])
            A[:, 2] = A[:, 2]**2

            # data ini to count how many data epochs are to the left and to the right of the center epoch etc

            l_max = torch.ones([2400**2, sg_window, 1]) * torch.max(data_block, dim=0).values
            l_min = torch.ones([2400**2, sg_window, 1]) * torch.min(data_block, dim=0).values

            noup_l = torch.sum(noup_tensor[:, 0:center, :], dim=1)                              # numbers of used epochs on the left side
            print("noup_l", noup_l)
            noup_r = torch.sum(noup_tensor[:, center + 1:, :], dim=1)                           # numbers of used epochs on the right side
            noup_c = torch.sum(noup_tensor[:, center, :], dim=1)                                # numbers of used epochs on the center epoch
            noup_c = torch.reshape(noup_c, (noup_c.shape[0], 1))
            n = torch.sum(noup_tensor, dim=1)
            print("n: ", n.shape)
            print("noup_l: ", noup_l.shape)
            print("noup_r: ", noup_r.shape)
            print("noup_c: ", noup_c.shape)
            ids_for_lin_fit = numpy.concatenate(
                                                    (numpy.where(noup_l.numpy() <= 3),
                                                     numpy.where(noup_r.numpy() <= 3),
                                                     numpy.where(noup_c.numpy() <= 0),
                                                     numpy.where(n.numpy() <= half_window)
                                                     ),
                                                    axis=1
                                                )
            iv = numpy.unique(ids_for_lin_fit)              # ids sind gescheckt und passen
            print("n:  ", n[:20])
            print("ids: ", ids_for_lin_fit[:20])
            print("iv: ", iv[:20])
            print(iv.shape)

            #todo: überlegen ob man nicht für links und rechtsseitig der zentralen bildmatrix einen linearen fit machen will wenn zu wenige daten sind
            #todo: fit aus check für cuda und numpy implementieren dann geht die sache in produktion


            print("reshaped data block: ", data_block.shape)
            print("reshaped qual block: ", qual_block.shape)

            print("Start fitting ...")
            [a0, a1, a2] = fitq_cpu(data_block.to(device), qual_block.to(device), A, sg_window)

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
            fit = torch.round(a0 + a1 * torch.reshape(A[:,1], (sg_window, 1)) + a2 * torch.reshape(A[:,2], (sg_window, 1)))      # fit.shape: [15,5_760_000])
            fit[fit != fit] = 0                             # set nan to 0
            print("fited layer")
            print(fit[center, :])
            # calc new weights
            delta_lv = torch.abs(fit - data_block)          # delta_lv.shape: [15,5_760_000]
            delta_lv[delta_lv != delta_lv] = 0              # set nan to 0
            print("delta_lv: ")
            print(delta_lv)
            delta_lv[delta_lv<1] = 1                        #
            sig = torch.sum(delta_lv, 0)                     # sig.shape: [5_760_000]
            print("sig")
            print(sig)
            sigm = sigm * sig                               # sigm.shape: [15, 5_760_000]

            print("sigm")
            print(sigm)

            qual_updated = sigm/delta_lv                    # qual_updated.shape: [15, 5_760_000]

            print("calc new raster matrix - 2nd iteration ...")
            [a0, a1, a2] = fitq_cuda(data_block, qual_updated, A[:, 1], sg_window)
            fit = torch.round(a0 + a1 * torch.reshape(A[:, 1], (sg_window, 1)) + a2 * torch.reshape(A[:, 2], (sg_window, 1)))
            print("fited layer")
            print(fit[center, :])

            #linear fit
            [a0,a1] = fitl_cuda(fit[:,iv], qual_updated[:,iv], A[:, 1], sg_window)
            fit[:,iv] = torch.round(a0 + a1*torch.reshape(A[:, 1], (sg_window, 1)))

            #check if fit is bigger than max occouring values
            fit = torch.where(fit>l_max, l_max, fit)
            fit = torch.where(fit<l_min, l_min, fit)

            # filtered epoch
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