from __future__ import print_function
import torch
import numpy
import os
import glob
import time
import multiprocessing
from torchvision.transforms import ToTensor

from osgeo import gdal

def process_tile(input_data_dict):

    band = input_data_dict["band"]


    window_size = input_data_dict["sg_window"]
    out_dir = input_data_dict["out_dir"]
    qual_dir = input_data_dict["in_dir_qs"]
    data_dir = input_data_dict["in_dir_tf"]
    tile = input_data_dict["tile"]
    os.chdir(os.path.join(qual_dir,tile))
    list_qual = sorted(glob.glob("MCD43A2*.band%d.tif"%band))

    os.chdir(os.path.join(data_dir,tile))
    list_data = sorted(glob.glob("MCD43A4.*.band%d.tif"%band))

    print("Process Band: ", band)


    if int(len(list_qual)) != int(len(list_data)):
        print("\nBand %s cannot be processed!\n" % band)
        #print("len data %d != %d qual: \n" % ( len(list_data), len(list_qual)))

    else:
        #print("\nStart Processing tile %s" % tile)
        #print("len data == qual: ", len(list_data) == len(list_qual))
        tile = list_data[0].split(".")[2]
        master_raster = gdal.Open(list_data[0], gdal.GA_ReadOnly)
        raster_band = master_raster.GetRasterBand(band)
        geo_trafo = master_raster.GetGeoTransform()
        projection = master_raster.GetProjection

        block_size_x = master_raster.RasterXSize
        block_size_y = master_raster.RasterYSize

        driver = master_raster.GetDriver()

        # initial datablock
        ras_data = raster_band.ReadAsArray()
        del master_raster

        print("RASTER size and type: ", ras_data.shape, " - ", type(ras_data))
        print("Qual dir: ", qual_dir)
        print("Ras dir: ", data_dir)

        numbers_of_data_epochs = len(list_data)

        data_block = numpy.zeros([window_size, block_size_x,block_size_y])
        qual_block = numpy.zeros([window_size, block_size_x,block_size_y])

        # Initialize data fiting -load satellite data into data blocks
        # ============================================================
        for i in range(0,window_size,1):

            #load qual file
            try:
                qual_ras = gdal.Open(os.path.join(qual_dir, tile, list_qual[i]), gdal.GA_ReadOnly)

                print("load qual data for band %d: %s" % (band, list_qual[i]))
                qual_band = qual_ras.GetRasterBand(band)
                qual_block[i, :, :] = qual_band.ReadAsArray()

                del qual_ras
            except Exception as ErrorQualRasReading:
                print("# ERROR while reading quality raster:\n {}".format(ErrorQualRasReading))
            # load satellite data
            try:
                data_ras = gdal.Open(os.path.join(data_dir, tile, list_data[i]), gdal.GA_ReadOnly)
                print("load sat data for band %d: %s" % (band, list_data[i]))
                data_band = data_ras.GetRasterBand(band)
                data_block[i, :, :] = data_band.ReadAsArray()

                del data_ras

            except Exception as ErrorRasterDataReading:
                print("# ERROR while reading satellite raster:\n {}".format(ErrorRasterDataReading))

        print("Datablock shape: ", data_block.shape)
        print(data_block)


        # END of Initializing
        # ===============================================================

        # START FITTING
        # ===============================================================

        # for i in range(0,len(list_qual),1):
        #     window_end_index = i + window_size
        #     if window_end_index < numbers_of_data_epochs:
        #
        #         names_of_interrest_qs = list_data[i:i+window_size]
        #         print("Process tile {} index from to: {} - {}".format(tile, i, window_end_index))
        #
        #
        #
        #     else:
        #         print("\nreached end of processing in tile {} - index {} - {}".format(tile, i, window_end_index))
        #         break

def get_master_raster_info(in_dir, tile, sat_product):
    os.chdir(os.path.join(in_dir, tile))
    print(os.path.join(in_dir, tile))
    master_raster_file_name = sorted(glob.glob("*.tif"))[0]

    master_raster = gdal.Open(master_raster_file_name, gdal.GA_ReadOnly)
    geo_trafo = master_raster.GetGeoTransform()
    projection = master_raster.GetProjection

    block_size_x = master_raster.RasterXSize
    block_size_y = master_raster.RasterYSize

    driver = master_raster.GetDriver()

    return [geo_trafo, projection, block_size_x, block_size_y, driver]



def multi_fit(jobs_list):
    # for root in jobs_list:
    #     convert_hdf(root)
    #     break
    with multiprocessing.Pool() as pool:
        pool.map(process_tile, jobs_list)

def multi_fit_gpu(patch_list):
    with torch.multiprocessing.Pool() as pool:
        return_liste = pool.map(checK_mp, patch_list)
        print("CHECK MULTIPROCESSING")

        return return_liste

def checK_mp(input_liste):

    return [element + "_was_processed" for element in input_liste]

def init_data_block(sg_window, in_dir_qs, in_dir_tf, tile, list_qual, list_data):
    for i in range(0, sg_window, 1):

        # load qual file
        try:
            qual_ras = gdal.Open(os.path.join(in_dir_qs, tile, list_qual[i]), gdal.GA_ReadOnly)

            print("load qual data for band %d: %s" % (b, list_qual[i]))
            #qual_band = qual_ras.GetRasterBand(1)
            qual_block[i, :, :] = torch.from_numpy(qual_ras.ReadAsArray()).to(device)

            del qual_ras
        except Exception as ErrorQualRasReading:
            print("# ERROR while reading quality raster:\n {}".format(ErrorQualRasReading))
        # load satellite data
        try:
            data_ras = gdal.Open(os.path.join(in_dir_tf, tile, list_data[i]), gdal.GA_ReadOnly)
            print("load sat data for band %d: %s" % (b, list_data[i]))
            #data_band = data_ras.GetRasterBand(1)
            data_block[i, :, :] = torch.from_numpy(data_ras.ReadAsArray()).to(device)

            del data_ras

        except Exception as ErrorRasterDataReading:
            print("# ERROR while reading satellite raster:\n {}".format(ErrorRasterDataReading))

    print("Datablock shape: ", data_block.shape)
    print(data_block)
    return data_block, qual_block

if __name__ == "__main__":

    start = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available")
    else:
        device = torch.device("cpu")
        print("CPU is available")

    in_dir_qs = r"E:\MODIS_Data\v6\tiff_single\MCD43A2"
    in_dir_tf = r"E:\MODIS_Data\v6\tiff_single\MCD43A4"
    out_dir_fit = r"E:\MODIS_Data\v6\fitted"


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

            data_block = torch.from_numpy(numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])).to(device)
            qual_block = torch.from_numpy(numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])).to(device)

            data_block, qual_block = init_data_block(sg_window, in_dir_qs, in_dir_tf, tile, list_qual, list_data)

            data_block_indizes = [i for i in range(0,2400,300)]
            print("datablock indizes: ", data_block_indizes)

            data_block_slices = []



            test = ["hallo", "check", "shshsh", "jsjsjsjsjsjsjs", "khaskjdhakjshd", "lksjlkfj"]

            return_test_liste = multi_fit_gpu(test)

            print(return_test_liste)

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


    #multi_fit(list_of_bands_to_process)













    print("elapsed time: ", time.time() - start , " [sec]")
    print("Programm ENDE")