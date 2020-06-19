from __future__ import print_function
import torch
import numpy
import os
import glob
import time
#import multiprocessing
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


def init_data_block(sg_window, in_dir_qs, in_dir_tf, tile, list_qual, list_data):

    data_block = torch.from_numpy(numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])).to(device).share_memory_()
    qual_block = torch.from_numpy(numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])).to(device).share_memory_()

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

def multi_fit_gpu(data_block, qual_block, patch_list):

    print("\n GPU MULTI")
    # with torch.multiprocessing.Pool() as pool:
    #     return_liste = pool.map(check_mp, [data_block, qual_block, patch_list])
    #     print(return_liste)
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.multiprocessing.spawn(check_mp, args=(data_block, qual_block, patch_list), nprocs=8)
    process_id = 0
    process_list = []
    for patch in patch_list:
        print("start process for patch", process_id)

        p = torch.multiprocessing.Process(target=check_mp, args=(data_block, qual_block, patch, process_id))
        process_list.append(p)
        p.start()
        process_id += 0
    print(data_block)
    print("CHECK MULTIPROCESSING")

def check_mp(data_block, qual_block, input_liste, process_id):
    print("Spawn process for id: ", process_id)
    data_block**0


if __name__ == "__main__":

    start = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
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



            data_block, qual_block = init_data_block(sg_window, in_dir_qs, in_dir_tf, tile, list_qual, list_data)
            print("SHAPE OF DATA: ", data_block.shape)

            data_block_indizes = [[index for index in range(i, i+300, 1)] for i in range(0, 2400, 300)]


            ##test = [data_block[:, i, :] for i in data_block_indizes]

            return_test_liste = multi_fit_gpu(data_block, qual_block, data_block_indizes)

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


    print("elapsed time: ", time.time() - start , " [sec]")
    print("Programm ENDE")