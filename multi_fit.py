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
    list_qual = input_data_dict["qual"]
    list_data = input_data_dict["data"]
    window_size = input_data_dict["sg_window"]
    out_dir = input_data_dict["out_dir"]
    qual_dir = input_data_dict["in_dir_qs"]
    data_dir = input_data_dict["in_dir_tf"]

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


def multi_fit(jobs_list):
    # for root in jobs_list:
    #     convert_hdf(root)
    #     break
    with multiprocessing.Pool() as pool:
        pool.map(process_tile, jobs_list)


if __name__ == "__main__":

    start = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available")
    else:
        device = torch.device("cpu")
        print("CPU is available")

    in_dir_qs = r"R:\modis\v6\tiff\MCD43A2"
    in_dir_tf = r"R:\modis\v6\tiff\MCD43A4"
    out_dir_fit = r"E:\MODIS_Data\v6\fitted"

    # kacheln = ["h18v04", "h18v03", "h19v03", "h19v04"]
    kacheln = "h18v04"
    bands = list(range(1,8,1))
    print(bands)
    sg_window = 15

    list_of_bands_to_process = []

    qual_dir = os.path.join(str(in_dir_qs), kacheln)
    os.chdir(qual_dir)
    qual_files_names_list = sorted(glob.glob("*.tif"))
    data_dir = os.path.join(str(in_dir_tf), kacheln)
    os.chdir(data_dir)
    data_files_names_list = sorted(glob.glob("*.tif"))

    for b in bands:
        list_of_bands_to_process.append({"band":b,
                                         "qual":qual_files_names_list,
                                         "data": data_files_names_list,
                                         "in_dir_qs": in_dir_qs,
                                         "in_dir_tf": in_dir_tf,
                                         "sg_window": sg_window,
                                         "out_dir": out_dir_fit})


    multi_fit(list_of_bands_to_process)













    print("elapsed time: ", time.time() - start , " [sec]")
    print("Programm ENDE")