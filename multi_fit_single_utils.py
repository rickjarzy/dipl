from __future__ import print_function
import torch
from torchvision.transforms import ToTensor
from osgeo import gdal
import os
import numpy
import glob
# INPUT GENERELL Gleich grosze bloecke mit nodata aufgefuellt mit gleicher aufloesung und gleichem
# zeitstempel am beginn des filenamens T2008239_B01_originalname  ... mit jahr und doy
# Optional Qualityfile dazu Q2008239_originalname
# Zeitbezug wenn erste szene T2001100 folgt differenz = 7 jahre und 139 Tage, da Schaltjahre dazwischen
# time funktion mit differenz der Tage also ca. 7*365=2555 + 1 schaltjahr + 139 = 2695 fuer xv vektor
# ==============================================================

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
    projection = master_raster.GetProjection()

    block_size_x = master_raster.RasterXSize
    block_size_y = master_raster.RasterYSize

    driver = master_raster.GetDriver()

    return [geo_trafo, projection, block_size_x, block_size_y, driver]


def multi_fit_gpu(data_block, qual_block, patch_list):

    print("\n GPU MULTI")
    # with torch.multiprocessing.Pool() as pool:
    #     return_liste = pool.map(check_mp, [data_block, qual_block, patch_list])
    #     print(return_liste)
    torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.multiprocessing.spawn(check_mp, args=(data_block, qual_block, patch_list), nprocs=8)

    process_id = 0
    process_list = []
    for patch in patch_list:
        print("start process for patch", process_id)

        p = torch.multiprocessing.Process(target=check_mp, args=(data_block, qual_block, patch, process_id))
        p.start()
        process_list.append(p)

        process_id += 0
    for pr in process_list:
        pr.join()

    print(data_block)
    print("CHECK MULTIPROCESSING")

def check_mp(data_block, qual_block, input_liste, process_id):
    print("Spawn process for id: ", process_id)
    data_block**0


if __name__ == "__main__":

    print("Execute Multi Fit Single Utls")