import numpy
import multiprocessing
from multiprocessing import shared_memory
from osgeo import gdal
import os
import glob

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


def init_data_block_mp(sg_window, band, in_dir_qs, in_dir_tf, tile, list_qual, list_data, num_ob_buf_bytes, master_raster_info, fit_nr, name_weights_addition):

    """
    Creates a initial datablock for the modis data and returns a numpy ndim array
    Parameters
    ----------
    sg_window
    band
    in_dir_qs
    in_dir_tf
    tile
    list_qual
    list_data
    device
    master_raster_info
    fit_nr
    name_weights_addition

    Returns
    -------

    """

    #data_block = numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])
    qual_block = numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])
    shm = shared_memory.SharedMemory(create=True, size=num_ob_buf_bytes)
    data_block = numpy.ndarray((sg_window, master_raster_info[2], master_raster_info[3]), dtype=numpy.int16, buffer=shm.buf)

    #data_block.share_memory_()
    #qual_block.share_memory_()
    print("\n# START READING SATDATA for BAND {}".format(band))
    for i in range(0, sg_window, 1):

        # load qual file
        try:
            qual_ras = gdal.Open(os.path.join(in_dir_qs, tile, list_qual[i]), gdal.GA_ReadOnly)

            #print("load qual data for band %d: %s" % (band, list_qual[i]))
            #qual_band = qual_ras.GetRasterBand(1)
            qual_block[i, :, :] = qual_ras.ReadAsArray()

            del qual_ras
        except Exception as ErrorQualRasReading:
            print("### ERROR while reading quality raster:\n {}".format(ErrorQualRasReading))
        # load satellite data
        try:
            data_ras = gdal.Open(os.path.join(in_dir_tf, tile, list_data[i]), gdal.GA_ReadOnly)

            print("# load sat data for band %s: %s" % (str(band), list_data[i]))
            #data_band = data_ras.GetRasterBand(1)
            data_block[i, :, :] = data_ras.ReadAsArray()

            # collect epochs raster name
            if fit_nr == i:
                print("\n# Name of fitted tile will be:  {} \n".format(os.path.join(tile, list_data[i])))

                fitted_raster_band_name = list_data[i][:-4] + name_weights_addition % str(sg_window)

            del data_ras

        except Exception as ErrorRasterDataReading:
            print("### ERROR while reading satellite raster:\n {}".format(ErrorRasterDataReading))

    print("data_block from readout: ", data_block[:,2000,100])
    return data_block, qual_block, fitted_raster_band_name, shm


def additional_stat_info_raster_mp(qual_block, weights):
    """
    Converts qual raster data to weights
    Parameters
    ----------
    qual_block ndarry - size (sg_window, master_raster_info[2], master_raster_info[3]
    weights - list - contains the weights for the spezific quality coding

    Returns qualdatablock with new weights encoding
    -------

    """
    print("# Processing Numpy")

    qual_block[qual_block == 0] = weights[0]
    qual_block[qual_block == 1] = weights[1]
    qual_block[qual_block == 2] = weights[2]
    qual_block[qual_block == 3] = weights[3]

    qual_block[qual_block == 255] = numpy.nan  # set to 0 so in the ausgleich the nan -> zero convertion is not needed


    return qual_block


def update_data_block_mp(data_block, qual_block, in_dir_tf, in_dir_qs, tile, list_data, list_qual, sg_window, fit_nr, ts, weights, name_weights_addition):

    # update datablock
    # -----------------
    data_block[0:-1, :, :] = data_block[1:, :, :]
    print("# UPDATE Ras Data File: ", list_data[sg_window-1 + ts])

    ras_data_new = gdal.Open(os.path.join(in_dir_tf, tile, list_data[sg_window-1 + ts])).ReadAsArray()
    data_block[sg_window-1, :, :] = ras_data_new

    # update qualblock
    # ----------------
    qual_block[0:-1, :, :] = qual_block[1:, :, :]
    print("# UPDATE Qual Data File: ", list_qual[sg_window - 1 + ts])

    qual_data_new = gdal.Open(os.path.join(in_dir_qs, tile, list_qual[sg_window-1 + ts])).ReadAsArray()


    # update weights
    qual_data_new = numpy.where(qual_data_new == 0, weights[0], qual_data_new)
    qual_data_new = numpy.where(qual_data_new == 1, weights[1], qual_data_new)
    qual_data_new = numpy.where(qual_data_new == 2, weights[2], qual_data_new)
    qual_data_new = numpy.where(qual_data_new == 3, weights[3], qual_data_new)

    qual_block[sg_window - 1, :, :] = qual_data_new

    fitted_raster_band_name = list_data[fit_nr + ts][:-4] + name_weights_addition % str(sg_window)

    return data_block, qual_block, fitted_raster_band_name


def multi_linear_interpolation(job_list):

    with multiprocessing.Pool() as pool:
        pool.map(multi_lin_interp_process, job_list)

def multi_lin_interp_process(input_info):

    print("\nspawn process nr : ", input_info["process_nr"])
    existing_shm = shared_memory.SharedMemory(name=input_info["shm"].name)

    # get data to process out of buffer
    reference_to_data_block = numpy.ndarray(input_info["dim"], dtype=numpy.int16, buffer=existing_shm.buf)[:, input_info["from"]:input_info["to"], :]

    # strore orig time, cols and row information - needed for reshaping
    orig_time = reference_to_data_block.shape[0]
    orig_rows = reference_to_data_block.shape[1]
    orig_cols = reference_to_data_block.shape[2]


    print("ref data dtype: ", reference_to_data_block.dtype)
    print("\n")

    # reshape data from buffer to 2d matrix with the time as y coords and x as the values
    data_mat = reference_to_data_block.reshape(reference_to_data_block.shape[0], reference_to_data_block.shape[1]*reference_to_data_block.shape[2])
    print(data_mat.shape)

    # setting int Nan value to numpy.nan --> transforms dataytpe to floa64!!!
    data_mat = numpy.where(data_mat == 32767, numpy.nan, data_mat)

    # iter through
    for i in range(0, data_mat.shape[1], 1):

        data_mat_v_nan = numpy.isfinite(data_mat[:, i])
        data_mat_v_t = numpy.arange(0, len(data_mat_v_nan), 1)

        if False in data_mat_v_nan:
            try:

                data_mat_v_interp = numpy.round(numpy.interp(data_mat_v_t, data_mat_v_t[data_mat_v_nan], data_mat[:,i][data_mat_v_nan]))

                if i == 0:
                    print("i == 0")
                    print("data mat: ", data_mat[:, i])
                    print("data_mat_v_interp", data_mat_v_interp)
                    data_mat[:, i] = data_mat_v_interp
                    # print("data mat:", data_mat[:, i])
                    # print("data_mat.dtype: ", data_mat.dtype)
                    # print("data_mat_interp.dtype: ", data_mat_v_interp.dtype)
                    data_mat_v_interp = numpy.round(data_mat_v_interp).astype(numpy.int16)
                    # print("\ntransfrom to int16: ", data_mat_v_interp)
                    # print("\ndata_mat_interp.dtype: ", data_mat_v_interp.dtype)
                    data_mat[:, i] = data_mat_v_interp
                    continue

                data_mat[:, i] = data_mat_v_interp
            except:
                break

        else:
            pass

    # transorm float64 back to INT16!!
    # save interpolation results on the shared memory object
    reference_to_data_block[:] = numpy.round(data_mat.reshape(orig_time, orig_rows, orig_cols)).astype(numpy.int16)



if __name__ == "__main__":
    print("Programm ENDE")