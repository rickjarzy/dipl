import os
import glob
import numpy
import multiprocessing
from multiprocessing import shared_memory
from osgeo import gdal, gdalconst
from matplotlib import pyplot as plt




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


def init_data_block_fft(sg_window, band, in_dir_qs, in_dir_tf, tile, list_qual, list_data, num_ob_buf_bytes, master_raster_info):

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

    shm = shared_memory.SharedMemory(create=True, size=num_ob_buf_bytes)
    data_block = numpy.ndarray((sg_window, master_raster_info[2], master_raster_info[3]), dtype=numpy.int16, buffer=shm.buf)

    shm_qual = shared_memory.SharedMemory(create=True, size=num_ob_buf_bytes)
    qual_block = numpy.ndarray((sg_window, master_raster_info[2], master_raster_info[3]), dtype=numpy.int16, buffer=shm_qual.buf)

    print("\n# START READING SATDATA for BAND {}".format(band))
    for i in range(0, len(list_data), 1):

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

            del data_ras

        except Exception as ErrorRasterDataReading:
            print("### ERROR while reading satellite raster:\n {}".format(ErrorRasterDataReading))

    print("data_block from readout: ", data_block[:,2000,100])
    return data_block, qual_block, shm, shm_qual

def update_data_block_fft(data_block, qual_block, in_dir_tf, in_dir_qs, tile, list_qual, list_data, weights):

    # iter through data and qual list
    for i in range(0, len(list_data), 1):
        try:

            # update datablock
            # -----------------
            print("# UPDATE Ras Data File: ", list_data[i])
            data_block[i, :, :] = gdal.Open(os.path.join(in_dir_tf, tile, list_data[i])).ReadAsArray()

        except Exception as ErrorQualRasReading:
            print("### ERROR while updateing satellite raster block:\n {}".format(ErrorQualRasReading))

        try:

            # update qualblock
            # ----------------
            print("# UPDATE Qual Data File: ", list_qual[i])
            qual_data_new = gdal.Open(os.path.join(in_dir_qs, tile, list_qual[i])).ReadAsArray()

            # update weights
            qual_data_new = numpy.where(qual_data_new == 0, weights[0], qual_data_new)
            qual_data_new = numpy.where(qual_data_new == 1, weights[1], qual_data_new)
            qual_data_new = numpy.where(qual_data_new == 2, weights[2], qual_data_new)
            qual_data_new = numpy.where(qual_data_new == 3, weights[3], qual_data_new)

            #qual_data_new[qual_data_new == 255] = numpy.nan

            qual_block[i, :, :] = qual_data_new
            del qual_data_new

        except Exception as ErrorQualRasReading:
            print("### ERROR while updateing quality raster block:\n {}".format(ErrorQualRasReading))

    return data_block, qual_block

def init_data_block_sg_fft(sg_window, band, in_dir_qs, in_dir_tf, tile, list_qual, list_data, num_ob_buf_bytes, fit_nr, name_weights_addition,  master_raster_info, poly=False):

    """

    Parameters
    ----------
    sg_window
    band
    in_dir_qs
    in_dir_tf
    tile
    list_qual
    list_data
    num_ob_buf_bytes
    fit_nr
    name_weights_addition
    master_raster_info

    Returns
    -------

    """

    #data_block = numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])
    shm = shared_memory.SharedMemory(create=True, size=num_ob_buf_bytes)
    shm_qual = shared_memory.SharedMemory(create=True, size=num_ob_buf_bytes)
    if poly:
        data_block = numpy.ndarray((sg_window, master_raster_info[2], master_raster_info[3]), dtype=numpy.float64, buffer=shm.buf)
        qual_block = numpy.ndarray((sg_window, master_raster_info[2], master_raster_info[3]), dtype=numpy.float64, buffer=shm_qual.buf)
    else:

        data_block = numpy.ndarray((sg_window, master_raster_info[2], master_raster_info[3]), dtype=numpy.int16, buffer=shm.buf)
        qual_block = numpy.ndarray((sg_window, master_raster_info[2], master_raster_info[3]), dtype=numpy.int16, buffer=shm_qual.buf)

    print("\n# START READING SATDATA for BAND {}".format(band))
    for i in range(0, sg_window, 1):

        # load qual file
        try:
            qual_ras = gdal.Open(os.path.join(in_dir_qs, tile, list_qual[i]), gdal.GA_ReadOnly)

            print("# load qual data for band %d: %s" % (band, list_qual[i]))
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
    return data_block, qual_block, shm, shm_qual, fitted_raster_band_name


def update_data_block_sg_fft(data_block, qual_block, in_dir_tf, in_dir_qs, tile, list_qual, list_data, sg_window, ts_epoch, fit_nr, name_weights_addition, weights, poly=False):

    try:

    # update datablock
    # -----------------
        data_block[0:-1, :, :] = data_block[1:, :, :]
        print("# UPDATE Ras Data File: ", list_data[sg_window - 1 + ts_epoch])

        data_block[sg_window - 1, :, :] = gdal.Open(os.path.join(in_dir_tf, tile, list_data[sg_window - 1 + ts_epoch])).ReadAsArray()

    except Exception as ErrorQualRasReading:
        print("### ERROR while updateing satellite raster block:\n {}".format(ErrorQualRasReading))

    try:

        # update qualblock
        # ----------------
        print("# UPDATE Qual Data File: ", list_qual[sg_window - 1 + ts_epoch])
        qual_block[0:-1, :, :] = qual_block[1:, :, :]


        qual_data_new = gdal.Open(os.path.join(in_dir_qs, tile, list_qual[sg_window - 1 + ts_epoch])).ReadAsArray()

        # update weights
        qual_data_new = numpy.where(qual_data_new == 0, weights[0], qual_data_new)
        qual_data_new = numpy.where(qual_data_new == 1, weights[1], qual_data_new)
        qual_data_new = numpy.where(qual_data_new == 2, weights[2], qual_data_new)
        qual_data_new = numpy.where(qual_data_new == 3, weights[3], qual_data_new)

        if poly:
            qual_data_new[qual_data_new == 255] = numpy.nan

        qual_block[sg_window - 1, :, :] = qual_data_new

        fitted_raster_band_name = list_data[fit_nr + ts_epoch][:-4] + name_weights_addition % str(sg_window)

        del qual_data_new

    except Exception as ErrorQualRasReading:
        print("### ERROR while updateing quality raster block:\n {}".format(ErrorQualRasReading))

    return data_block, qual_block, fitted_raster_band_name


def perform_fft(input_info, plot=False):


    print("\nspawn FFT process nr : ", input_info["process_nr"])

    existing_shm = shared_memory.SharedMemory(name=input_info["shm"].name)

    # check if poly calculation is on to be executed, because then there must be an additional key in the dictionary
    if "poly" in input_info.keys():
        poly = True
        reference_to_data_block = numpy.ndarray(input_info["dim"], dtype=numpy.float, buffer=existing_shm.buf)[:, input_info["from"]:input_info["to"], :]
    else:
        poly = False
        reference_to_data_block = numpy.ndarray(input_info["dim"], dtype=numpy.int16, buffer=existing_shm.buf)[:, input_info["from"]:input_info["to"], :]

    # get data to process out of buffer


    print("ref data dtype: ", reference_to_data_block.dtype)
    print("\n")

    # reshape data from buffer to 2d matrix with the time as y coords and x as the values
    data_mat = reference_to_data_block.reshape(reference_to_data_block.shape[0],
                                               reference_to_data_block.shape[1] * reference_to_data_block.shape[2])

    # strore orig time, cols and row information - needed for reshaping
    orig_time = reference_to_data_block.shape[0]
    orig_rows = reference_to_data_block.shape[1]
    orig_cols = reference_to_data_block.shape[2]

    # if plots are whished
    if plot:
        existing_shm_qual = shared_memory.SharedMemory(name=input_info["shm_qual"].name)
        if poly:
            reference_to_qual_block = numpy.ndarray(input_info["dim"], dtype=numpy.float64, buffer=existing_shm_qual.buf)[:, input_info["from"]:input_info["to"], :]
        else:
            reference_to_qual_block = numpy.ndarray(input_info["dim"], dtype=numpy.int16, buffer=existing_shm_qual.buf)[:, input_info["from"]:input_info["to"], :]

        qual_weights = input_info["weights"]
        qual_factor = 1
        qual_mat = reference_to_qual_block.reshape(reference_to_qual_block.shape[0],
                                                   reference_to_qual_block.shape[1] * reference_to_qual_block.shape[2])
        qual_mat[qual_mat == 255] = numpy.nan

    print("Data Mat Shape", data_mat.shape)

    # setting int Nan value to numpy.nan --> transforms dataytpe to floa64!!!
    data_mat = numpy.where(data_mat == 32767, numpy.nan, data_mat)
    n = data_mat.shape[0]
    t = numpy.arange(0, n, 1)

    # iter through
    for i in range(0, data_mat.shape[1], 1):

        data_mat_v_nan = numpy.isfinite(data_mat[:, i])
        data_mat_v_t = numpy.arange(0, len(data_mat_v_nan), 1)

        if False in data_mat_v_nan:
            try:

                # interpolate on that spots
                data_mat_v_interp = numpy.round(numpy.interp(data_mat_v_t,
                                                             data_mat_v_t[data_mat_v_nan],
                                                             data_mat[:, i][data_mat_v_nan]))

                # calculate the fft
                f_hat = numpy.fft.fft(data_mat_v_interp, n)
                # and the power spectrum - which frequencies are dominant
                power_spectrum = f_hat * numpy.conj(f_hat) / n

                # get the max power value
                max_fft_spectr_value = numpy.max(power_spectrum)
                # set it to zeros so one can find those frequencies that are far lower and important but still no noise
                power_spec_no_max = numpy.where(power_spectrum == max_fft_spectr_value, 0, power_spectrum)

                threshold_remaining_values = numpy.nanmax(power_spec_no_max) / 2

                indices = power_spectrum > threshold_remaining_values
                f_hat = indices * f_hat
                ffilt = numpy.fft.ifft(f_hat)

                if plot:
                    if i <= 3:
                        print("proces nr %d i == %d" % (input_info["process_nr"], i))
                        print("data mat: ", data_mat[:, i])
                        print("data_mat_v_interp", data_mat_v_interp)

                        # print("data mat:", data_mat[:, i])
                        # print("data_mat.dtype: ", data_mat.dtype)
                        # print("data_mat_interp.dtype: ", data_mat_v_interp.dtype)
                        ffilt = numpy.round(ffilt).astype(numpy.int16)
                        # print("\ntransfrom to int16: ", data_mat_v_interp)
                        # print("\ndata_mat_interp.dtype: ", data_mat_v_interp.dtype)
                        fig, axs = plt.subplots(3, 1)

                        good_qual = numpy.where(qual_mat[:, i] == qual_weights[0], qual_weights[0], numpy.nan) * qual_factor
                        okay_qual = numpy.where(qual_mat[:, i] == qual_weights[1], qual_weights[1], numpy.nan) * qual_factor
                        bad_qual = numpy.where(qual_mat[:, i] == qual_weights[2], qual_weights[2], numpy.nan) * qual_factor
                        really_bad_qual = numpy.where(qual_mat[:, i] == qual_weights[3], qual_weights[3],numpy.nan) * qual_factor
                        nan_values = numpy.where(qual_mat[:, i] == 255, qual_weights[3],numpy.nan )

                        plt.sca(axs[0])
                        plt.plot(t, data_mat[:, i], color='c', LineWidth=3, label="raw data")
                        plt.plot(t, data_mat_v_interp, color='k', LineWidth=1, linestyle='--', label='lin interp')
                        plt.plot(t, ffilt, color="k", LineWidth=2, label='FFT Filtered')

                        plt.plot(t, good_qual, 'go', label="Good Quality")
                        plt.plot(t, okay_qual, 'yo', label="Okay Quality")
                        plt.plot(t, bad_qual, 'o', color='orange', label="Bad Quality")
                        plt.plot(t, really_bad_qual, 'ro', label="Really Bad Quality")
                        plt.plot(t, nan_values, 'ko', label="NaN Values")

                        plt.xlim(t[0], t[-1])
                        plt.ylabel("Intensity [%]")
                        plt.xlabel("Time [days]")
                        plt.legend()

                        plt.sca(axs[1])
                        plt.plot(t, power_spectrum, color="c", LineWidth=2, label="Noisy")
                        plt.plot(t, power_spectrum, 'b*', LineWidth=2, label="Noisy")
                        plt.plot(t[0], t[-1])
                        plt.xlabel("Power Spectrum [Hz]")
                        plt.ylabel("Power")
                        plt.title("Power Spectrum Analyses - Max: {} - Threshold: {}".format(max_fft_spectr_value,
                                                                                             numpy.nanmean(
                                                                                                 power_spectrum)))

                        plt.sca(axs[2])
                        plt.plot(t, power_spec_no_max, color="c", LineWidth=2, label="Noisy")
                        plt.plot(t, power_spec_no_max, 'b*', LineWidth=2, label="Noisy")
                        plt.plot(t[0], t[-1])
                        plt.xlabel("Power Spectrum no max [Hz]")
                        plt.ylabel("Power")
                        plt.title("Power Spectrum Analysis - removed big max {} - Max: {} - Threshold: {}".format(
                            max_fft_spectr_value, numpy.nanmax(power_spec_no_max), threshold_remaining_values))

                        # plot data
                        plt.show()

                data_mat[:, i] = ffilt

            except:
                # gets triggered most if there are only nans in the array
                continue

        else:
            # calculate the fft
            f_hat = numpy.fft.fft(data_mat[:, i], n)
            # and the power spectrum - which frequencies are dominant
            power_spectrum = f_hat * numpy.conj(f_hat) / n

            # get the max power value
            max_fft_spectr_value = numpy.max(power_spectrum)
            # set it to zeros so one can find those frequencies that are far lower and important but still no noise
            power_spec_no_max = numpy.where(power_spectrum == max_fft_spectr_value, 0, power_spectrum)

            threshold_remaining_values = numpy.nanmax(power_spec_no_max) / 2

            indices = power_spectrum > threshold_remaining_values
            f_hat = indices * f_hat
            ffilt = numpy.fft.ifft(f_hat)
            data_mat[:, i] = ffilt



    # transorm float64 back to INT16!!
    # save interpolation results on the shared memory object
    if poly:
        reference_to_data_block[:] = numpy.round(data_mat.reshape(orig_time, orig_rows, orig_cols))
    else:
        reference_to_data_block[:] = numpy.round(data_mat.reshape(orig_time, orig_rows, orig_cols)).astype(numpy.int16)
    #if plot:
        #existing_shm_qual.close()
    #existing_shm.close()
def multi_fft(job_list):

    with multiprocessing.Pool() as pool:
        pool.map(perform_fft, job_list)

def write_fitted_raster_to_disk_fft(data_block, list_data_names, out_dir_fit, tile, master_raster_info, name_weights_addition):

    # set negative values to nan value of 32767
    data_block[data_block < 0] = 32767
    data_block[data_block > 9999] = 9999

    for i in range(0, len(list_data_names), 1):

        # write output raster
        print("- Fitlayer stats: ", data_block.shape)
        out_ras_name = os.path.join(out_dir_fit, tile, list_data_names[i][:-4]+name_weights_addition)
        print("- Writing file : ", out_ras_name)

        #master_raster_info[-1] holds the driver for the file to create
        out_ras = master_raster_info[-1].Create(out_ras_name, master_raster_info[2], master_raster_info[3], 1,
                                                gdalconst.GDT_Int16, options=['COMPRESS=LZW'])
        out_band = out_ras.GetRasterBand(1)
        out_band.WriteArray(data_block[i, :, :])
        out_band.SetNoDataValue(32767)
        out_ras.SetGeoTransform(master_raster_info[0])
        out_ras.SetProjection(master_raster_info[1])
        out_ras.FlushCache()
        del out_ras, out_band, out_ras_name