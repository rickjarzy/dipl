import os
import numpy
import time
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
from scipy.interpolate import interp1d, griddata
from scipy.signal import argrelextrema



def perfom_fft(data_block):

    print("FFT Data shape: ", data_block.shape)

    f = data_block
    n = data_block.shape[0]
    t = numpy.arange(0,n,1)

    # interpolate on these positions of the data array where nans occure
    data_block_nan_true = numpy.isfinite(data_block)
    data_block_with_lin_interpol_where_nan = numpy.interp(t,t[data_block_nan_true], data_block[data_block_nan_true] )


    f_hat = numpy.fft.fft(data_block_with_lin_interpol_where_nan, n)
    power_spectrum = f_hat * numpy.conj(f_hat) / n

    print("f_hat: ", f_hat)

    fig,axs = plt.subplots(2,1)

    plt.sca(axs[0])
    plt.plot(t,f,color='c', LineWidth=1.5, label="Noisy")
    plt.plot(t,data_block_with_lin_interpol_where_nan, color='k', LineWidth=0.5, label='with interpolation')
    plt.xlim(t[0], t[-1])
    plt.legend()

    plt.sca(axs[1])
    plt.plot(t, power_spectrum, color="c", LineWidth=2, label="Noisy")
    plt.plot(t[0], t[-1])
    plt.xlabel("Power Spectrum [Hz]")
    plt.ylabel("Power")

    # todo: find solution to get good schwellwert for intensity of power value
    print("MAx power spectrum half: ", int(numpy.max(power_spectrum))/2)
    indices = power_spectrum > int(numpy.max(power_spectrum))/2
    #indices = power_spectrum > 500
    power_spectrum_clean = power_spectrum * indices
    f_hat = indices * f_hat
    ffilt = numpy.fft.ifft(f_hat)

    fig,axs2 = plt.subplots(3,1)

    plt.sca(axs2[0])
    plt.plot(t,f,color='c', LineWidth=1.5, label="Noisy")
    plt.plot(t, data_block_with_lin_interpol_where_nan, color='k', LineWidth=1, linestyle='--', label='with interpolation')
    plt.xlim(t[0], t[-1])
    plt.legend()

    plt.sca(axs2[1])
    plt.plot(t,ffilt, color="k", LineWidth=2, label='Filtered')
    plt.xlim(t[0], t[-1])
    plt.legend()

    plt.legend()
    plt.show()

def plot_raw_interp_fitted_data(raw_array, interp_array, fitted_array, qual_data_array, qual_weights_array):
    print("Accessing Raw - Interp - Fitted Data Plot")
    qual_factor = 100
    n = raw_array.shape[0]
    t = numpy.arange(0, n, 1)
    print("here")

    good_qual = numpy.where(qual_data_array == qual_weights_array[0],qual_weights_array[0],numpy.nan) * qual_factor
    okay_qual = numpy.where(qual_data_array == qual_weights_array[1], qual_weights_array[1], numpy.nan) * qual_factor
    bad_qual  = numpy.where(qual_data_array == qual_weights_array[2],qual_weights_array[2], numpy.nan) * qual_factor
    really_bad_qual = numpy.where(qual_data_array == qual_weights_array[3], qual_weights_array[3], numpy.nan) * qual_factor

    fig, axs = plt.subplots(1, 1)

#    plt.sca(axs[0])
    plt.plot(t, raw_array, color='c', LineWidth=3, label="raw data")
    plt.plot(t, interp_array, color='k', LineWidth=1, linestyle='--', label='lin interp')
    plt.plot(t, fitted_array,  color='b', LineWidth=2, label='Poly Filtered data')

    plt.plot(t, good_qual, 'go', label="Good Quality")
    plt.plot(t, okay_qual, 'yo', label="Okay Quality")
    plt.plot(t, bad_qual, 'o', color='orange', label="Bad Quality")
    plt.plot(t, really_bad_qual, 'ro', label="Really Bad Quality")

    plt.xlim(t[0], t[-1])
    plt.ylabel("Intensity [%]")
    plt.xlabel("Time [days]")
    plt.legend()
    plt.show()

    print("Finished Plot Raw Interp Fitted Data")

def plot_raw_data(data_block,qual_block,qual_weights, fit_data=[]):
    print("Poly Data shape: ", data_block.shape)
    print("poly data: ", data_block)
    n = data_block.shape[0]
    t = numpy.arange(0, n, 1)

    qual_factor = 100
    print("here")

    good_qual = numpy.where(qual_block == qual_weights[0],qual_weights[0],numpy.nan) * qual_factor
    okay_qual = numpy.where(qual_block == qual_weights[1], qual_weights[1], numpy.nan) * qual_factor
    bad_qual  = numpy.where(qual_block == qual_weights[2],qual_weights[2], numpy.nan) * qual_factor
    really_bad_qual = numpy.where(qual_block == qual_weights[3], qual_weights[3], numpy.nan) * qual_factor

    # interpolate on these positions of the data array where nans occure
    data_block_nan_true = numpy.isfinite(data_block)
    print("nan search: ", data_block_nan_true)
    data_block_with_lin_interpol_where_nan = numpy.interp(t, t[data_block_nan_true], data_block[data_block_nan_true])

    print("t: ", t)
    print("data_block_nan_true: ", data_block_nan_true)
    print("t[data_block_nan_true]: ", t[data_block_nan_true])
    print("data_block[data_block_nan_true]", data_block[data_block_nan_true])


    f_hat = numpy.fft.fft(data_block_with_lin_interpol_where_nan, n)
    power_spectrum = f_hat * numpy.conj(f_hat) / n
    print("Power Spektrum Max: {} - Mean {} - mean int {}".format(numpy.max(power_spectrum), numpy.mean(power_spectrum), int(numpy.mean(power_spectrum))/5))
    print("Power Spektrum schwellwert: ", int(numpy.max(power_spectrum.real))/2)

    max_fft_spectr_value = numpy.max(power_spectrum)
    power_spec_no_max = numpy.where(power_spectrum == max_fft_spectr_value, 0, power_spectrum)

    threshold_remaining_values = numpy.nanmax(power_spec_no_max)/2
    print("Power Spectrum no max - max: ", numpy.nanmax(power_spec_no_max))
    print("Power Spectrum no max - max/2: ", numpy.nanmax(power_spec_no_max)/2)
    print("Power Spectrum Threshold no max: ", int(threshold_remaining_values))


    indices = power_spectrum > threshold_remaining_values
    power_spectrum_clean = power_spectrum * indices
    f_hat = indices * f_hat
    ffilt = numpy.fft.ifft(f_hat)

    # interpolate cubic object - has to be feeded with a time array - check out plot
    data_block_with_cubic_interp_where_nan = interp1d(t, data_block_with_lin_interpol_where_nan, kind="cubic")

    if len(fit_data) > 0:


        print("DATABLOCK with Interplation: \n", data_block_with_lin_interpol_where_nan)
        fig, axs = plt.subplots(3, 1)

        plt.sca(axs[0])
        plt.plot(t, data_block, color='c', LineWidth=3, label="raw data")
        plt.plot(t, data_block_with_lin_interpol_where_nan, color='k', LineWidth=1, linestyle='--', label='lin interp')
        plt.plot(t, data_block_with_cubic_interp_where_nan(t), color='r', LineWidth=1, linestyle=':', label='cubic interp')
        plt.plot(t, fit_data,  color='b', LineWidth=2, label='Poly Filtered data')
        plt.plot(t, ffilt, color="k", LineWidth=2, label='FFT Filtered')

        plt.plot(t, good_qual, 'go', label="Good Quality")
        plt.plot(t, okay_qual, 'yo', label="Okay Quality")
        plt.plot(t, bad_qual, 'o', color='orange', label="Bad Quality")
        plt.plot(t, really_bad_qual, 'ro', label="Really Bad Quality")

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
        plt.title("Power Spectrum Analyses - Max: {} - Threshold: {}".format(max_fft_spectr_value, numpy.nanmean(power_spectrum)))


        plt.sca(axs[2])
        plt.plot(t, power_spec_no_max, color="c", LineWidth=2, label="Noisy")
        plt.plot(t, power_spec_no_max, 'b*', LineWidth=2, label="Noisy")

        plt.plot(t[0], t[-1])
        plt.xlabel("Power Spectrum no max [Hz]")
        plt.ylabel("Power")
        plt.title("Power Spectrum Analysis - removed big max {} - Max: {} - Threshold: {}".format(max_fft_spectr_value,
                                                                                                  numpy.nanmax(
                                                                                                      power_spec_no_max),
                                                                                                  threshold_remaining_values))
        plt.show()
    else:
        fig, axs = plt.subplots(3, 1)

        plt.sca(axs[0])
        plt.plot(t, data_block, color='c', LineWidth=3, label="raw data")
        plt.plot(t, data_block_with_lin_interpol_where_nan, color='k', LineWidth=1, linestyle='--', label='lin interp')
        plt.plot(t, data_block_with_cubic_interp_where_nan(t), color='r', LineWidth=1, linestyle=':',label='cubic interp')
        plt.plot(t, ffilt, color="k", LineWidth=2, label='FFT Filtered')


        plt.plot(t, good_qual, 'go', label="Good Quality")
        plt.plot(t, okay_qual, 'yo', label="Okay Quality")
        plt.plot(t, bad_qual, 'o', color='orange', label="Bad Quality")
        plt.plot(t, really_bad_qual, 'ro', label="Really Bad Quality")


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
        plt.title("Power Spectrum Analyses - Max: {} - Threshold: {}".format(max_fft_spectr_value, numpy.nanmean(power_spectrum)))

        plt.sca(axs[2])
        plt.plot(t, power_spec_no_max, color="c", LineWidth=2, label="Noisy")
        plt.plot(t, power_spec_no_max, 'b*', LineWidth=2, label="Noisy")
        plt.plot(t[0], t[-1])
        plt.xlabel("Power Spectrum no max [Hz]")
        plt.ylabel("Power")
        plt.title("Power Spectrum Analysis - removed big max {} - Max: {} - Threshold: {}".format(max_fft_spectr_value, numpy.nanmax(power_spec_no_max), threshold_remaining_values))
        plt.show()

def interp_2d_data(data_block, window_size):

    """
    Interpolates linear between nan values in the datablock of the satdata
    e.g [350, 360, nan, nan, 390  400, nan, nan, 380] --> [342, 360, 370 ,380, 390, 400, 395, 390, 380]
    Parameters
    ----------
    data_block      ndim array [window, 2400**2]
    window_size     integer - describes how many epochs represent the data_block

    Returns         numpy ndim array
    -------

    """

    print("Executeing LINEAR data interpolation")

    #data_block = data_block[:,:4,:4]

    data_block_mat = data_block.reshape(window_size, data_block.shape[1]*data_block.shape[2])
    # orig_data_block_rows = data_block.shape[1]
    # orig_data_block_cols = data_block.shape[2]

    # print("data_block_mat shape: ", data_block_mat.shape)
    # print(data_block_mat)
    # data_block_v = data_block_mat.T.reshape(window_size*orig_data_block_cols*orig_data_block_rows)
    # print("data_block_v shape: ", data_block_v.shape)
    # print(data_block_v)
    #
    # data_block_v_nan = numpy.isfinite(data_block_v)
    # data_block_t = numpy.arange(0, len(data_block_v_nan))
    # data_block_v_interp = numpy.round(numpy.interp(data_block_t, data_block_t[data_block_v_nan], data_block_v[data_block_v_nan]))
    #
    # data_block_mat_interp = data_block_v_interp.reshape(orig_data_block_cols*orig_data_block_rows, window_size).T
    #
    # print("datablock v interp: ", data_block_v_interp)
    # print("databock interp shape: ", data_block_mat_interp.shape)
    # print(data_block_mat_interp)

    time_start = time.time()
    print("Start for loop")
    total = 2400**2
    cou = 0
    for i in range(0,data_block.shape[1]*data_block.shape[2],1):
        try:
            #print(" %d of %d left"%(total - i, total))

            #print(data_block_mat[:,i])
            data_block_v_nan = numpy.isfinite(data_block_mat[:,i])
            data_block_t = numpy.arange(0, len(data_block_v_nan))
            data_block_v_interp = numpy.round(numpy.interp(data_block_t, data_block_t[data_block_v_nan], data_block_mat[:,i][data_block_v_nan]))
            #print(data_block_v_interp)
            #print("\n")
            cou += 1
        except Exception as failure:
            #print(failure.__traceback__.)
            #print(cou)
            #print(data_block_mat[:,i])
            cou += 1
    print("Finished loop after ", time.time() - time_start, " seconds")
    print("cou: ", cou)
    return None



def init_data_block_numpy(sg_window, band, in_dir_qs, in_dir_tf, tile, list_qual, list_data, device, master_raster_info, fit_nr, name_weights_addition):

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

    data_block = numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])
    qual_block = numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])

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

    return data_block, qual_block, fitted_raster_band_name


def additional_stat_info_raster_numpy(data_block, qual_block, sg_window, device, half_window, weights, center):
    """
    Creates A Matrix and matrizzes that count the number of usefull observation of the time window
    Parameters
    ----------
    data_block
    qual_block
    sg_window
    device
    half_window
    weights
    center

    Returns
    -------

    """
    print("# Processing Numpy")
    print("# No Need of Device : ", device)

    qual_block[qual_block == 0] = weights[0]
    qual_block[qual_block == 1] = weights[1]
    qual_block[qual_block == 2] = weights[2]
    qual_block[qual_block == 3] = weights[3]

    noup_array = numpy.where(qual_block == 255, 0, 1)      # exchange NaN Value 255 with 0

    qual_block[qual_block == 255] = numpy.nan  # set to 0 so in the ausgleich the nan -> zero convertion is not needed
    #                                                     # nan will be replaced by zeros so this is a shortcut to avoid that transformation
    data_block[data_block == 32767] = numpy.nan

    # # data ini to count how many data epochs are to the left and to the right of the center epoch etc
    l_max = numpy.ones([sg_window, qual_block.shape[1], qual_block.shape[2]]) * numpy.nanmax(data_block, axis=0)
    l_min = numpy.ones([sg_window, qual_block.shape[1], qual_block.shape[2]]) * numpy.nanmin(data_block, axis=0)

    print("# l max: ", numpy.nanmax(l_max))
    print("# l min: ", numpy.nanmin(l_min))

    noup_l = numpy.sum(noup_array[0:center, :, :], axis=0).reshape(noup_array.shape[1]*noup_array.shape[2])  # numbers of used epochs on the left side
    noup_r = numpy.sum(noup_array[center + 1:, :, :], axis=0).reshape(noup_array.shape[1]*noup_array.shape[2])  # numbers of used epochs on the right side
    noup_c = noup_array[center].reshape(noup_array.shape[1]*noup_array.shape[2])  # numbers of used epochs on the center epoch


    print("\n# Dim Check for NOUP:")
    print("\n# noup_l: ", noup_l.shape)
    print("# noup_r: ", noup_r.shape)
    print("# noup_c: ", noup_c.shape)

    n = numpy.sum(noup_array, axis=0).reshape(noup_array.shape[1]*noup_array.shape[2])  # count all pixels that are used on the entire sg_window for the least square

    print("# n: ", n.shape)
    print("# Numbers of Observations check R: \n", noup_r[:4])
    print("# Min {} - Max {} - Median {}".format(noup_r.min(), noup_r.max(), numpy.nanmedian(noup_r)))
    print("# Numbers of Observations check L: \n", noup_l[:4])
    print("# Min {} - Max {} - Median {}".format(noup_l.min(), noup_l.max(), numpy.nanmedian(noup_l)))
    print("# Numbers of Observations check C: \n", noup_c[:4])
    print("# Min {} - Max {} - Median {}".format(noup_c.min(), noup_c.max(), numpy.nanmedian(noup_c)))
    print("# Numbers of Observations check N: \n", n[:4])
    print("# Min {} - Max {} - Median {}".format(n.min(), n.max(), numpy.nanmedian(n)))


    ids_for_lin_fit = numpy.concatenate(
                                        (numpy.where(noup_l <= 3),
                                         numpy.where(noup_r <= 3),
                                         numpy.where(noup_c <= 0),
                                         numpy.where(n <= half_window)),
                                        axis=1)
    iv = numpy.unique(ids_for_lin_fit)  # ids sind gescheckt und passen
    print("# IV.shape: ", iv.shape)
    return data_block, qual_block, noup_array,  iv, l_max, l_min


def update_data_block_numpy(data_block, qual_block, noup_array, in_dir_tf, in_dir_qs, tile, list_data, list_qual, sg_window, center, half_window, fit_nr, ts, weights, name_weights_addition):

    # update datablock
    # -----------------
    data_block[0:-1, :, :] = data_block[1:, :, :]
    print("# UPDATE Ras Data File: ", list_data[sg_window-1 + ts])

    ras_data_new = gdal.Open(os.path.join(in_dir_tf, tile, list_data[sg_window-1 + ts])).ReadAsArray()
    data_block[sg_window-1, :, :] = numpy.where(ras_data_new == 32767, numpy.nan, ras_data_new)

    # update noup_array
    # ------------------
    noup_array[0:-1, :, :] = noup_array[1:, :, :]

    # update qualblock
    # ----------------
    qual_block[0:-1, :, :] = qual_block[1:, :, :]
    print("# UPDATE Qual Data File: ", list_qual[sg_window - 1 + ts])

    qual_data_new = gdal.Open(os.path.join(in_dir_qs, tile, list_qual[sg_window-1 + ts])).ReadAsArray()

    # update new noup_array epoch
    noup_array[-1, :, :] = numpy.where(qual_data_new == 255, 0, 1)

    # update weights
    qual_data_new = numpy.where(qual_data_new == 0, weights[0], qual_data_new)
    qual_data_new = numpy.where(qual_data_new == 1, weights[1], qual_data_new)
    qual_data_new = numpy.where(qual_data_new == 2, weights[2], qual_data_new)
    qual_data_new = numpy.where(qual_data_new == 3, weights[3], qual_data_new)

    qual_block[sg_window - 1, :, :] = numpy.where(qual_data_new == 255, numpy.nan, qual_data_new)

    # # data ini to count how many data epochs are to the left and to the right of the center epoch etc
    l_max = numpy.ones([sg_window, qual_block.shape[1], qual_block.shape[2]]) * numpy.nanmax(data_block, axis=0)
    l_min = numpy.ones([sg_window, qual_block.shape[1], qual_block.shape[2]]) * numpy.nanmin(data_block, axis=0)

    print("# l max: ", numpy.nanmax(l_max))
    print("# l min: ", numpy.nanmin(l_min))

    noup_l = numpy.sum(noup_array[0:center, :, :], axis=0).reshape(noup_array.shape[1] * noup_array.shape[2])  # numbers of used epochs on the left side
    noup_r = numpy.sum(noup_array[center + 1:, :, :], axis=0).reshape(noup_array.shape[1] * noup_array.shape[2])  # numbers of used epochs on the right side
    noup_c = noup_array[center].reshape(noup_array.shape[1] * noup_array.shape[2])  # numbers of used epochs on the center epoch

    print("\n# Dim Check for NOUP:")
    print("\n# noup_l: ", noup_l.shape)
    print("# noup_r: ", noup_r.shape)
    print("# noup_c: ", noup_c.shape)

    n = numpy.sum(noup_array, axis=0).reshape(noup_array.shape[1] * noup_array.shape[2])  # count all pixels that are used on the entire sg_window for the least square

    print("# n: ", n.shape)
    print("# Numbers of Observations check R: \n", noup_r[:4])
    print("# Min {} - Max {} - Median {}".format(noup_r.min(), noup_r.max(), numpy.nanmedian(noup_r)))
    print("# Numbers of Observations check L: \n", noup_l[:4])
    print("# Min {} - Max {} - Median {}".format(noup_l.min(), noup_l.max(), numpy.nanmedian(noup_l)))
    print("# Numbers of Observations check C: \n", noup_c[:4])
    print("# Min {} - Max {} - Median {}".format(noup_c.min(), noup_c.max(), numpy.nanmedian(noup_c)))
    print("# Numbers of Observations check N: \n", n[:4])
    print("# Min {} - Max {} - Median {}".format(n.min(), n.max(), numpy.nanmedian(n)))

    ids_for_lin_fit = numpy.concatenate(
        (numpy.where(noup_l <= 3),
         numpy.where(noup_r <= 3),
         numpy.where(noup_c <= 0),
         numpy.where(n <= half_window)),
        axis=1)
    iv = numpy.unique(ids_for_lin_fit)  # ids sind gescheckt und passen
    print("# IV.shape: ", iv.shape)


    fitted_raster_band_name = list_data[fit_nr + ts][:-4] + name_weights_addition % str(sg_window)

    return data_block, qual_block, noup_array, fitted_raster_band_name, iv, l_max, l_min



def write_fitted_raster_to_disk(fit_layer, out_dir_fit, tile, fitted_raster_band_name, master_raster_info):

    # set negative values to nan value of 32767
    fit_layer[fit_layer < 0] = 32767
    fit_layer[fit_layer > 9999] = 32767
    # write output raster
    print("- Fitlayer stats: ", fit_layer.shape)
    out_ras_name = os.path.join(out_dir_fit, tile, fitted_raster_band_name)
    print("- Writing file : ", out_ras_name)

    # master_raster_info[-1] holds the driver for the file to create
    out_ras = master_raster_info[-1].Create(out_ras_name, master_raster_info[2], master_raster_info[3], 1,
                                            gdalconst.GDT_Int16, options=['COMPRESS=LZW'])
    out_band = out_ras.GetRasterBand(1)
    out_band.WriteArray(fit_layer)
    out_band.SetNoDataValue(32767)
    out_ras.SetGeoTransform(master_raster_info[0])
    out_ras.SetProjection(master_raster_info[1])
    out_ras.FlushCache()
    del out_ras, out_band, out_ras_name

def fitl(lv, pv, xv, sg_window, iv):

    """
    # Linearer Fit, wenn zu wenige Beobachtungen im Zeitfenster vorliegen nach def. Kriterien
    # Bezeichnungen wie bei fitq

    Parameters
    ----------
    lv
    pv
    xv

    Returns
    -------

    """
    print("# Start Fit Linear ...")

    lv = lv.reshape(lv.shape[0], lv.shape[1] * lv.shape[2])
    fit = lv
    pv = pv.reshape(pv.shape[0], pv.shape[1] * pv.shape[2])
    xv = xv[:, 1].reshape(sg_window, 1)

    lv = lv[:, iv]
    pv = pv[:, iv]

    print("# lv.shape : ", lv.shape)
    print("# pv.shape : ", pv.shape)
    print("# xv.shape : ", xv.shape)

    ax0 = xv ** 0  # schneller im vergleich zu funktion ones da kein gesonderter funktionsaufruf
    ax1 = xv

    a11 = numpy.nansum(ax0 * pv * ax0, 0)
    a12 = numpy.nansum(ax0 * pv * ax1, 0)
    a22 = numpy.nansum(ax1 * pv * ax1, 0)

    det = a11 * a22 - a12 * a12

    ai11 = a22 / det
    ai12 = -a12 / det
    ai22 = a11 / det

    vx0 = numpy.nansum(ax0 * pv * lv, 0)
    vx1 = numpy.nansum(ax1 * pv * lv, 0)

    a0 = ai11 * vx0 + ai12 * vx1
    a1 = ai12 * vx0 + ai22 * vx1

    print("# shape a0: ", a0.shape)
    print("# shape a1: ", a1.shape)

    fit[:, iv] = numpy.round(a0 + a1*xv)

    return fit.reshape(sg_window, 2400,2400)


def fitq(lv, pv, xv, sg_window):
    """
    Least Square Apporach to find the parameters of a polynomial second order. input data lv is a 3d numpyarray with dim
    [sg_window, 2400, 2400]
    Parameters
    ----------
    lv - 3ddim numpy array holding sat timeseries, shape: [sg_window, 2400,2400]
    pv - 3ddim numpy array holding weights, shape: [sg_window, 2400,2400]
    xv - 2ddim numpy array (A Matrix), shape: [sg_window, 3]
    sg_window - int, describes the size of the savitzky golay filter window

    Returns
    -------

    """

    # Quadratischer Fit Input Matrix in Spalten Pixelwerte in Zeilen die Zeitinformation
    # lv ... Beobachtungsvektor = Grauwerte bei MODIS in Prozent z.B. (15, 12 ....)
    # pv ... Gewichtsvektor mit  p = 1 fuer MCD43A2 = 0 0.2 bei MCD43A2=1 (bei MODIS) in erster
    # Iteration, der bei den weiteren Iterationen entsprechend ueberschrieben wird.
    # xv ... Zeit in day of year. Damit die Integerwerte bei Quadrierung nicht zu gross werden anstatt
    # direkte doy's die Differenz zu Beginn, also beginnend mit 1 doy's
    # A [ax0, ax1, ax2] Designmatrix
    # Formeln aus
    print("# Start Fit Polynom ...")
    lv = lv.reshape(lv.shape[0], lv.shape[1] * lv.shape[2])
    pv = pv.reshape(pv.shape[0], pv.shape[1] * pv.shape[2])
    xv = xv[:, 1].reshape(sg_window, 1)

    print("# lv.shape : ", lv.shape)
    print("# pv.shape : ", pv.shape)
    print("# xv.shape : ", xv.shape)

    ax0 = xv ** 0  # Vektor Laenge = 15 alle Elemente = 1 aber nur derzeit so bei Aufruf, spaeter bei z.B.
    # Fit von Landsat Aufnahmen doy Vektor z.B. [220, 780, 820, 1600 ...]
    ax1 = xv ** 1  # Vektor Laenge = 15 [1, 2 , 3 , 4 ... 15]
    ax2 = xv ** 2  # [ 1 , 4 , 9 ... 225]

    # ATPA Normalgleichungsmatrix
    a11 = numpy.nansum(ax0 * pv * ax0, 0)
    a12 = numpy.nansum(ax0 * pv * ax1, 0)
    a13 = numpy.nansum(ax0 * pv * ax2, 0)

    a22 = numpy.nansum(ax1 * pv * ax1, 0)
    a23 = numpy.nansum(ax1 * pv * ax2, 0)
    a33 = numpy.nansum(ax2 * pv * ax2, 0)

    # Determinante (ATPA)
    det = a11 * a22 * a33 + a12 * a23 * a13 \
          + a13 * a12 * a23 - a13 * a22 * a13 \
          - a12 * a12 * a33 - a11 * a23 * a23 \

        # Invertierung (ATPA) mit: Quelle xxx mit Zitat
    # da die Inverse von A symmetrisch ueber die Hauptdiagonale ist, entspricht ai12 = ai21
    # (ATPA)-1
    ai11 = (a22 * a33 - a23 * a23) / det
    ai12 = (a13 * a23 - a12 * a33) / det
    ai13 = (a12 * a23 - a13 * a22) / det
    ai22 = (a11 * a33 - a13 * a13) / det
    ai23 = (a13 * a12 - a11 * a23) / det
    ai33 = (a11 * a22 - a12 * a12) / det

    # ATPL mit Bezeichnung vx0 fueer Vektor x0 nansum ... fuer nodata-summe
    vx0 = numpy.nansum(ax0 * pv * lv, 0)
    vx1 = numpy.nansum(ax1 * pv * lv, 0)
    vx2 = numpy.nansum(ax2 * pv * lv, 0)

    # Quotienten der quadratischen Gleichung ... bzw. Ergebnis dieser Funktion
    a0 = ai11 * vx0 + ai12 * vx1 + ai13 * vx2
    a1 = ai12 * vx0 + ai22 * vx1 + ai23 * vx2
    a2 = ai13 * vx0 + ai23 * vx1 + ai33 * vx2
    print("# shape a0: ", a0.shape)
    print("# shape a1: ", a1.shape)
    print("# shape a2: ", a2.shape)

    fit = numpy.round(a0 + a1*xv + a2*(xv**2))

    delta_lv = abs(fit - lv)
    delta_lv = numpy.where(delta_lv<1, 1, delta_lv)
    sig = numpy.nansum(delta_lv,0)
    print("# SIG.shape. ", sig.shape)
    return fit.reshape(sg_window, 2400,2400), sig.reshape(2400, 2400), delta_lv.reshape(sg_window, 2400,2400)

def fitq_numpy(lv, pv, A, sq_window):

    """
    Calculate the parameters of a polynome second degree for the handed  savitzky golay window. data is reshaped to speed up the
    estimation using least square algorithm
    Parameters
    ----------
    lv  numpy ndarray - shape [15,2400,2400] - holding the data from the satellite time series
    pv  numpy ndarray - shape [15,2400,2400] - holding the quality info from the satellite time series
    A   numpy array   - shape [15,3]         - design matrix for the least square algorithm
    sq_window int                            - size of the savitzky golay filter

    Returns list with estimated parameters of shape [5760000,3,1]
    -------

    """

    print("input lv.shape: ", lv.shape)
    print("input pv.shape: ", pv.shape)
    print("lv[15,0,0]:\n", lv[:, 0, 0])
    print("pv[15,0,0]:\n", pv[:, 0, 0])

    lv = lv.reshape(sq_window, lv.shape[1]*lv.shape[2]).T       # change to (15,5760000) and via transpose to (5760000,15)
    lv = lv.reshape(lv.shape[0], lv.shape[1], 1)                # change to (5760000,15,1) shape
    print("reshape: lv[0]: \n", lv[0], "\nshape: ", lv.shape)

    pv = pv.reshape(sq_window, pv.shape[1]*pv.shape[2]).T           # change to (15,5760000) and via transpose to (5760000,15)
    pv = pv.reshape(pv.shape[0], 1, pv.shape[1])                    # change to (5760000,1,15) shape
    print("reshape: pv[0]: \n", pv[0], "\nshape: ", pv.shape)

    ATP = numpy.multiply(A.T, pv)
    del pv
    print("Numpy - ATP: ", ATP[0].shape)
    ATPA = numpy.linalg.inv(numpy.dot(ATP, A))
    print("Numpy - ATPA: ", ATPA.shape)

    ATPL = numpy.matmul(ATP, lv)
    del lv
    x_dach = numpy.matmul(ATPA, ATPL)
    print("x_dach: shape {}".format(x_dach.shape), "\n", x_dach)
    print("a0: ", x_dach[0,0,0])
    print("")
    return None, None, None

def fit_fft(lv, pv, sq_window):

    print("Starting FFT")



    return None

if __name__ == "__main__":
    print("Utils Numpy")