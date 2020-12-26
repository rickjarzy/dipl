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

def fitq_mp(lv, pv, xv, sg_window):
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
    # xv ... Zeit in day of year. Damit die Integerwerte bei Quadrierung nicht zu groß werden anstatt
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

if __name__ == "__main__":
    print("Programm ENDE")