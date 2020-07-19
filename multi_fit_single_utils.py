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

# ============ Functionlisting =================================

def fitq_cuda(lv, pv, xv, sg_window):
    # Quadratischer Fit Input Matrix in Spalten Pixelwerte in Zeilen die Zeitinformation
    # lv ... Beobachtungsvektor = Grauwerte bei MODIS in Prozent z.B. (15, 12 ....)
    # pv ... Gewichtsvektor mit  p = 1 fuer MCD43A2 = 0 0.2 bei MCD43A2=1 (bei MODIS) in erster
    # Iteration, der bei den weiteren Iterationen entsprechend ueberschrieben wird.
    # xv ... Zeit in day of year. Damit die Integerwerte bei Quadrierung nicht zu groß werden anstatt
    # direkte doy's die Differenz zu Beginn, also beginnend mit 1 doy's
    # A [ax0, ax1, ax2] Designmatrix
    # Formeln aus

    ax0 = torch.reshape(xv ** 0, (sg_window,1))  # Vektor Laenge = 15 alle Elemente = 1 aber nur derzeit so bei Aufruf, spaeter bei z.B.
    # Fit von Landsat Aufnahmen doy Vektor z.B. [220, 780, 820, 1600 ...]
    ax1 = torch.reshape(xv ** 1, (sg_window,1))  # Vektor Laenge = 15 [1, 2 , 3 , 4 ... 15].T
    ax2 = torch.reshape(xv ** 2, (sg_window,1))  # [ 1 , 4 , 9 ... 225].T

    # ATPA Normalgleichungsmatrix
    a11 = torch.sum(ax0 * pv * ax0, 0)
    a12 = torch.sum(ax0 * pv * ax1, 0)
    a13 = torch.sum(ax0 * pv * ax2, 0)

    a22 = torch.sum(ax1 * pv * ax1, 0)
    a23 = torch.sum(ax1 * pv * ax2, 0)
    a33 = torch.sum(ax2 * pv * ax2, 0)

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
    vx0 = torch.sum(ax0 * pv * lv, 0)
    vx1 = torch.sum(ax1 * pv * lv, 0)
    vx2 = torch.sum(ax2 * pv * lv, 0)

    # Quotienten der quadratischen Gleichung ... bzw. Ergebnis dieser Funktion
    a0 = ai11 * vx0 + ai12 * vx1 + ai13 * vx2
    a1 = ai12 * vx0 + ai22 * vx1 + ai23 * vx2
    a2 = ai13 * vx0 + ai23 * vx1 + ai33 * vx2

    return a0, a1, a2

def fitq(lv, pv, xv):
    # Quadratischer Fit Input Matrix in Spalten Pixelwerte in Zeilen die Zeitinformation
    # lv ... Beobachtungsvektor = Grauwerte bei MODIS in Prozent z.B. (15, 12 ....)
    # pv ... Gewichtsvektor mit  p = 1 fuer MCD43A2 = 0 0.2 bei MCD43A2=1 (bei MODIS) in erster
    # Iteration, der bei den weiteren Iterationen entsprechend ueberschrieben wird.
    # xv ... Zeit in day of year. Damit die Integerwerte bei Quadrierung nicht zu groß werden anstatt
    # direkte doy's die Differenz zu Beginn, also beginnend mit 1 doy's
    # A [ax0, ax1, ax2] Designmatrix
    # Formeln aus

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

    return a0, a1, a2


# Linearer Fit, wenn zu wenige Beobachtungen im Zeitfenster vorliegen nach def. Kriterien
# Bezeichnungen wie bei fitq
def fitl_cuda(lv, pv, xv):
    ax0 = xv ** 0  # schneller im vergleich zu funktion ones da kein gesonderter funktionsaufruf
    ax1 = xv

    a11 = torch.sum(ax0 * pv * ax0, 0)
    a12 = torch.sum(ax0 * pv * ax1, 0)
    a22 = torch.sum(ax1 * pv * ax1, 0)

    det = a11 * a22 - a12 * a12

    ai11 = a22 / det
    ai12 = -a12 / det
    ai22 = a11 / det

    vx0 = torch.sum(ax0 * pv * lv, 0)
    vx1 = torch.sum(ax1 * pv * lv, 0)

    a0 = ai11 * vx0 + ai12 * vx1
    a1 = ai12 * vx0 + ai22 * vx1

    return a0, a1

def fitl(lv, pv, xv):
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

    return a0, a1

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


def init_data_block(sg_window, band, in_dir_qs, in_dir_tf, tile, list_qual, list_data, device, master_raster_info):

    data_block = torch.from_numpy(numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])).to(device)
    qual_block = torch.from_numpy(numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]])).to(device)

    data_block.share_memory_()
    qual_block.share_memory_()
    for i in range(0, sg_window, 1):

        # load qual file
        try:
            qual_ras = gdal.Open(os.path.join(in_dir_qs, tile, list_qual[i]), gdal.GA_ReadOnly)

            print("load qual data for band %d: %s" % (band, list_qual[i]))
            #qual_band = qual_ras.GetRasterBand(1)
            qual_block[i, :, :] = torch.from_numpy(qual_ras.ReadAsArray()).to(device)

            del qual_ras
        except Exception as ErrorQualRasReading:
            print("# ERROR while reading quality raster:\n {}".format(ErrorQualRasReading))
        # load satellite data
        try:
            data_ras = gdal.Open(os.path.join(in_dir_tf, tile, list_data[i]), gdal.GA_ReadOnly)
            print("load sat data for band %d: %s" % (band, list_data[i]))
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