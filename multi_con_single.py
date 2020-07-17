import multiprocessing
import time
import os
import glob
from osgeo import gdal, gdalconst


def convert_hdf(root_in_dir):
    in_dir = root_in_dir
    os.chdir(in_dir)
    hdf_list = glob.glob("*.hdf")

    # get geo reference
    ref_dir = in_dir.split("\\")
    topic = ref_dir[4]
    ref_dir[2]="v5"
    ref_dir[3]="reference_tiffs"
    ref_dir = r"\\".join(ref_dir)

    ref_ras = gdal.Open(os.path.join(ref_dir, "reff.tif"))
    ref_ras_geo = ref_ras.GetGeoTransform()
    ref_ras_pro = ref_ras.GetProjection()

    out_dir = in_dir.split("\\")
    out_dir[3] = "tiff_single"
    out_dir_tiles = out_dir[3:]
    #out_dir = "E:\\MODIS_Data\\v6\\" + "\\".join(out_dir_tiles)
    out_dir = "R:\\modis\\v6\\" + "\\".join(out_dir_tiles)
#    out_dir = "\\".join(out_dir)


    raster_count = len(hdf_list)
    driver_tiff = gdal.GetDriverByName("GTiff")

    if topic == "MCD43A2":
        hdf_bands = [i for i in range(11, 18, 1)]
        bytes_raster = gdalconst.GDT_Byte       # bytes 0 - 255
        no_data_value = 255

    else:
        hdf_bands = [i for i in range(7,14,1)]
        bytes_raster = gdalconst.GDT_Int16      # 16 bit data#
        no_data_value = 32767

    tif_bands = [i for i in range(1,8,1)]

    cols_hdf = ref_ras.RasterXSize
    rows_hdf = ref_ras.RasterYSize

    del ref_ras

    for raster in hdf_list:
        print("processing {} to out_dir: {}".format(raster, out_dir))


        hdf_ras = gdal.Open(raster, gdal.GA_ReadOnly)
        hdf_sub_data_sets = hdf_ras.GetSubDatasets()

        del hdf_ras


        # pull all bands from the hdf and store it in the new rasterfile
        for sat_band_counter in range(0,len(hdf_bands),1):
            print("Writing hdf band {} to tif band {} of raster {}".format(hdf_bands[sat_band_counter], tif_bands[sat_band_counter], raster[:-4] + ".tif"))

            # creating the tif file
            try:
                raster_name = ".".join(raster.split(".")[:4]) + ".band_%d" % tif_bands[sat_band_counter]
                print("FUll Out Path: ", os.path.join(str(out_dir), raster_name + ".tif"))
                tif_ras = driver_tiff.Create(os.path.join(str(out_dir), raster_name + ".tif"),
                                             xsize=cols_hdf,
                                             ysize=rows_hdf,
                                             bands=1,
                                             eType=bytes_raster)  # use no compression because the file size increased to the double size
            except Exception as create_ras_exception:
                print(create_ras_exception)

            # set projection
            tif_ras.SetProjection(ref_ras_pro)
            tif_ras.SetGeoTransform(ref_ras_geo)
            hdf_ras = gdal.Open(hdf_sub_data_sets[hdf_bands[sat_band_counter]][0])

            tif_band = tif_ras.GetRasterBand(1)
            hdf_band = hdf_ras.GetRasterBand(1)             # each subdata set in the hdf is counted as a seperate raster file with band one!


            tif_band.SetNoDataValue(no_data_value)
            hdf_rast_data = hdf_band.ReadAsArray()


            tif_band.WriteArray(hdf_rast_data)
            tif_band.FlushCache()
            del hdf_band, tif_band, tif_ras



def multi_convert(jobs_list):
    # for root in jobs_list:
    #     convert_hdf(root)
    #     break
    with multiprocessing.Pool() as pool:
        pool.map(convert_hdf, jobs_list)


def cpu_bound(number):
    return sum(i * i for i in range(number))

def find_sums(numbers):
    with multiprocessing.Pool() as pool:
        pool.map(cpu_bound, numbers)

if __name__ == "__main__":
    try:
        start_time = time.time()
        root_in_dir = r"R:\modis\v6\hdf"
        root_ref_dir = r"R:\modis\v5\reference_tiffs"


        topics = ["MCD43A2", "MCD43A4"]
        kacheln = ["h18v04", "h18v03", "h19v03", "h19v04"]
        job_list = []

        for topic in topics:
            for k in kacheln:
                job_list.append(os.path.join(root_in_dir, topic, k))
        print("JOBLIST: \n", job_list)
        multi_convert(job_list)






        # numbers = [5_000_000 + x for x in range(20)]
        # find_sums(numbers)
        duration = time.time() - start_time

        print(f"Duration {duration} seconds")
        print("Programm ENDE")
    except KeyboardInterrupt:
        print("Programm Stoped by User")
