from osgeo import gdal
import os
import glob
import threading





if __name__ == "__main__":

    input_dir_data = r"E:\MODIS_Data\tiff\MCD43A4\h18v04"
    input_dir_qm = r"E:\MODIS_Data\tiff\MCD43A2\h18v04"

    ws = os.chdir(input_dir_data)

    data_names_list = glob.glob("*.tif")
    data_bands = ["Band"+str(i) for i in range(1,8,1)]

    print("data_bands: ", data_bands)
    print("data_names: ", data_names_list)
    print("Programm ENDE")