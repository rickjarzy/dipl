import multiprocessing
import time
import os
import glob
from osgeo import gdal

def convert_hdf(root_in_dir, root_out_dir, topic, tile):
    in_dir = os.path.join(root_in_dir, topic, tile)
    out_dir = os.path.join(root_out_dir, topic, tile)
    os.chdir(in_dir)
    hdf_list = glob.glob("*hdf")
    raster_count = len(hdf_list)
    for raster in hdf_list:
        print("processing {} from tile {}".format(raster, tile))
def multi_convert()


def cpu_bound(number):
    return sum(i * i for i in range(number))

def find_sums(numbers):
    with multiprocessing.Pool() as pool:
        pool.map(cpu_bound, numbers)

if __name__ == "__main__":


    root_in_dir = r"E:\modis\v6\hdf"
    root_out_dir = r"E:\modis\v6\tiff"

    topics = ["MCD43A2", "MCD43A4"]
    kacheln = ["h18v04", "h18v03", "h19v03", "h19v04"]

    for topic in topics:
        print("processing topc: ", topic)

        for k in kacheln:
            convert_hdf(root_in_dir, root_out_dir, topic, k)














    numbers = [5_000_000 + x for x in range(20)]

    start_time = time.time()
    find_sums(numbers)
    duration = time.time() - start_time

    print(f"Duration {duration} seconds")
    print("Programm ENDE")