import multiprocessing
import time
import os
import glob
from osgeo import gdal


def convert_hdf(root_in_dir):
    in_dir = root_in_dir
    out_dir = in_dir.split("\\")
    out_dir[3] = "tiff"
    out_dir = r"\\".join(out_dir)


    os.chdir(in_dir)
    hdf_list = glob.glob("*hdf")
    raster_count = len(hdf_list)
    for raster in hdf_list:
        print("processing {} to out_dir: {}".format(raster, out_dir))

        break


def multi_convert(jobs_list):
    for root in jobs_list:
        convert_hdf(root)
        break
    # with multiprocessing.Pool() as pool:
    #     pool.map(convert_hdf, jobs_list)


def cpu_bound(number):
    return sum(i * i for i in range(number))

def find_sums(numbers):
    with multiprocessing.Pool() as pool:
        pool.map(cpu_bound, numbers)

if __name__ == "__main__":

    root_in_dir = r"R:\modis\v6\hdf"


    topics = ["MCD43A2", "MCD43A4"]
    kacheln = ["h18v04", "h18v03", "h19v03", "h19v04"]
    job_list = []

    for topic in topics:
        print("processing topic: ", topic)

        for k in kacheln:
            job_list.append(os.path.join(root_in_dir, topic, k))

    print(job_list)
    out_dir = job_list[0].split("\\")
    out_dir[3] = "tiff"
    print(job_list[0].split("\\"))
    print(r"\\".join(out_dir))

    multi_convert(job_list)



    numbers = [5_000_000 + x for x in range(20)]

    start_time = time.time()
    find_sums(numbers)
    duration = time.time() - start_time

    print(f"Duration {duration} seconds")
    print("Programm ENDE")