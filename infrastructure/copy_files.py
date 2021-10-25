import sys
import os
import shutil
import glob


if __name__ == "__main__":

    copy_from = r"E:\MODIS_Data\v6\fitted\h18v04"
    copy_to = r"D:\MODIS_Data\v6\fitted\h18v04"
    
    year_range = ['*A' + str(year) + "*.tif" for year in range(2012,2021,1)]

    os.chdir(copy_from)
    sum_files = 0
    for year_info in year_range:
        list_to_copy = glob.glob(year_info)
        print("copy year: ", year_info)
        for file in list_to_copy:

            shutil.copy2(
                os.path.join(copy_from, file),
                os.path.join(copy_to, file)
            )

        sum_files += len(list_to_copy)
    print(year_range)
    print(sum_files)
    print("Programm ENDE")