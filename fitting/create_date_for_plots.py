from matplotlib import pyplot as plt
import numpy
from datetime import datetime, timedelta

def download_dates():

    doys = [i for i in range(1,362,8)]
    years = [i for i in range(2011,2021,1)]

    for year in years:

        if year == 2000:
            doys = doys[7:]
        elif year == 2020:
            doys = doys[:14]
        else:
            doys = [i for i in range(1,362,8)]
        for doy in doys:
            calc_date = date.fromordinal(date(year,1,1).toordinal() + doy-1)

            print("date: {}.{}.{}".format(calc_date.year, calc_date.month, calc_date.day)," - doy: ", doy)


def get_fileslist_from_loop_index(files_list_full, loop_index, sg_window):
    return files_list_full[loop_index:loop_index+sg_window]

def get_dates_from_doy(file_name_list, shp_date_info):

    dates = []
    for filename in file_name_list:
        doy_info = filename.split(".")[1]
        doy = int(doy_info[-3:])
        year = int(doy_info[1:5])
        date = datetime(year,1,1) + timedelta(doy)
        dates.append(date)
    shp_date_info["plot_dates"]={"dates":dates}
    print(date)
    return shp_date_info

if __name__ == "__main__":
    doys = list(range(1,7410,8))
    t = numpy.arange(datetime(2000,1,1), datetime(2020,4,14), timedelta(days=8)).astype(datetime)
    check = numpy.arange(datetime(2000,1,1), datetime(2020,4,15), timedelta(days=8)).astype(datetime)

    convert_doy_to_date(2000, 31)

    print("Programm ENDE")