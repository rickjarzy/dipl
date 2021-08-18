from matplotlib import pyplot as plt
import numpy
from datetime import datetime, timedelta

""" 
Contains all the logic that creates the date information for the x axe labes on the plots
"""

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
        date = datetime(year-1,12,31) + timedelta(doy)
        dates.append("%d-%d-%d"%(date.year, date.month, date.day))
    shp_date_info["plot_dates"]={"dates":dates}
    print(date)
    return shp_date_info

def create_date_hub_dict():

    date_hub = {}
    date_hub_list = []
    cou = 0
    doys = [i for i in range(1,362,8)]

    for year in range(2000,2021,1):

        if year == 2000:
            doys = doys[7:]
        elif year == 2020:
            doys = doys[:14]
        else:
            doys = [i for i in range(1,362,8)]
        
        for doy in doys:
            date = datetime(year-1, 12,31)+timedelta(doy)
            
            date_hub[cou] = {"date":"%s-%s-%s" % (year, date.month, date.day), "doy": doy}
            date_hub_list.append("%s-%s-%s" % (year, date.month, date.day))
            cou += 1    

    return date_hub

def create_date_hub_list():

    
    date_hub_list = []
    cou = 0
    doys = [i for i in range(1,362,8)]

    for year in range(2000,2021,1):

        if year == 2000:
            doys = doys[7:]
        elif year == 2020:
            doys = doys[:14]
        else:
            doys = [i for i in range(1,362,8)]
        
        for doy in doys:
            date = datetime(year-1, 12,31)+timedelta(doy)
            date_hub_list.append("%s-%s-%s" % (year, date.month, date.day))
            
    return date_hub_list

def get_files_from_year_and_doy(list_data, year, doy, sg_window):

    # brauche ein dict mit year als key und passenden files als list in einem attribut
    date_hub = {}
    return 

if __name__ == "__main__":
    doys = list(range(1,7410,8))
    t = numpy.arange(datetime(2000,1,1), datetime(2020,4,14), timedelta(days=8)).astype(datetime)
    check = numpy.arange(datetime(2000,1,1), datetime(2020,4,15), timedelta(days=8)).astype(datetime)

    convert_doy_to_date(2000, 31)

    print("Programm ENDE")