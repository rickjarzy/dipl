from osgeo import gdal


def _show_supported_drivers() -> None:
    driver_list = []
    for i in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(i)
        driver_list.append(driver.GetDescription())

        # list comprehension
        driver_list = [gdal.GetDriver(i).GetDescription() for i in range(gdal.GetDriverCount())]

        # to get name as string
        print("name: ", gdal.GetDriver(i).ShortName)


if __name__ == "__main__":

    _show_supported_drivers()

    driver = gdal.GetDriverByName("TileDB")
    print("driver: ", driver)
