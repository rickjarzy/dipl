from osgeo import gdal, gdalconst
import numpy
import os
import glob


if __name__ == "__main__":

    data_dir = r"E:\MODIS_Data\v6\tiff_single\MCD43A4\h18v04"
    qual_dir = r"E:\MODIS_Data\v6\tiff_single\MCD43A2\h18v04"
    out_dir = r"E:\MODIS_Data\v6\tiff_single\quality_check"
    data_name = r"MCD43A4.A2000057.h18v04.006.band_4.tif"
    qual_name = r"MCD43A2.A2000057.h18v04.006.band_4.tif"

    data_path = os.path.join(data_dir, data_name)
    qual_path = os.path.join(qual_dir, qual_name)

    print("data: %s"%data_path)
    print("qual: %s"%qual_path)

    data_obj = gdal.Open(data_path, gdal.GA_ReadOnly)
    qual_obj = gdal.Open(qual_path, gdal.GA_ReadOnly)
    driver = data_obj.GetDriver()
    geo = data_obj.GetGeoTransform()
    trfo = data_obj.GetProjection()
    block_size_x = data_obj.RasterXSize
    block_size_y = data_obj.RasterYSize
    gdal_objects = [data_obj, qual_obj]

    data_stack = numpy.ones((2400,2400,2))
    print(data_stack.shape)
    cou = 0
    for i in gdal_objects:
        
        band = i.GetRasterBand(1)
        tmp = band.ReadAsArray()
        print("shape: ", tmp.shape)
        print("Get No Data value: ", band.GetNoDataValue())
        data_stack[:,:,cou] = band.ReadAsArray()        
        del band
        cou += 1
    test_array = numpy.where(((data_stack[:,:,0]==32767) & (data_stack[:,:,1]!=255)), 1, 0)
    #test_array = numpy.where((data_stack[:,:,0]==32767) , 1, 0)

    outras = driver.Create(os.path.join(out_dir, data_name[:-4]+".qul_check.tif"), block_size_x, block_size_y, 1,
                                            gdalconst.GDT_Int16, options=['COMPRESS=LZW'])
    outras.SetGeoTransform(geo)
    outras.SetProjection(trfo)
    outband = outras.GetRasterBand(1)
    outband.WriteArray(test_array)
    
    outband.SetNoDataValue(32767)
    outras.FlushCache()

    del outras, outband
    print("Programm ENDE")