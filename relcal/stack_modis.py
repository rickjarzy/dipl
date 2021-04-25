"""# Stack fitted modis data to a multi band tiff"""
import os
import sys
import glob
import argparse

from typing import List, Dict
import pandas as pd
from osgeo import gdal, gdalconst


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("tile", type=str, help="which tile to stack")
    parser.add_argument("csv", type=str, help="path to csv file that contains the fitted products")

    parser.add_argument("-d", type=str, help="path to directory where the modis tiles are stored",
                        default=r"E:\MODIS_Data\v6\fitted")
    parser.add_argument("-o", type=str, help="path to directory where the modis stacks will be stored",
                        default=r"E:\MODIS_Data\v6\stacked")

    args: argparse.Namespace = parser.parse_args()
    print(args)

    tile: str = args.tile
    print("Processing tile: ", tile)

    modis_path_dir: str = os.path.join(args.d, tile)

    out_put_path: str = os.path.join(args.o, tile)

    fitting_types: List[str] = pd.read_csv(args.csv, delimiter=";").keys().to_list()[:-1]
    print(fitting_types)
    os.chdir(modis_path_dir)

    modis_bands: List[str] = [2, 3, 4, 5, 6, 7]     # band_1 is the master list

    stack_dict: Dict[str, any] = {}

    for fit in fitting_types:

        out_put_dir: str = os.path.join(out_put_path, fit.replace(".", ""))
        out_put_dir = out_put_dir.replace(".", "_")
        if os.path.isdir(out_put_dir):
            print("output dir exists: ", out_put_dir)

        else:
            print("output dir does not exist", out_put_dir)
            os.makedirs(out_put_dir)

        search_string: str = "*.band_%d.%s.tif" % (1, fit)

        print(search_string)

        master_list: List[str] = sorted(glob.glob(search_string))

        print("len master_list: ", len(master_list))
        print("first_element: ", master_list[0])

        for filename_epoch_band1 in master_list:

            output_name_tiles: List[str] = filename_epoch_band1.split(".")
            file_name_epoche_part: List[str] = output_name_tiles[1]

            # access first band here to get out meta data

            gdal_obj_band_1 = gdal.Open(os.path.join(modis_path_dir, filename_epoch_band1), gdal.GA_ReadOnly)
            geo_trafo: List[str] = gdal_obj_band_1.GetGeoTransform()
            projection:List[str] = gdal_obj_band_1.GetProjection()

            block_size_x: int = gdal_obj_band_1.RasterXSize
            block_size_y: int = gdal_obj_band_1.RasterYSize

            band_1_data = gdal_obj_band_1.GetRasterBand(1).ReadAsArray()

            driver = gdal_obj_band_1.GetDriver()

            # create output stack here
            output_name_tiles.pop(4)
            name_parts = output_name_tiles[:4] + [fit.replace(".", ""), "tif"]
            output_raster_name = ".".join(name_parts)
            output_raster_path = os.path.join(out_put_dir, output_raster_name)

            band_cou = 1

            print("# output path and name: ", output_raster_path)
            out_ras = driver.Create(output_raster_path, block_size_x, block_size_y, 7,
                                                    gdalconst.GDT_Int16, options=['COMPRESS=LZW'])

            out_ras.SetGeoTransform(geo_trafo)
            out_ras.SetProjection(projection)

            out_band = out_ras.GetRasterBand(1)
            out_band.WriteArray(band_1_data)
            out_band.SetNoDataValue(32767)

            for band in modis_bands:
                band_cou += 1
                search_pattern_band: str = "*.%s.*.band_%d.%s.tif" % (file_name_epoche_part, band, fit)
                print("related bands search patterns: ", search_pattern_band)

                file_name_relating_band: List[str] = glob.glob(search_pattern_band)

                if file_name_relating_band:
                    print(file_name_relating_band)
                    gdal_obj_band_x = gdal.Open(os.path.join(modis_path_dir, file_name_relating_band[0]), gdal.GA_ReadOnly)
                    band_x_data = gdal_obj_band_x.GetRasterBand(1).ReadAsArray()
                    out_band = out_ras.GetRasterBand(band_cou)
                    out_band.WriteArray(band_x_data)
                    out_band.SetNoDataValue(32767)


                else:
                    print("Filename not found: ", file_name_relating_band)
                    sys.exit()
            out_ras.FlushCache()
            del out_ras, out_band, output_raster_path, output_raster_name






    print("Programm ENDE")
