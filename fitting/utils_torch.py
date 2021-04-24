import torch
import numpy
from osgeo import gdal
import os

# ============ Functionlisting =================================
def additional_stat_info_raster_torch(data_block, qual_block, sg_window, device, half_window, center):
    A = torch.ones(sg_window, 3).to(device)
    torch.arange(1, sg_window + 1, 1, out=A[:, 1])
    torch.arange(1, sg_window + 1, 1, out=A[:, 2])
    A[:, 2] = A[:, 2] ** 2
    A = A.type(torch.DoubleTensor)
    print("A Matrix : ", A.shape)

    qual_block[qual_block == 0] = 1
    qual_block[qual_block == 1] = 0.75
    qual_block[qual_block == 2] = 0.1
    qual_block[qual_block == 3] = 0.01

    noup_zero = torch.zeros(sg_window, qual_block.shape[1], qual_block.shape[2])  # noup = number of used epochs/pixels
    noup_ones = torch.ones(sg_window, qual_block.shape[1], qual_block.shape[2])
    print("TYPES noup_zero: {}, noup_ones: {}, qual_block: {}".format(noup_zero.shape, noup_ones.shape, qual_block.shape))
    noup_tensor = torch.where(qual_block == 255, noup_zero, noup_ones)
    #            noup_tensor[noup_tensor != 0]=1        # obsolete
    del noup_zero, noup_ones

    qual_block[qual_block == 255] = 0  # set to 0 so in the ausgleich the nan -> zero convertion is not needed
    #                                                     # nan will be replaced by zeros so this is a shortcut to avoid that transformation
    data_block[data_block == 32767] = 0

    # # data ini to count how many data epochs are to the left and to the right of the center epoch etc
    l_max = torch.ones([sg_window, qual_block.shape[1], qual_block.shape[2]]) * torch.max(data_block, dim=0).values
    l_min = torch.ones([sg_window, qual_block.shape[1], qual_block.shape[2]]) * torch.min(data_block, dim=0).values

    noup_l = torch.sum(noup_tensor[0:center], dim=0)  # numbers of used epochs on the left side
    noup_r = torch.sum(noup_tensor[center + 1:], dim=0)  # numbers of used epochs on the right side
    noup_c = noup_tensor[center]  # numbers of used epochs on the center epoch
    # noup_c = torch.reshape(noup_c, (noup_c.shape[0], 1))

    print("\nDim Check for NOUP:")
    print("\nnoup_l: ", noup_l.numpy().shape)
    print("noup_r: ", noup_r.numpy().shape)
    print("noup_c: ", noup_c.numpy().shape)

    n = torch.sum(noup_tensor, dim=0)  # count all pixels that are used on the entire sg_window for the least square
    del noup_tensor
    print("n: ", n.shape)

    ids_for_lin_fit = numpy.concatenate(
                                        (numpy.where(noup_l.numpy() <= 3),
                                         numpy.where(noup_r.numpy() <= 3),
                                         numpy.where(noup_c.numpy() <= 0),
                                         numpy.where(n.numpy() <= half_window)),
                                        )
    iv = numpy.unique(ids_for_lin_fit)  # ids sind gescheckt und passen

    return A, data_block, qual_block, noup_c, noup_r, noup_l, iv

def init_data_block_torch(sg_window, band, in_dir_qs, in_dir_tf, tile, list_qual, list_data, device, master_raster_info):

    """
    Creates a initial datablock for the modis data and returns a torch ndim array
    """

    data_block = torch.from_numpy(numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]]))
    qual_block = torch.from_numpy(numpy.zeros([sg_window, master_raster_info[2], master_raster_info[3]]))

    #data_block.share_memory_()
    #qual_block.share_memory_()
    for i in range(0, sg_window, 1):

        # load qual file
        try:
            qual_ras = gdal.Open(os.path.join(in_dir_qs, tile, list_qual[i]), gdal.GA_ReadOnly)

            print("load qual data for band %d: %s" % (band, list_qual[i]))
            #qual_band = qual_ras.GetRasterBand(1)
            qual_block[i, :, :] = torch.from_numpy(qual_ras.ReadAsArray()).to(device)

            del qual_ras
        except Exception as ErrorQualRasReading:
            print("# ERROR while reading quality raster:\n {}".format(ErrorQualRasReading))
        # load satellite data
        try:
            data_ras = gdal.Open(os.path.join(in_dir_tf, tile, list_data[i]), gdal.GA_ReadOnly)

            print("load sat data for band %d: %s" % (band, list_data[i]))
            #data_band = data_ras.GetRasterBand(1)
            data_block[i, :, :] = torch.from_numpy(data_ras.ReadAsArray()).to(device)

            del data_ras

        except Exception as ErrorRasterDataReading:
            print("# ERROR while reading satellite raster:\n {}".format(ErrorRasterDataReading))

    return data_block, qual_block




def fitq_cuda(lv, pv, A, sg_window, device):

    """
    Using Torch CUDA device
    """
    # Quadratischer Fit Input Matrix in Spalten Pixelwerte in Zeilen die Zeitinformation
    # lv ... Beobachtungsvektor = Grauwerte bei MODIS in Prozent z.B. (15, 12 ....)
    # pv ... Gewichtsvektor mit  p = 1 fuer MCD43A2 = 0 0.2 bei MCD43A2=1 (bei MODIS) in erster
    # Iteration, der bei den weiteren Iterationen entsprechend ueberschrieben wird.
    # xv ... Zeit in day of year. Damit die Integerwerte bei Quadrierung nicht zu groß werden anstatt
    # direkte doy's die Differenz zu Beginn, also beginnend mit 1 doy's
    # A [ax0, ax1, ax2] Designmatrix
    # Formeln aus
    #lv = lv.to(device)
    #pv = pv.to(device)
    #xv = xv.to(device)

    pv = torch.reshape(pv, (pv.shape[0] * pv.shape[1], 1, sg_window))  # contains instead of 255 --> 0
    lv = torch.reshape(lv, (lv.shape[0] * lv.shape[1], sg_window, 1))  # contains instead of 32767 --> 0

    print("inside fit q\nlv.shape: ", lv.shape)
    print("pv.shape: ", pv.shape)
    print("A: ", A , " - ", A.shape)
    ATP = torch.mul(A, pv)
    print("ATP: ", ATP, " - ", ATP.shape)
    # stopped development - torch is not able to handle a singularity of a matrix - it crashes instead of marking it NaN
    #return a0, a1, a2


def fitq_cpu(lv, pv, A, sg_window):
    """
    Using Torch on CPU
    """
    pv = torch.reshape(pv, (pv.shape[0] * pv.shape[1], 1, sg_window))  # contains instead of 255 --> 0
    lv = torch.reshape(lv, (lv.shape[0] * lv.shape[1], sg_window, 1))  # contains instead of 32767 --> 0

    print("inside fit q\nlv.shape: ", lv.shape)
    print("pv.shape: ", pv.shape, " - type: ", pv.type())
    print("A: ", A , " - ", A.shape, " - type: ", A.type())

    ATP = torch.mul(A.T, pv)
    print("ATP: ", ATP[0], " - ", ATP.shape, " - type: ", ATP.type())
    ATPA = torch.inverse(torch.matmul(ATP, A))
    print("ATPA: ", ATPA, " - ", ATPA.shape)
    ATPL = torch.matmul(ATP, lv)
    del ATP
    x_dach = torch.matmul(ATPA, ATPL)
    print("x_dach: ", x_dach)

    return None, None, None


def fitl_cuda(lv, pv, xv, sg_window):

    """
    using Torch
    """
    pv = torch.reshape(pv, (pv.shape[0] * pv.shape[1], 1, sg_window))  # contains instead of 255 --> 0
    lv = torch.reshape(lv, (lv.shape[0] * lv.shape[1], sg_window, 1))  # contains instead of 32767 --> 0

    # todo: check if dimensions of data and pc fit with following algorithnm

    ax0 = torch.reshape(xv ** 0, (
    sg_window, 1))  # Vektor Laenge = 15 alle Elemente = 1 aber nur derzeit so bei Aufruf, spaeter bei z.B.
    # Fit von Landsat Aufnahmen doy Vektor z.B. [220, 780, 820, 1600 ...]
    ax1 = torch.reshape(xv ** 1, (sg_window, 1))  # Vektor Laenge = 15 [1, 2 , 3 , 4 ... 15].T

    a11 = torch.sum(ax0 * pv * ax0, 0)
    a12 = torch.sum(ax0 * pv * ax1, 0)
    a22 = torch.sum(ax1 * pv * ax1, 0)

    det = a11 * a22 - a12 * a12

    ai11 = a22 / det
    ai12 = -a12 / det
    ai22 = a11 / det

    vx0 = torch.sum(ax0 * pv * lv, 0)
    vx1 = torch.sum(ax1 * pv * lv, 0)

    a0 = ai11 * vx0 + ai12 * vx1
    a1 = ai12 * vx0 + ai22 * vx1

    return a0, a1


if __name__ == "__main__":
    print("")