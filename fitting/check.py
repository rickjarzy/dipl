import torch
import numpy
import os
import glob
from time import time


def calc_ausgleich():

    # hier werden die parameter eines polynom zweiten grades ermittelt
    # f(t) = a0 + a1*t + a2*t^2
    # das stimmt auch in der A matrix

    A = numpy.array(([1,1,1], [1,2,4],[1,3,9]))
    P = numpy.eye(3)
    l = numpy.random.rand(3,1)

    print(A)
    print(P)
    print(l)

    ATPA = numpy.dot(numpy.dot(A.T,P),A)
    print("ATPA. ", ATPA)

    ATPL = numpy.dot(numpy.dot(A.T,P), l)
    print("ATPL: ", ATPL)

    x_dach = numpy.dot(ATPA, ATPL)
    print("x_dach: ", x_dach)

    l_dach = numpy.dot(A, x_dach)
    print("l_dach: ", l_dach)

    


def calc_numpy():
    qual_block = numpy.ones([2400**2,3,1])

    data_block = numpy.random.rand(2400**2,3,1)

    A = numpy.array(([1,1,1], [1,2,4],[1,3,9]))

    #A_multi_dim = numpy.ones((2400*2,3,3))
    #A_multi_dim_T = numpy.ones((2400**2,3,3))

    #numpy.multiply(A_multi_dim, A, out=A_multi_dim)

    #numpy.multiply(A_multi_dim_T, A.T, out=A_multi_dim_T)
    pv = numpy.array([1, 0.7, 0.1])


    pvv = numpy.multiply(qual_block, pv.reshape(3,1))
    ATP = numpy.multiply(A.T, pvv)
    ATPA = numpy.linalg.inv(numpy.dot(ATP, A))

    print(pvv)
    print("###############")
    print("ATP : ", ATP)
    print("ATPA: ", ATPA)
    print("#################")

    #ATPA_check = numpy.dot(ATP[0],)

    print("A: shape {}\n".format(A.shape),A)
    print("pvv: shape {}\n".format(pvv.shape), pvv)
    print("ATP: shape {}\n".format(ATP.shape), ATP)
    print("ATPA: shape {}\n".format(ATPA.shape), ATPA)

    ATPL = numpy.matmul(ATP, data_block)

    print("ATPL: shape {}".format(ATPL.shape), ATPL)

    x_dach = numpy.matmul(ATPA, ATPL)
    print("x_dach: shape {}".format(x_dach.shape), x_dach)


def calc_cuda():

    window = 3
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda is available")
    else:
        device = torch.device("cpu")

    qual_block = torch.ones([2400**2,window,1]).to(device)
    data_block = torch.ones([2400**2,window,1]).to(device)                   # 5760000,3
    P = torch.ones([2400**2,window,3]).to(device)
    A = torch.tensor(([1.,1.,1.],[1.,2.,4.],[1.,3.,9.])).to(device)     # 3,3

    pv = torch.reshape(torch.tensor([1,0.7,0.1]).to(device), (3,1))     # 3,1
    pvv = torch.mul(qual_block,pv).to(device)           # elementwise multiplication - 5760000,3,1

    P_temp = torch.eye(window).to(device)                    # elementwise multiplication

    P[:,1,1] = 0.7
    P[:,2,2] = 0.1
    P = torch.mul(P,P_temp)

    ATP = torch.mul(A, pvv)
    ATPA = torch.inverse(torch.matmul(ATP,A))                          # has the ability to multiply a 2d and a 3d matrix
    print("P: \n", P[0], " - shape: ", P.shape)
    print("pv: shape {}\n".format(pv.shape), pv)
    print("pvv: shape {}\n".format(pvv.shape), pvv)
    print("A: \n", A, " - shape: ", A.shape)
    print("ATP: \n", ATP[0], " - shape: ", ATP.shape)
    print("ATPA: \n", ATPA[0], " - shape: ", ATPA.shape)

    ATPL = torch.matmul(ATP,data_block)

    print("ATPL: shape {}".format(ATPL.shape), ATPL)
    x_dach = torch.matmul(ATPA, ATPL)
    print("x_dach: shape {}".format(x_dach.shape), x_dach)

def calc_torch_cpu():

    window = 15
    if torch.cuda.is_available():
        device = torch.device("cpu")
        print("Cuda is available")
    else:
        device = torch.device("cpu")

    qual_block = torch.ones([2400**2,1,window]).to(device)
    data_block = torch.ones([2400**2,window,1]).to(device)                   # 5760000,3
    pv = torch.rand([2400**2,1,window]).to(device)
    A = torch.ones(window, 3).to(device)
    torch.arange(1, window + 1, 1, out=A[:, 1])
    torch.arange(1, window + 1, 1, out=A[:, 2])
    A[:, 2] = A[:, 2] ** 2
    print("size A: ", A.shape)
    print(A)
    print("size P: ", pv.shape)
    print(pv)
    ATP = torch.mul(A.T, pv)
    print("size ATP", ATP.shape)
    print(ATP)
    ATPA = torch.inverse(torch.matmul(ATP,A))                          # has the ability to multiply a 2d and a 3d matrix

    print("pv: shape {}\n".format(pv.shape), pv)
    print("A: \n", A, " - shape: ", A.shape)
    print("ATP: \n", ATP[0], " - shape: ", ATP.shape)
    print("ATPA: \n", ATPA[0], " - shape: ", ATPA.shape)

    ATPL = torch.matmul(ATP,data_block)

    print("ATPL: shape {}".format(ATPL.shape), ATPL)
    x_dach = torch.matmul(ATPA, ATPL)
    print("x_dach: shape {}".format(x_dach.shape), x_dach)

def is_copy(input_data):
    
    input_data.astype(numpy.float64)
    input_data[input_data==32767]=numpy.nan
    return input_data

if __name__ == "__main__":

    start = time()
    #calc_cuda()
    #calc_numpy()

    # list_tif = sorted(glob.glob("MCD43A4.*.band_1.tif"))[39:-14]
    # sg_window = 46
    # data_block = numpy.round(numpy.random.rand(15,2400,2400)*1000).astype(numpy.float64)
    # print(data_block[:,1,1])
    # data_block[data_block>750]=32767
    # print(data_block[:,1,1])


    # return_data = is_copy(numpy.copy(data_block))

    # print("danach")
    # print(data_block[:,1,1])
    # print(return_data[:,1,1])


    calc_ausgleich()

    print("Programm ENDE")



