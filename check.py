import torch
import numpy
from time import time

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda is available")
    else:
        device = torch.device("cpu")

    qual_block = torch.ones([2400**2,3,1]).to(device)
    data_block = torch.ones([2400**2,3,1]).to(device)                   # 5760000,3
    P = torch.ones([2400,3,3]).to(device)
    A = torch.tensor(([1.,1.,1.],[1.,2.,4.],[1.,3.,9.])).to(device)     # 3,3

    pv = torch.reshape(torch.tensor([1,0.7,0.1]).to(device), (3,1))     # 3,1
    pvv = torch.mul(qual_block,pv).to(device)           # elementwise multiplication - 5760000,3,1


    P_temp = torch.eye(3).to(device)                    # elementwise multiplication

    P[:,1,1] = 0.7
    P[:,2,2] = 0.1
    P = torch.mul(P,P_temp)

    ATP = torch.mul(A.T, pvv)
    ATPA = torch.inverse(torch.matmul(ATP,A))                          # has the ability to multiply a 2d and a 3d matrix
    print("P: \n", P[0], " - shape: ", P.shape)
    print("A: \n", A, " - shape: ", A.shape)
    print("ATP: \n", ATP[0], " - shape: ", ATP.shape)
    print("ATPA: \n", ATPA[0], " - shape: ", ATPA.shape)

    ATPL = torch.matmul(ATP,data_block)

    print("ATPL: shape {}".format(ATPL.shape), ATPL)
    x_dach = torch.matmul(ATPA, ATPL)
    print("x_dach: shape {}".format(x_dach.shape), x_dach)

if __name__ == "__main__":

    start = time()
    calc_cuda()
    print("time elapsed: ", time()-start, " [sec]")
    print("Programm ENDE")



