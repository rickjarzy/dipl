import torch
import numpy


if __name__ == "__main__":

    qual_block = numpy.ones([2400**2,3,1])

    data_block = numpy.random.rand(2400**2,3,1)

    A = numpy.array(([1,1,1], [1,2,4],[1,3,9]))

    #A_multi_dim = numpy.ones((2400*2,3,3))
    #A_multi_dim_T = numpy.ones((2400**2,3,3))

    #numpy.multiply(A_multi_dim, A, out=A_multi_dim)

    #numpy.multiply(A_multi_dim_T, A.T, out=A_multi_dim_T)
    pv = numpy.array([1, 0.7, 0.1])
    pvv = numpy.array((pv,pv,pv,pv,pv)).reshape(5,3,1)

    pvv = numpy.multiply(qual_block, pv.reshape(3,1))

    print(pvv)
    print("###############")
    print("ATP : ", numpy.multiply(A.T,pv.reshape(3,1)))
    print("ATPA: ", numpy.linalg.inv(numpy.dot(numpy.multiply(A.T,pv.reshape(3,1)),A)))
    print("#################")

    ATP = numpy.multiply(A.T,pvv)
    ATPA = numpy.linalg.inv(numpy.dot(ATP,A))

    ATPL = numpy.multiply(ATP,data_block)

    #ATPA_check = numpy.dot(ATP[0],)

    print(A)
    print(pvv)

    print(ATP)
    print(ATPA)
    print("Programm ENDE")



