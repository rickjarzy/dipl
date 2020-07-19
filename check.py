import torch
import numpy


if __name__ == "__main__":

    A = torch.ones(15,3)
    data = torch.rand(15,2400,2400)
    qual = torch.round(torch.rand(15,2400,2400)*10)

    torch.arange(1,16,1, out=A[:,1])
    torch.arange(1,16,1, out=A[:, 2])
    A[:, 2] = A[: ,2]**2


    data = torch.reshape(data, (15,2400*2400))
    qual = torch.reshape(qual, (15, 2400 * 2400))

    qual_np = numpy.random.rand(15,2400,2400)
    qual_np = qual_np.reshape(15,2400*2400)
    qual_np = numpy.where(qual_np>0.5,1, qual_np)


    print("numpy qual")
    print(qual_np)




    print("data")
    print(data)

    print("Programm ENDE")



