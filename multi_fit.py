from __future__ import print_function
import torch
import numpy



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    z = [i for i in range(1,6,1)]
    print("type z: ", type(z))
    z = torch.tensor(z)
    print("type z: ", type(z))
    x = torch.ones(5,5)*5
    y = torch.ones(5,5)
    y = y.to(device)
    print("type x: ", type(x))
    print("type y: ", type(y))

    z = z.to(device)
    x = x.to(device)

    print("type x: ", type(x))
    print("type y: ", type(y))

    print(z)
    print(x)
    print(y)


    print(torch.mm(x,y))





    print("Programm ENDE")