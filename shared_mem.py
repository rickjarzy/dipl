import torch.multiprocessing as mp
import torch
import numpy
#https://stackoverflow.com/questions/50735493/how-to-share-a-list-of-tensors-in-pytorch-multiprocessing

def foo(worker,tl):
    tl[worker] += (worker+1) * 1000

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA is available")
    else:
        device = torch.device("cpu")
        print("CPU is available")


    #tl = [torch.randn(2), torch.randn(3)]
    data1 = torch.randn(15).to(device)
    data2 = torch.randn(15).to(device)
    tl = [data1, data2]
    print(tl[0])
    for t in tl:
        t.share_memory_()

    print("before mp: tl=")
    print(tl)

    p0 = mp.Process(target=foo, args=(0, tl))
    p1 = mp.Process(target=foo, args=(1, tl))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print("after mp: tl=")
    print(tl)