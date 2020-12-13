import numpy
from multiprocessing import shared_memory
import multiprocessing
import time


def multi_linear_interpolation(job_list):

    with multiprocessing.Pool() as pool:
        pool.map(function_to_call, job_list)

def function_to_call(input_info):

    print("spawn process nr : ", input_info["process_nr"])

    reference_to_data_block = numpy.ndarray(input_info["dim"], dtype=numpy.int16, buffer=input_info["shm"].buf)

    print("dim reference block: ", reference_to_data_block.shape)
    data_mat = input_info["data"].reshape(input_info["data"].shape[0], input_info["data"].shape[1]*input_info["data"].shape[2])

    for i in range(0, data_mat.shape[1], 1):

        data_mat_v_nan = numpy.isfinite(data_mat[:,i])
        data_mat_v_t = numpy.arange(0, len(data_mat_v_nan),1)

        try:
            if False in data_mat_v_nan:

                data_mat_v_interp = numpy.round(numpy.interp(data_mat_v_t, data_mat_v_t[data_mat_v_nan], data_mat[:,i][data_mat_v_nan]))
        except:
            print("process_nr: ", input_info["process_nr"])
            print(data_mat[:,i])









if __name__ == "__main__":
    number_cores = 4-1
    number_of_rows_data_part = 2400 // number_cores
    num_of_bytes = 15*2400*2400*8
    shm = shared_memory.SharedMemory(create=True, size=15*2400*2400*8)
    data_shm = numpy.ndarray((15,2400,2400), dtype=numpy.int16, buffer=shm.buf)
    #data_shm = numpy.random.rand((15,2400,2400), dtype=numpy.int16, buffer=shm.buf)
    data_shm = numpy.random.randint(low=0, high=32767, size=(15,2400,2400), dtype=numpy.int16)
    data_shm = numpy.where(data_shm > 30000, numpy.nan, data_shm)

    print("data shape: ", data_shm.shape)
    job_list_with_data_indizes=[]
    cou = 0

    # calculate indizes to split data:
    for part in range(0, 2400, number_of_rows_data_part):
        print("\nfrom part: %d to part: %d" % (part, part+number_of_rows_data_part))
        print("data shape: ", data_shm[:, part:part+number_of_rows_data_part, :].shape)
        print(part)
        info_dict = {"from": part, "to": part + number_of_rows_data_part, "process_nr": cou, "shm": shm, "dim": (15, 2400, 2400),
                     "num_of_bytes": num_of_bytes, "data": data_shm[:,part:part+number_of_rows_data_part, :]}

        job_list_with_data_indizes.append(info_dict)

        cou +=1

    print("Start timeing")
    start_time = time.time()
    multi_linear_interpolation(job_list_with_data_indizes)
    print("finished in ", time.time()-start_time, " seconds ")








    print("shared memory name: ", shm.name)





