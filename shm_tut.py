import numpy
from multiprocessing import shared_memory
import multiprocessing
import time


def multi_linear_interpolation(job_list):

    with multiprocessing.Pool() as pool:
        pool.map(function_to_call, job_list)

def function_to_call(input_info):

    print("\nspawn process nr : ", input_info["process_nr"])
    existing_shm = shared_memory.SharedMemory(name=input_info["shm"].name)
    reference_to_data_block = numpy.ndarray(input_info["dim"], dtype=numpy.float64, buffer=existing_shm.buf)
    orig_time = input_info["data"].shape[0]
    orig_rows = input_info["data"].shape[1]
    orig_cols = input_info["data"].shape[2]


    print(input_info["data"][:,0,0])
    data_mat = input_info["data"].reshape(input_info["data"].shape[0], input_info["data"].shape[1]*input_info["data"].shape[2])

    print("data_mat: ", data_mat[:, 0])
    print("ref_data: ", reference_to_data_block[:, input_info["from"], 0])

    reference_to_data_block[:, input_info["from"], 0] = data_mat[:,0]//2
    print("ref_data: ", reference_to_data_block[:, input_info["from"], 0])

    for i in range(0, data_mat.shape[1], 1):

        data_mat_v_nan = numpy.isfinite(data_mat[:,i])
        data_mat_v_t = numpy.arange(0, len(data_mat_v_nan),1)

        try:

            data_mat_v_interp = numpy.round(numpy.interp(data_mat_v_t, data_mat_v_t[data_mat_v_nan], data_mat[:,i][data_mat_v_nan]))

            if i == 0:
                print("i == 0")
                print("data mat: ", data_mat[:, i])
                print("data_mat_v_interp", data_mat_v_interp)
                data_mat[:, i] = data_mat_v_interp
                print("data mat:", data_mat[:, i])
                continue


        except:
            print("process_nr: ", input_info["process_nr"])
            print(data_mat[:,i])

    #reference_to_data_block[:, input_info["from"]:input_info["to"], :] = data_mat.reshape(orig_time, orig_rows, orig_cols)

    print("ref_data: ", reference_to_data_block[:, input_info["from"], 0])
    #return {"data": data_mat.reshape(orig_time, orig_rows, orig_cols), "from":input_info["from"], "to":input_info["to"]}


if __name__ == "__main__":
    number_cores = 4-1
    number_of_rows_data_part = 2400 // number_cores
    num_of_bytes = 15*2400*2400*8
    shm = shared_memory.SharedMemory(create=True, size=15*2400*2400*8)
    data_shm = numpy.ndarray((15,2400,2400), dtype=numpy.float64, buffer=shm.buf)
    data_shm = numpy.random.rand(15,2400,2400)*100
    #data_shm = numpy.random.randint(low=0, high=32767, size=(15,2400,2400), dtype=numpy.int16)
    data_shm = numpy.where(data_shm > 60, numpy.nan, data_shm)

    print("data shape: ", data_shm.shape)
    print("data shape dtype: ", data_shm.dtype)
    job_list_with_data_indizes = []
    cou = 0

    compare_buffer = []
    # calculate indizes to split data:
    for part in range(0, 2400, number_of_rows_data_part):
        print("\nfrom part: %d to part: %d" % (part, part+number_of_rows_data_part))
        print("data shape: ", data_shm[:, part:part+number_of_rows_data_part, :].shape)
        print(part)
        compare_buffer.append(data_shm[:, part, 0])

        info_dict = {"from": part, "to": part + number_of_rows_data_part, "process_nr": cou, "shm": shm, "dim": (15, 2400, 2400),
                     "num_of_bytes": num_of_bytes, "data": data_shm[:,part:part+number_of_rows_data_part, :]}

        job_list_with_data_indizes.append(info_dict)

        cou +=1

    print("Start timeing")
    start_time = time.time()
    multi_linear_interpolation(job_list_with_data_indizes)
    print("finished in ", time.time()-start_time, " seconds ")

    print(data_shm[:,800,0])

    print("shared memory name: ", shm.name)





