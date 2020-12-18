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

    reference_to_data_block = numpy.ndarray(input_info["dim"], dtype=numpy.int16, buffer=existing_shm.buf)[:, input_info["from"]:input_info["to"], :]

    print("ref data dtype: ", reference_to_data_block.dtype)
    print("\n")
    data_mat = reference_to_data_block.reshape(reference_to_data_block.shape[0], reference_to_data_block.shape[1]*reference_to_data_block.shape[2])
    print(data_mat.shape)
    print(data_mat)

    # for i in range(0, data_mat.shape[1], 1):
    #
    #     data_mat_v_nan = numpy.isfinite(data_mat[:,i])
    #     data_mat_v_t = numpy.arange(0, len(data_mat_v_nan),1)
    #
    #     try:
    #
    #         data_mat_v_interp = numpy.round(numpy.interp(data_mat_v_t, data_mat_v_t[data_mat_v_nan], data_mat[:,i][data_mat_v_nan]))
    #
    #         if i == 0:
    #             print("i == 0")
    #             print("data mat: ", data_mat[:, i])
    #             print("data_mat_v_interp", data_mat_v_interp)
    #             data_mat[:, i] = data_mat_v_interp
    #             print("data mat:", data_mat[:, i])
    #             continue
    #
    #
    #     except:
    #         print("process_nr: ", input_info["process_nr"])
    #         print(data_mat[:,i])

    #reference_to_data_block[:, input_info["from"]:input_info["to"], :] = data_mat.reshape(orig_time, orig_rows, orig_cols)
    #return {"data": data_mat.reshape(orig_time, orig_rows, orig_cols), "from":input_info["from"], "to":input_info["to"]}

def big_data(num_cores):
    col_row = 2400
    number_cores = num_cores
    number_of_rows_data_part = 2400 // number_cores
    num_of_bytes = 15*2400*2400*8
    shm = shared_memory.SharedMemory(create=True, size=15*2400*2400*8)
    data_shm = numpy.ndarray((15,2400,2400), dtype=numpy.float64, buffer=shm.buf)
    data_shm = numpy.random.rand(15,2400,2400)*100
    #data_shm = numpy.random.randint(low=0, high=32767, size=(15,2400,2400), dtype=numpy.int16)
    data_shm = numpy.where(data_shm > 60, numpy.nan, data_shm)

    return data_shm, num_of_bytes, number_of_rows_data_part, col_row, shm

def small_data(num_cores):
    col_row = 3
    number_cores = num_cores
    number_of_rows_data_part = col_row // number_cores
    num_of_bytes = 15*col_row*col_row*8
    shm = shared_memory.SharedMemory(create=True, size=15*col_row*col_row*8)
    data_shm = numpy.ndarray((15,col_row,col_row), dtype=numpy.int16, buffer=shm.buf)
    #data_shm = numpy.random.rand(15,col_row,col_row)*100
    data_shm[:] = numpy.random.randint(low=0, high=32767, size=(15,col_row,col_row), dtype=numpy.int16)
    #data_shm = numpy.where(data_shm > 60, numpy.nan, data_shm)
    print("RAND INT DTYPE: ", data_shm.dtype)

    return data_shm, num_of_bytes, number_of_rows_data_part, col_row, shm

if __name__ == "__main__":
    number_cores = 4-1
    #data_shm, num_of_bytes, number_of_rows_data_part, col_row, shm = big_data(number_cores)
    data_shm, num_of_bytes, number_of_rows_data_part, col_row, shm = small_data(number_cores)


    print("data shape: ", data_shm.shape)
    print("data shape dtype: ", data_shm.dtype)

    data_shm[0] = data_shm[0]**0
    print(data_shm)
    job_list_with_data_indizes = []
    cou = 0

    compare_buffer = []
    # calculate indizes to split data:
    for part in range(0, col_row, number_of_rows_data_part):
        print("\nfrom part: %d to part: %d" % (part, part+number_of_rows_data_part))
        print("data shape: ", data_shm[:, part:part+number_of_rows_data_part, :].shape)
        print(part)

        info_dict = {"from": part, "to": part + number_of_rows_data_part, "process_nr": cou, "shm": shm, "dim": (15, col_row, col_row),
                     "num_of_bytes": num_of_bytes}

        job_list_with_data_indizes.append(info_dict)

        cou +=1

    print("Start timeing")
    start_time = time.time()
    multi_linear_interpolation(job_list_with_data_indizes)
    print("finished in ", time.time()-start_time, " seconds ")

    print(data_shm)

    print("shared memory name: ", shm.name)




