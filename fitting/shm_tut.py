import numpy
from multiprocessing import shared_memory
import multiprocessing
import time


def multi_linear_interpolation(job_list):

    with multiprocessing.Pool() as pool:
        pool.map(multi_lin_interp, job_list)

def multi_lin_interp(input_info):
    print("inpuft_info", input_info)
    print("\nspawn process nr : ", input_info["process_nr"])
    existing_shm = shared_memory.SharedMemory(name=input_info["shm"].name)

    # get data to process out of buffer
    reference_to_data_block = numpy.ndarray(input_info["dim"], dtype=numpy.int16, buffer=existing_shm.buf)[:, input_info["from"]:input_info["to"], :]

    # strore orig time, cols and row information - needed for reshaping
    orig_time = reference_to_data_block.shape[0]
    orig_rows = reference_to_data_block.shape[1]
    orig_cols = reference_to_data_block.shape[2]

    print("ref data dtype: ", reference_to_data_block.dtype)
    print("\n")

    # reshape data from buffer to 2d matrix with the time as y coords and x as the values
    data_mat = reference_to_data_block.reshape(reference_to_data_block.shape[0], reference_to_data_block.shape[1]*reference_to_data_block.shape[2])
    print(data_mat.shape)

    # create some missing data --> transforms dataytpe to floa64!!!
    data_mat = numpy.where(data_mat > 25000, numpy.nan, data_mat)

    # iter through
    for i in range(0, data_mat.shape[1], 1):

        data_mat_v_nan = numpy.isfinite(data_mat[:, i])
        data_mat_v_t = numpy.arange(0, len(data_mat_v_nan), 1)

        try:

            data_mat_v_interp = numpy.round(numpy.interp(data_mat_v_t, data_mat_v_t[data_mat_v_nan], data_mat[:,i][data_mat_v_nan]))

            if i == 0:
                print("i == 0")
                print("data mat: ", data_mat[:, i])
                print("data_mat_v_interp", data_mat_v_interp)
                data_mat[:, i] = data_mat_v_interp
                # print("data mat:", data_mat[:, i])
                # print("data_mat.dtype: ", data_mat.dtype)
                # print("data_mat_interp.dtype: ", data_mat_v_interp.dtype)
                data_mat_v_interp = numpy.round(data_mat_v_interp).astype(numpy.int16)
                # print("\ntransfrom to int16: ", data_mat_v_interp)
                # print("\ndata_mat_interp.dtype: ", data_mat_v_interp.dtype)

                continue

            data_mat[:, i] = data_mat_v_interp
        except:
            print("process_nr: ", input_info["process_nr"])
            print(data_mat[:,i])

    reference_to_data_block[:] = numpy.round(data_mat.reshape(orig_time, orig_rows, orig_cols)).astype(numpy.int16)
    #return {"data": data_mat.reshape(orig_time, orig_rows, orig_cols), "from":input_info["from"], "to":input_info["to"]}



def small_data(num_cores):
    col_row = 2400
    number_cores = num_cores
    number_of_rows_data_part = col_row // number_cores
    num_of_bytes = 15*col_row*col_row*8
    shm = shared_memory.SharedMemory(create=True, size=15*col_row*col_row*8)
    data_shm = numpy.ndarray((15,col_row,col_row), dtype=numpy.int16, buffer=shm.buf)
    #data_shm = numpy.random.rand(15,col_row,col_row)*100
    data_shm[:] = numpy.random.randint(low=0, high=32767, size=(15,col_row,col_row), dtype=numpy.int16)
    print("RAND INT DTYPE: ", data_shm.dtype)

    return data_shm, num_of_bytes, number_of_rows_data_part, col_row, shm

if __name__ == "__main__":
    number_cores = 3
    data_shm, num_of_bytes, number_of_rows_data_part, col_row, shm = small_data(number_cores)


    print("data shape: ", data_shm.shape)
    print("data shape dtype: ", data_shm.dtype)

    job_list_with_data_indizes = []
    cou = 0

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

    print(data_shm[:,0,0])
    print(data_shm[:, 800, 0])
    print(data_shm[:, 1600, 0])

    print("shared memory name: ", shm.name)
