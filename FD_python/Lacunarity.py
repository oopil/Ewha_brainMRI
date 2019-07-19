import nrrd
import numpy as np

def lacun_own(data, bs, shape):
    # can set the sliding interval

    # slide = 1
    slide = bs // 2
    # slide = bs
    if slide == 0:
        slide = 1
    s1, s2, s3 = shape // slide
    print(s1, s2, s3, bs)
    buf_mean, buf_var, buf_lambda = [], [], []

    for i in range(s1 + 1):
        for j in range(s2 + 1):
            for k in range(s3 + 1):
                box = data[i * slide:i * slide + bs, j * slide:j * slide + bs, k * slide:k * slide + bs]
                pos = np.where(box)
                if len(pos[0]) or len(pos[1]) or len(pos[2]):
                    avg = np.mean(box)
                    var = np.var(box)

                    buf_mean.append(avg)
                    buf_var.append(var)
                    buf_lambda.append(var / (avg * avg))

    # print( buf_lambda)
    return buf_mean, buf_var, buf_lambda

def box_count(data, bs, shape):
    slide = bs
    s1, s2, s3 = shape//bs
    print(s1, s2, s3, bs)
    box_count = 0

    for i in range(s1 + 1):
        for j in range(s2 + 1):
            for k in range(s3 + 1):
                box = data[i * slide:i * slide + bs, j * slide:j * slide + bs, k * slide:k * slide + bs]
                pos = np.where(box)
                if len(pos[0]) or len(pos[1]) or len(pos[2]):
                    box_count += 1

    return box_count

def box_count_FD(data):
    shape = np.array(data.shape)
    print('<< compute box counting >>')
    bs_list = [2 ** i for i in range(0, 10)]
    box_count_result = np.zeros_like(bs_list)
    # box counting
    for i, bs in enumerate(bs_list):
        cnt = box_count(data, bs, shape)
        box_count_result[i] = cnt
    print(box_count_result)
    return box_count_result

def lacunarity(data):
    shape = np.array(data.shape)
    print('<< compute lacunarity >>')
    bs_list = [2**i for i in range(1, 10)]
    box_count = np.zeros_like(bs_list)
    lac = []
    print(shape)
    print(bs_list)
    print(box_count)
    for bs in bs_list:
        buf_mean, buf_var, buf_lambda = lacun_own(data, bs, shape)
        lac.append(np.mean(buf_lambda))
        # print(buf_lambda)
        # print(np.mean(buf_lambda))
    # print(cnt)
    print(lac)
    return lac

if __name__ == "__main__":
    filename = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
    dir_path = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/"
    # 8056.0,1241.0,254.0,57.0,23.0,8.0,2.0,2.0,1.0,1.0,1.0 => same result with my code now..
    data, header = nrrd.read(filename)
    print(data.shape)
    print(header)

    def_int = data[np.where(data)][0]  # 7
    assert np.all(data[np.where(data)] == def_int)
    data = data / def_int
    lac = lacunarity(data)
    print(np.log(lac))

    filename = "/home/soopil/Desktop/Dataset/brain_ewha/1550930_CE-label_sample.nrrd"
    data, header = nrrd.read(filename)
    assert np.all(data[np.where(data)] == def_int)
    data = data / def_int
    lac = lacunarity(data)
    print(np.log(lac))

    # assert False


    pass
