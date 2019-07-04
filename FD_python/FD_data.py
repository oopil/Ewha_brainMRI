from scipy import stats
import numpy as np
import openpyxl
from matplotlib import pyplot as plt

def FD_merge_dataloader(xl_path):
    pass

def FD_dataloader(xl_path, sheet_name):
    xl = openpyxl.load_workbook(xl_path, read_only=True)
    ws = xl[sheet_name]
    data_excel = []
    for row in ws.rows:
        line = []
        for cell in row:
            line.append(cell.value)
        data_excel.append(line)
    data_excel = np.array(data_excel)
    print(data_excel.shape)
    low_subj_pos = np.where(data_excel[:, 2] == 0)
    high_subj_pos = np.where(data_excel[:, 2] == 1)
    low_fdim, high_fdim = [], []
    low_subj_list = [data_excel[i, 0] for i in low_subj_pos][0]
    high_subj_list = [data_excel[i, 0] for i in high_subj_pos][0]
    print(np.shape(low_subj_list))
    print(np.shape(high_subj_list))
    print(type(high_subj_list))
    result_file_name = './fd_result/fd_result_file.txt'
    fd = open(result_file_name, 'r')
    contents = fd.readlines()[1:]
    # print(contents)

    fd_list, name_list = [], []
    for e in contents:
        subj_name = e.split('/')[0]
        fd = e.split('/')[1].replace('\n', '').split(',')
        if subj_name in name_list:
            continue
        fd_list.append([subj_name, fd])
        name_list.append(subj_name)
        if subj_name == '1550930':
            print(contents.index(e))

    l_c, h_c, max_c = 0, 0, 200
    for i, e in enumerate(fd_list):
        sample = e
        subj_name = int(e[0])
        n = np.array(sample[1]).astype(np.float32)
        length = len(n)
        r = np.array([2 ** i for i in range(0, length)]).astype(np.float32)
        n_grad = np.gradient(np.log(n))
        r_grad = np.gradient(np.log(r))
        dim = np.divide(n_grad, r_grad) * (-1)
        print(len(dim), end='/')
        if subj_name in low_subj_list and l_c < max_c:
            print('low list')
            # low_fdim.append(length)
            low_fdim.append(dim[:9])
            plt.plot(np.log(r), dim, '-bo')
            l_c += 1
        elif subj_name in high_subj_list and h_c < max_c:
            print('high list')
            # high_fdim.append(length)
            high_fdim.append(dim[:9])
            plt.plot(np.log(r), dim, '-ro')
            h_c += 1
    label_high = [1 for _ in range(h_c)]
    label_low = [0 for _ in range(l_c)]
    print()
    print(h_c,l_c) # 30 and 95
    # print(label_high)
    # print(label_low)
    return high_fdim, low_fdim, label_high, label_low

def Lac_dataloader_save():
    xl_test = '/home/soopil/Desktop/Dataset/brain_ewha/Test_Meningioma_20180508.xlsx'
    xl_train = '/home/soopil/Desktop/Dataset/brain_ewha/Train_Meningioma_20180508.xlsx'
    xl = openpyxl.load_workbook(xl_train, read_only=True)
    ws = xl['20171108_New_N4ITK corrected']  # 'Sheet1 for test excel file
    data_excel = []
    for row in ws.rows:
        line = []
        for cell in row:
            line.append(cell.value)
        data_excel.append(line)
    data_excel = np.array(data_excel)
    print(data_excel.shape)
    low_subj_pos = np.where(data_excel[:, 2] == 0)
    high_subj_pos = np.where(data_excel[:, 2] == 1)
    low_fdim, high_fdim = [], []
    low_subj_list = [data_excel[i, 0] for i in low_subj_pos][0]
    high_subj_list = [data_excel[i, 0] for i in high_subj_pos][0]
    print(np.shape(low_subj_list))
    print(np.shape(high_subj_list))
    print(type(high_subj_list))
    result_file_name = './fd_result/Lac_result_v2.txt' # Lac_result_v2.txt
    fd = open(result_file_name, 'r')
    contents = fd.readlines()[1:]
    # print(contents)
    fd_list = []
    for e in contents:
        subj_name = e.split('/')[0]
        fd = e.split('/')[1].replace('\n', '').split(',')
        if subj_name == '6651225':
            continue
        fd_list.append([subj_name, fd])

    for e in fd_list:
        print(e)
    # assert False
    fd_list = np.array(fd_list)
    l_c, h_c, max_c = 0, 0, 20
    for i, e in enumerate(fd_list):
        sample = e
        subj_name = int(e[0])
        n = np.array(sample[1])[:5].astype(np.float32)
        # n = np.ones_like(n) / n
        # n = np.log(n)
        length = len(n)
        r = np.array([2 ** i for i in range(0, length)]).astype(np.float32)
        # print(n)
        for k in n:
            if str(k)=='nan':
                print(subj_name, n,e)
                assert False
            print(k, str(k)=='nan', end='/')
        print()
        # print(np.where(str(n) == 'nan'))
        # n_grad = np.gradient(np.log(n))
        # r_grad = np.gradient(np.log(r))
        if subj_name in low_subj_list and l_c < max_c:
            print('low list')
            # low_fdim.append(length)
            low_fdim.append(n)
            plt.plot(np.log(r), n, '-bo')
            l_c += 1
        elif subj_name in high_subj_list and h_c < max_c:
            print('high list')
            # high_fdim.append(length)
            high_fdim.append(n)
            plt.plot(np.log(r), n, '-ro')
            h_c += 1
    label_high = [1 for _ in range(h_c)]
    label_low = [0 for _ in range(l_c)]
    plt.show()
    print()
    print(h_c,l_c) # 30 and 95
    # print(label_high)
    # print(label_low)
    tTestResult = stats.ttest_ind(low_fdim, high_fdim)
    print(tTestResult)
    return high_fdim+low_fdim, label_high+label_low

def Lac_dataloader(xl_path, sheet_name):
    xl = openpyxl.load_workbook(xl_path, read_only=True)
    ws = xl[sheet_name]  # 'Sheet1 for test excel file
    low_fdim, high_fdim, data_excel = [], [] , []
    for row in ws.rows:
        line = []
        for cell in row:
            line.append(cell.value)
        data_excel.append(line)
    data_excel = np.array(data_excel)
    print(data_excel.shape)

    low_subj_pos = np.where(data_excel[:, 2] == 0)
    high_subj_pos = np.where(data_excel[:, 2] == 1)
    low_subj_list = [data_excel[i, 0] for i in low_subj_pos][0]
    high_subj_list = [data_excel[i, 0] for i in high_subj_pos][0]
    print(np.shape(low_subj_list))
    print(np.shape(high_subj_list))
    print(type(high_subj_list))

    result_file_name = './fd_result/Lac_result_v2.txt' # Lac_result_v2.txt
    fd = open(result_file_name, 'r')
    contents = fd.readlines()[1:]
    fd_list, name_list = [], []
    for e in contents:
        subj_name = e.split('/')[0]
        fd = e.split('/')[1].replace('\n', '').split(',')
        if subj_name == '6651225':
            continue
        if subj_name in name_list:
            continue
        fd_list.append([subj_name, fd])
        name_list.append(subj_name)

    # name_list = np.array(name_list)
    # print(name_list)
    # for n in name_list:
    #     index = np.where(name_list == n)
    #     print(index[0], len(index[0]))
    #
    #     # print(name_list[index])
    #     if len(index[0]) > 1:
    #         print(index)
    #
    #         for i in index[0]:
    #             print(fd_list[i])
    # assert False
    # for e in fd_list:
    #     print(e)
    # assert False
    fd_list = np.array(fd_list)
    l_c, h_c, max_c = 0, 0, 200
    for i, e in enumerate(fd_list):
        sample = e
        subj_name = int(e[0])
        n = np.array(sample[1])[:5].astype(np.float32)
        # n = np.ones_like(n) / n
        # n = np.log(n)

        for k in n:
            if str(k)=='nan':
                print(subj_name, n,e)
                assert False
            # print(k, str(k)=='nan', end='/')
        # print()

        if subj_name in low_subj_list and l_c < max_c:
            # print('low list')
            low_fdim.append(n)
            l_c += 1
        elif subj_name in high_subj_list and h_c < max_c:
            # print('high list')
            high_fdim.append(n)
            h_c += 1

    label_high = [1 for _ in range(h_c)]
    label_low = [0 for _ in range(l_c)]
    # plt.show()
    print()
    print(h_c,l_c)
    return high_fdim, low_fdim, label_high, label_low


if __name__ == "__main__":
    xl_test = '/home/soopil/Desktop/Dataset/brain_ewha/Test_Meningioma_20180508.xlsx'
    xl_train = '/home/soopil/Desktop/Dataset/brain_ewha/Train_Meningioma_20180508.xlsx'
    tr_high_fdim, tr_low_fdim, tr_label_high, tr_label_low = Lac_dataloader(xl_train,'20171108_New_N4ITK corrected')
    tst_high_fdim, tst_low_fdim, tst_label_high, tst_label_low = Lac_dataloader(xl_test,'Sheet1')
    train_x = tr_high_fdim + tr_low_fdim
    train_y = tr_label_high + tr_label_low
    test_x =  tst_high_fdim + tst_low_fdim
    test_y =  tst_label_high + tst_label_low
    # print(np.shape(data), np.shape(label))
    # print(np.shape(data), np.shape(label))
    high_fd = tr_high_fdim + tst_high_fdim
    low_fd = tr_low_fdim + tst_low_fdim
    print(np.shape(high_fd),np.shape(low_fd))
    tTestResult = stats.ttest_ind(high_fd, low_fd)
    print(tTestResult)

    # FD_dataloader(xl_train,'20171108_New_N4ITK corrected')

