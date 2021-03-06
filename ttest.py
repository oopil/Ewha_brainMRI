from scipy import stats
import numpy as np
import openpyxl
from matplotlib import pyplot as plt

xl_test = '/home/soopil/Desktop/Dataset/brain_ewha/Test_Meningioma_20180508.xlsx'
xl_train = '/home/soopil/Desktop/Dataset/brain_ewha/Train_Meningioma_20180508.xlsx'
xl = openpyxl.load_workbook(xl_train, read_only=True)
ws = xl['20171108_New_N4ITK corrected'] # 'Sheet1 for test excel file
data_excel = []
for row in ws.rows:
    line = []
    for cell in row:
        line.append(cell.value)
    data_excel.append(line)

data_excel = np.array(data_excel)
print(data_excel)
print(data_excel.shape)
print(data_excel[0])
low_subj_pos = np.where(data_excel[:,2] == 0)
high_subj_pos = np.where(data_excel[:,2] == 1)
low_fdim, high_fdim = [], []
low_subj_list = [data_excel[i,0] for i in low_subj_pos][0]
high_subj_list = [data_excel[i,0] for i in high_subj_pos][0]

print(np.shape(low_subj_list))
print(np.shape(high_subj_list))
print(type(high_subj_list))
# assert False

result_file_name = 'fd_result_file.txt'
fd = open(result_file_name, 'r')
contents = fd.readlines()[1:]
# print(contents)
fdim_list = []
for e in contents:
    subj_name = e.split('/')[0]
    fdim = e.split('/')[1].replace('\n','').split(',')
    # print(subj_name, fdim)
    # print(len(fdim), end=' ')
    fdim_list.append([subj_name, fdim])
    if subj_name == '1550930':
        print(contents.index(e))

l_c, h_c, max_c = 0,0,200
for i,e in enumerate(fdim_list):
    # print('iter ', i)
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

    else:
        # print(subj_name)
        # print(high_subj_list)
        pass
print()
print(low_fdim[0])
# assert False
plt.ylim(bottom = 0)
plt.show()
    #
    # if subj_name == '1550930':
    #     print(int(subj_name) in high_subj_list)
    #     print(list(high_subj_list))
    #     assert False
print()
# print(low_fdim, high_fdim)
# stats.ttest
tTestResult = stats.ttest_ind(low_fdim, high_fdim)
print(tTestResult)
# tTestResultDiffVar = stats.ttest_ind(titanic_survived['age'], titanic_n_survived['age'], equal_var=False)

    # print('boxcounting fractal dimension : ')
    # print(dim[:4])
    # plt.plot(np.log(r), dim, '-o')
    # plt.ylim(bottom = 0)
    # plt.show()


assert False
for i,e in enumerate(fdim_list):
    sample = e
    n = np.array(sample[1]).astype(np.float32)
    length = len(n)
    r = np.array([2 ** i for i in range(0, length)]).astype(np.float32)
    # print(sample)
    # print(length, n)

    # dim = np.divide(np.log(n),np.log(r)) * (-1)
    n_grad = np.gradient(np.log(n))
    r_grad = np.gradient(np.log(r))
    dim = np.divide(n_grad, r_grad) * (-1)
    # dim = np.gradient(np.log(n), np.log(r)) *(-1)
    print('boxcounting fractal dimension : ')
    # print(n_grad)
    # print(r_grad)
    print(dim[:4])
    # plt.scatter(np.log(r), dim)
    plt.plot(np.log(r), dim, '-o')
    plt.ylim(bottom = 0)
    plt.show()

    if i == 3:
        break
assert False

sample = fdim_list[25]
n = np.array(sample[1]).astype(np.float32)
length = len(n)
r = np.array([2**i for i in range(0, length)]).astype(np.float32)
# print(sample)
# print(length, n)

# dim = np.divide(np.log(n),np.log(r)) * (-1)
n_grad = np.gradient(np.log(n))
r_grad = np.gradient(np.log(r))
dim = np.divide(n_grad, r_grad) * (-1)
# dim = np.gradient(np.log(n), np.log(r)) *(-1)
print('boxcounting fractal dimension : ')
print(n_grad)
print(r_grad)
print(dim)
plt.plot(np.log(r), dim, '-o')
plt.ylim(bottom = 0)
# plt.plot(dim)
plt.show()
# assert False
# stats.ttest
# tTestResult = stats.ttest_ind(titanic_survived['age'], titanic_n_survived['age'])
# tTestResultDiffVar = stats.ttest_ind(titanic_survived['age'], titanic_n_survived['age'], equal_var=False)

"""
xl_file_name = excel_path
xl_password = '!adai2018@#'
xl = openpyxl.load_workbook(xl_file_name, read_only=True)
ws = xl['Sheet1']
self.data_excel = []
for row in ws.rows:
    line = []
    for cell in row:
        line.append(cell.value)
    self.data_excel.append(line)
# self.data_excel = np.array(self.data_excel)
"""
