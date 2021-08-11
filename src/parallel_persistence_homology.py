import multiprocessing
import os

import pandas as pd


def task(p):
    print(p[0].split('\\')[-1], "====", p[1].split('\\')[-1])
    cmd = "C:\\Users\\18056\\Desktop\\Wasserstein\\wasserstein\\cmake-build-debug\\wasserstein.exe " + p[0] + " " + p[1]
    os.system(cmd)
    return


if __name__ == "__main__":
    cores = int(multiprocessing.cpu_count() * 0.6)  # multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    folder_path = 'C:\\Users\\18056\\Desktop\\Wasserstein\\1728_picked_persistence_dia\\'
    file_list = os.listdir(folder_path)
    num = len(file_list)

    params = []
    # for i in range(num):
    #     for j in range(i, num):
    #         # print(i, j)
    #         file1_path = folder_path + '\\' + file_list[i]
    #         file2_path = folder_path + '\\' + file_list[j]
    #         params.append([file1_path, file2_path])

    # for i in range(1):
    #     for j in range(i, num):
    #         # print(i, j)
    #         file1_path = folder_path + '\\' + "1_17300_3426_x20339_y44872.swc"
    #         file2_path = folder_path + '\\' + file_list[j]
    #         params.append([file1_path, file2_path])

    m = pd.read_csv(r'C:\Users\18056\Desktop\Wasserstein\m.csv', sep=' ')
    for i in m.index.tolist():
        file1_path = folder_path + m.loc[i,'p1']
        file2_path = folder_path + m.loc[i,'p2']
        params.append([file1_path, file2_path])

    print("do the task")
    for p in params:
        pool.apply_async(task, (p,))
        # print(p)
    pool.close()
    pool.join()


