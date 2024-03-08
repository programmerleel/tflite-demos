# -*- coding: utf-8 -*-
# @Time    : 2024/03/08 11:32
# @Author  : LiShiHao
# @FileName: infer.py
# @Software: PyCharm

from concurrent import futures
import time
import random


def returnNumber(number: int) -> int:
    print("start return number {}".format(number))
    # time.sleep(random.randint(10, 20))
    print("end return number {}".format(number))
    return number


if __name__ == '__main__':
    with futures.ThreadPoolExecutor(5) as executor:
        to_do = []
        for number in range(0, 10):
            future = executor.submit(returnNumber, number)  # 添加到futures的任务队列，等待排队执行
            print(future)
            to_do.append(future)
        print("-----middle-----")
        futures.wait(to_do)
        for future in futures.as_completed(to_do):  # 当某一个future任务执行完毕后，执行下面代码。会阻塞，等待线程完成后执行
            res = future.result()  # 获取线程的返回结果
            print(res)

    print("-----end-----")