# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 19:13
# @Author  : Lee
# @Project ：yolov5-hamlet-detection 
# @File    : test.py
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
import queue
import threading

pool_1 = ThreadPoolExecutor(16)
pool_2 = ThreadPoolExecutor(8)
q = queue.Queue(maxsize=16)
tasks = []

def p(x):
    q.put(x)
    print("p:{}".format(x))

def c():
    x = q.get()
    print("c:{}".format(x))

start = time.time()

for x in range(1000000):
    tasks.append(pool_1.submit(p,x))

for task in as_completed(tasks):
    pool_2.submit(c)

end = time.time()
print(end-start)