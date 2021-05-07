import math
import sys
import os
import numpy as np

f = open('up.txt')

data = np.zeros((84,2))
num = 0

sys.stdout = open('up_result.txt', 'w')
for line in f:
    data_list = line.split()
    num1 = 15*3
    num2 = 13*3
    x1 = float(data_list[num1+1])
    y1 = float(data_list[num1+2])
    x2 = float(data_list[num2+1])
    y2 = float(data_list[num2+2])

    p1 = float(data_list[16*3+1])
    k1 = float(data_list[16*3+2])
    p2 = float(data_list[14*3+1])
    k2 = float(data_list[14*3+2])

    width1 = math.fabs(x1 - x2)
    height1 = math.fabs(y1 - y2)
#    radian1 = math.atan2(height1, width1)
#    angle1 = radian1*180/math.pi
    res1 = height1/width1
    
    width2 = math.fabs(p1 - p2)
    height2 = math.fabs(k1 - k2)
#    radian2 = math.atan2(height2, width2)
#    angle2 = radian2*180/math.pi
    res2 = height2/width2
    
    data[num, 0] = res1
    data[num, 1] = res2
    print(res1 , res2)
    num = num + 1
    
f.close()


print('왼쪽 기울기 평균: ', np.mean(data[:, 0]))
print('오른쪽 기울기 평균: ', np.mean(data[:, 1]))
