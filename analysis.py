import os
import numpy as np
from matplotlib import pyplot as plt
if __name__ == '__main__':
    rootdir='./LOGS'
    dirs=os.listdir(rootdir)
    acc=np.zeros((len(dirs),))
    iou=np.zeros((len(dirs),))
    size=np.zeros((len(dirs),))
    i=0
    for dir in dirs:
        path=os.path.join(rootdir,dir,'result.txt')
        with open(path,'r') as f:
            for line in f:
                aline = list(line.strip('\n').split(' '))
                if aline[0]=='acc':
                    acc[i]=float(aline[1])
                if aline[0]=='iou':
                    iou[i]=float(aline[1])
                if aline[0]=='size':
                    size[i]=float(aline[1])
            i+=1
        f.close()
    print('average_acc:',np.mean(acc))
    print('average_iou:',np.mean(iou))
    print('average_size:',np.mean(size))
    plt.hist(iou,bins=[0,10,20,30,40,50,60,70,80,90,100])
    plt.title("chair_iou")
    plt.show()
print('finish analysis')