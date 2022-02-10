import os
import shutil
import pickle
import numpy as np
from chamfer_distance import compute_chamfer_distance
from itertools import groupby
from matplotlib import pyplot as plt
from CONSTANTS import *
from mlp_generator import MLPSearchSpace
from tensorflow.keras.models import load_model
from validator import iou
from focalLoss import binary_focal_loss
# from keras import backend as K

########################################################
#                   DATA PROCESSING                    #
########################################################


def unison_shuffled_copies(a, b,weights=None):
    assert len(a) == len(b)
    if weights is None:#Disorganize sample order
        p = np.random.permutation(len(a))
    else:#choice samples by weights
        p=np.random.choice(len(a),size=round(len(a)/4),replace=True,p=weights)
    return a[p], b[p]


def regular(x):
    x=(x-64.5)/64
    return x

########################################################
#                       LOGGING                        #
########################################################
def clean_log():
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            os.remove('LOGS/{}'.format(file))

def log_event(path):
    dest = 'LOGS'
    while os.path.exists(dest):
        dest = 'LOGS/{}'.format(path)
    os.mkdir(dest)
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            shutil.move('LOGS/{}'.format(file),dest)


def get_latest_event_id():
    all_subdirs = ['LOGS/' + d for d in os.listdir('LOGS') if os.path.isdir('LOGS/' + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('LOGS/', ''))


########################################################
#                 RESULTS PROCESSING                   #
########################################################


def load_nas_data():
    event = get_latest_event_id()
    data_file = 'LOGS/{}/nas_data.pkl'.format(event)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def load_mlp_model():
    model_file = 'LOGS/mlp_model.h5'
    return load_model(model_file,custom_objects={'iou':iou,'binary_focal_loss_fixed':binary_focal_loss()})

def sort_search_data(nas_data):
    val_accs = [item[1] for item in nas_data]
    sorted_idx = np.argsort(val_accs)[::-1]
    nas_data = [nas_data[x] for x in sorted_idx]
    return nas_data

########################################################
#                EVALUATION AND PLOTS                  #
########################################################

def get_top_n_architectures(n):#show architectures
    data = load_nas_data()
    data = sort_search_data(data)
    search_space = MLPSearchSpace(TARGET_CLASSES)
    print('Top {} Architectures:'.format(n))
    for seq_data in data[:n]:
        print('Architecture', search_space.decode_sequence(seq_data[0]))
        print('Validation Accuracy:', seq_data[1])

def get_best_sequence(data):#reurn sequence
    data = sort_search_data(data)
    i=1
    choice=0
    min_size=data[0][2]
    while i<len(data)-1:
        if data[0][1]-data[i][1]>THREAD_ACC:
            break
        if(data[i][2]<min_size):
            min_size=data[i][2]
            choice=i
        i=i+1
    seq_data=data[choice][0]
    print("choice :",choice)
    return seq_data

def get_nas_accuracy_plot():
    data = load_nas_data()
    accuracies = [x[1] for x in data]
    plt.plot(np.arange(len(data)), accuracies)
    plt.show()


def get_accuracy_distribution():
    event = get_latest_event_id()
    data = load_nas_data()
    accuracies = [x[1]*100. for x in data]
    accuracies = [int(x) for x in accuracies]
    sorted_accs = np.sort(accuracies)
    count_dict = {k: len(list(v)) for k, v in groupby(sorted_accs)}
    plt.bar(list(count_dict.keys()), list(count_dict.values()))
    plt.show()

########################################################
#                     SHOW DATA                        #
########################################################
def save_points(x,y,path):
    assert len(x) == len(y)
    with open(path, "w") as f:
        for i in range(x.__len__()):
            f.write(str(x[i][0]) + " " + str(x[i][1]) + " " + str(x[i][2])+"\n")
    f.close()
    print("done")
def save_solid_points(x,y,path):
    assert len(x) == len(y)
    with open(path, "w") as f:
        for i in range(x.__len__()):
            if(y[i]==1):
                f.write(str(x[i][0]) + " " + str(x[i][1]) + " " + str(x[i][2]) + "\n")
    f.close()
    print("done")
def save_reconstruct(x,lable,predict,path):
    assert len(lable) == len(predict)==len(x)
    inside_right_count=0
    inside_wrong_count=0
    outside_right_count=0
    outside_wrong_count=0
    with open(path, "w") as f:
        for i in range(len(lable)):
            point_type=OUTSIDE_RIGHT
            if(lable[i]==predict[i]):#分类正确的点
                if(lable[i]==1):#内部正确点
                    point_type=INSIDE_RIGHT
                    inside_right_count+=1
                else:#外部正确点
                    outside_right_count+=1
                    continue
            else:#分类错误的点
                if(lable[i]==1):#内部错误点
                    point_type=INSIDE_WRONG
                    inside_wrong_count+=1
                else:#外部错误点
                    point_type=OUTSIDE_WRONG
                    outside_wrong_count+=1
            f.write(get_color(x[i],point_type))
    f.close()
    acc=100*(inside_right_count+outside_right_count)/(inside_right_count+outside_right_count+inside_wrong_count+outside_wrong_count)
    iou=100*(inside_right_count)/(inside_right_count+inside_wrong_count+outside_wrong_count)
    chamfer=compute_chamfer_distance(x,predict,lable)*1000
    print("acc：{:.6f}%".format(acc))
    print("iou: {:.6f}%".format(iou))
    print("chamfer: {:.6f}".format(chamfer))
    with open('LOGS/result.txt', "a") as f:
        f.write('inside_right_num '+str(inside_right_count)+'\n')
        f.write('inside_wrong_num ' + str(inside_wrong_count) + '\n')
        f.write('outside_wrong_num ' + str(outside_wrong_count) + '\n')
        f.write('err_num '+str(inside_wrong_count+outside_wrong_count)+'\n')
        f.write('acc '+str(acc)+'\n')
        f.write('iou '+str(iou)+'\n')
        f.write('chamfer '+str(chamfer))
    f.close()
def get_color(position,point_type):
    ret=str(position[0])+" "+str(position[1])+" "+str(position[2])
    if(point_type==INSIDE_RIGHT):
        ret+=" "+INSIDE_RIGHT_COLOR+"\n"
    elif(point_type==INSIDE_WRONG):
        ret+=" "+INSIDE_WRONG_COLOR+"\n"
    else:
        ret+=" "+OUTSIDE_WRONG_COLOR+"\n"
    return ret
########################################################
#                   SAVE VOX PATH                      #
########################################################
def log_vox_path(path,value):
    with open(path, "a") as f:
        f.write('path '+value+'\n')
    f.close()

def architectrue_size(architectrue):
    architectrue.insert(0, (3, ''))
    architectrue_size = 0
    for i in range(0,len(architectrue)-1):
        architectrue_size+=architectrue[i][0]*architectrue[i+1][0]+architectrue[i+1][0]
    return architectrue_size

def log_best_architectrue(sequence):
    search_space = MLPSearchSpace(TARGET_CLASSES)
    best_architectrue=search_space.decode_sequence(sequence)
    best_architectrue_size=architectrue_size(best_architectrue)
    with open('LOGS/result.txt','w') as f:
        f.write('architectrue '+str(best_architectrue)+'\n')
        f.write('size '+str(best_architectrue_size)+'\n')
    f.close()














