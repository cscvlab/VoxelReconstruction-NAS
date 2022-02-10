import pandas as pd
from dataLoader import VoxDataset
from utils import *
import argparse
import sys
from path import Path
from mlpnas import MLPNAS
from utils import regular
from CONSTANTS import TOP_N
from CONSTANTS import VOX_PATH

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)
def sample(voxdataset):
    x, y = voxdataset.getdata()
    edges_index = voxdataset.getedges()

    surface_x = x[edges_index == 1]
    surface_y = y[edges_index == 1]
    unsurface_x = x[edges_index == 0]
    unsurface_y = y[edges_index == 0]

    # sample unsurface
    p = np.random.choice(len(unsurface_x), size=int(len(unsurface_x) / 4), replace=False)  # 随机采样1/4
    unsurface_x = unsurface_x[p]
    unsurface_y = unsurface_y[p]

    surface_num = len(surface_x)
    unsurface_num = len(unsurface_x)

    tile_surface_x = np.tile(surface_x, (unsurface_num // surface_num, 1))
    tile_surface_y = np.tile(surface_y, unsurface_num // surface_num)

    train_x = np.concatenate((tile_surface_x, unsurface_x))
    train_y = np.concatenate((tile_surface_y, unsurface_y))
    validation_x = np.concatenate((surface_x, unsurface_x))
    validation_y = np.concatenate((surface_y, unsurface_y))

    p = np.random.permutation(len(train_x))
    train_x = train_x[p]
    train_y = train_y[p]

    train_x = regular(train_x)
    validation_x = regular(validation_x)
    return x,y,train_x, train_y,validation_x,validation_y

def run_mlpnas(args):
    vox_path = args.vox_path
    voxdataset=VoxDataset(vox_path)

    x,y,train_x, train_y, validation_x, validation_y=sample(voxdataset)

    nas_object = MLPNAS(train_x, train_y,validation_x,validation_y)
    data= nas_object.search()
    bestmodel=load_mlp_model()
    x=regular(x)
    result=np.squeeze((bestmodel.predict(x)> 0.5).astype("int32"),-1)
    path='LOGS/rect.txt'
    log_vox_path('LOGS/result.txt',vox_path)
    save_reconstruct(x,y,result,path)

    is_log_event = True
    if np.sum(result) == 0:
        is_log_event = False
    if is_log_event:
        log_event(os.path.basename(vox_path).split('.')[0])
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vox_path', type=Path, help='voxfile path')
    args = parser.parse_args()
    run_mlpnas(args)


