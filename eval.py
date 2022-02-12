import os

from validator import iou
from tensorflow.keras.models import load_model
from focalLoss import binary_focal_loss
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
save_file_root="./out"
if not os.path.exists(save_file_root):
    os.mkdir(save_file_root)
def regular(x):
    x=(x-64.5)/64
    return x
def save_reconstruct(x,y,save_file_path):
    assert len(x) == len(y)
    with open(save_file_path, "w") as f:
        for i in range(len(x)):
            if y[i]==1:
                f.write(str(x[i][0])+" "+str(x[i][1])+" "+str(x[i][2])+"\n")
    f.close()
def render(renderfile):
    x = np.array([[k, j, i]
              for i in range(130)
              for j in range(130)
              for k in range(130)])
    print("start render:" + renderfile + "...")
    model=load_model(renderfile,custom_objects={'iou':iou,'binary_focal_loss_fixed':binary_focal_loss()})
    x=regular(x)
    y=np.squeeze((model.predict(x)> 0.5).astype("int32"),-1)
    save_file_path = os.path.join(save_file_root,renderfile.split("/")[-2]+".txt")
    save_reconstruct(x,y,save_file_path)

if __name__=='__main__':
    root_dir="./RESULTS/model/thingi10k"
    dirs=os.listdir(root_dir)
    for dir in dirs:
        renderfile=os.path.join(os.path.join(root_dir,dir),"mlp_model.h5")
        render(renderfile)
        print("render finish")
    print("finish")