import sys
sys.path.insert(0, "./submodules/libigl/python/")

import numpy as np
import pyigl as igl
import iglhelpers
SAMPLE_SPHERE_RADIUS=1
def createGrid(res):
    K = np.linspace(
        -SAMPLE_SPHERE_RADIUS,
        SAMPLE_SPHERE_RADIUS,
        res
    )
    grid = [[x, y, z] for x in K for y in K for z in K]
    return np.array(grid)


class Mesh():
    _V = igl.eigen.MatrixXd()
    _F = igl.eigen.MatrixXi()
    _normalized = False

    def __init__(
            self,
            meshPath=None,
            V=None,
            F=None,
            viewer=None,
            ):

        if meshPath == None:
            if V == None or F == None:
                raise ("Mesh path or Mesh data must be given")
            else:
                self._V = V
                self._F = F
        else:
            self._loadMesh(meshPath)

        self._viewer = viewer

    def _loadMesh(self, fp):
        # load mesh
        igl.read_triangle_mesh(fp, self._V, self._F)


    def V(self):
        return iglhelpers.e2p(self._V)


    def show(self, doLaunch=True):
        if self._viewer == None:
            self._viewer = igl.glfw.Viewer()

        self._viewer.data(0).set_mesh(self._V, self._F)

        if doLaunch:
            self._viewer.launch()

    def save(self, fp):
        igl.writeOBJ(fp, self._V, self._F)


gridPath='/home/ch/Desktop/mlpnas_chair/DATASETS/cube/cube.obj'
meshPath='/home/ch/Desktop/mlpnas_chair/DATASETS/39993.obj'
if __name__ == '__main__':
    mesh_v=Mesh(meshPath).V()*64+0.5+64#0~129

    grid_v=Mesh(gridPath).V()*64+0.5+64#0~129
    print(len(grid_v))
    data=np.full(len(grid_v),0,dtype=int)
    for i in range(len(mesh_v)):
        index=int(mesh_v[i][0]*1+mesh_v[i][1]*129+mesh_v[i][2]*129*129)
        data[index]=1

    print("done")
