import numpy as np
import geometry
import os

def regular(x):
    x=(x-64.5)/64
    return x

def getNeighbour(i,dims):
    x=i%dims[0]
    y=int(i/dims[0])%dims[1]
    z=int(i/dims[0]/dims[1])%dims[2]
    p=[x,y,z]
    neighbours=[]
    if p[0]>0:#x>0
        neighbours.append([p[0]-1,p[1],p[2]])
    if (p[1]>0):#y>0
        neighbours.append([p[0], p[1]-1, p[2]])
    if (p[2]>0):#z>0
        neighbours.append([p[0], p[1], p[2]-1])
    if p[0]<dims[0]-1:#x<x_max
        neighbours.append([p[0]+1,p[1],p[2]])
    if p[1]<dims[1]-1:#y<y_max
        neighbours.append([p[0], p[1]+1, p[2]])
    if p[2]<dims[2]-1:#z<z_max
        neighbours.append([p[0], p[1], p[2]+1])
    return neighbours
def is_solid_edge(i,data,dims):
    neighbours=getNeighbour(i, dims)
    for neighbour in neighbours:
        idx=get_index(neighbour,dims)
        if (data[i] == 1 and data[idx] == 0):
            return True
    return False

def get_index(xyz,dims):
    return int(xyz[0]*1+xyz[1]*dims[1]+xyz[2]*dims[1]*dims[2])

def is_out_edge(xyz,data,dims):
    idx = get_index(xyz, dims)
    if(data[idx]==0):
        return True

class VoxDataset():
    def __init__(self,mesh_file):
        mesh=geometry.Mesh(mesh_file)
        self.filename=os.path.basename(mesh_file)
        self.vertices=mesh.V()*64+64.5
        self.dims = [130, 130, 130]

        size=int(self.dims[0]*self.dims[1]*self.dims[2])

        self.data = np.full(size, 0, dtype=int)
        #data
        for i in range(len(self.vertices)):
            index = get_index(self.vertices[i],self.dims)
            self.data[index] = 1
        #solid_edge
        self.edges = np.full(size, 0, dtype=int)
        self.solid_edges=np.full(size, 0, dtype=int)
        for i in range(len(self.vertices)):
            index = get_index(self.vertices[i],self.dims)
            if is_solid_edge(index,self.data,self.dims):
                self.solid_edges[index]=1
                self.edges[index] = 1

        #get out_edge
        for i in range(len(self.edges)):
            if(self.solid_edges[i]==1):
                neighbours=getNeighbour(i,self.dims)
                for neighbour in neighbours:
                    if is_out_edge(neighbour,self.data,self.dims):
                        self.edges[get_index(neighbour,self.dims)]=1

        #position xyz
        self.coordinateas =np.array([[k, j, i]
             for i in range(self.dims[0])
             for j in range(self.dims[1])
             for k in range(self.dims[2])])
    def getdata(self):
        return self.coordinateas,self.data

    def getsolidedges(self):
        return self.solid_edges

    def getedges(self):
        return self.edges