import numpy
import torch
import mcubes
import trimesh
import torchcumesh2sdf
import matplotlib.pyplot as plotlib

from diso import DiffDMC, DiffMC
import glob
import time
import sapien.core as sapien

import time

# vol2mesh params
vol2mesh = DiffDMC(dtype=torch.float32).cuda()

# mesh2vol params
R = 256
threshold = 4 / R
margin = threshold * 2 + 1

meshFiles = glob.glob('/home/sarahwei/dataset/thingi10k_160/*.stl')
# meshFiles = ["/home/sarahwei/dataset/thingi10k_160/37730.stl"]
out_dir = "openvdb/"

engine = sapien.Engine()
for meshFile in meshFiles:
    mesh = trimesh.load(meshFile)
    start = time.perf_counter()
    geometry = engine.create_mesh_geometry(mesh.vertices, mesh.faces)
    p = sapien.coacd.run_remesh(geometry, resolution=R, level_set=0.)
    end = time.perf_counter()
    print("time: {}ms".format((end - start)*1000))
    file = open(out_dir + meshFile.split('/')[-1].split('.')[0] + ".txt", "w")
    file.write("time: {}ms".format((end - start)*1000))
    file.close()

    mcubes.export_obj(p.vertices, p.indices.reshape(-1, 3), out_dir + meshFile.split('/')[-1].split('.')[0] + ".obj")


