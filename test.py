import numpy
import torch
import mcubes
import trimesh
import torchcumesh2sdf
import matplotlib.pyplot as plotlib

from diso import DiffDMC, DiffMC
import glob
import time

# vol2mesh params
vol2mesh = DiffDMC(dtype=torch.float32).cuda()

# mesh2vol params
R = 256
band = 3 / R
margin = band * 2 + 1

def load_and_preprocess(p):
    mesh = trimesh.load(p)
    tris = numpy.array(mesh.triangles, dtype=numpy.float32, subok=False)

    tris_min = tris.min(0).min(1)
    tris = tris - tris_min
    tris_max = tris.max()
    tris = (tris / tris_max + band) / margin

    return torch.tensor(tris, dtype=torch.float32, device='cuda:0'), tris_min, tris_max

meshFiles = glob.glob('/home/sarahwei/dataset/thingi10k_160/*.stl')
# meshFiles = ["tmp/spot.obj"]
out_dir = "output256/"

for meshFile in meshFiles:
    tris, tris_min, tris_max = load_and_preprocess(meshFile)
    print(meshFile)

    file = open(out_dir + meshFile.split('/')[-1].split('.')[0] + ".txt", "w")

    tris = torch.tensor(tris, dtype=torch.float32, device='cuda:0')
    torch.cuda.synchronize()
    start = time.perf_counter()
    start1 = time.perf_counter()
    d = torchcumesh2sdf.get_sdf(tris, R, band)
    if d.min() >= 0:
        print("Using UDF!")
        file.write("Using UDF!\n")
        d = d - 0.9 / R
    torch.cuda.synchronize()
    end1 = time.perf_counter()
    start2 = time.perf_counter()
    v, f = vol2mesh(d, None)
    torch.cuda.synchronize()
    end2 = time.perf_counter()
    end = time.perf_counter()
    print("stage1: {}ms".format((end1 - start1)*1000))
    print("stage2: {}ms".format((end2 - start2)*1000))
    print("time: {}ms".format((end - start)*1000))

    file.write("stage1: {}ms\n".format((end1 - start1)*1000))
    file.write("stage2: {}ms\n".format((end2 - start2)*1000))
    file.write("time: {}ms".format((end - start)*1000))
    file.close()

    v, f = v.cpu().numpy(), f.cpu().numpy()
    v = ((v * R +0.5)/(R+1)* margin - band) * tris_max + tris_min

    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.export(out_dir + meshFile.split('/')[-1].split('.')[0] + ".obj")