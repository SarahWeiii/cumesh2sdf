import numpy
import torch
import mcubes
import trimesh
import torchcumesh2sdf
import matplotlib.pyplot as plotlib

from diso import DiffDMC, DiffMC
import glob
import time

def tri_area(v0, v1, v2):
    ab = v1 - v0
    ac = v2 - v0
    return 0.5 * numpy.linalg.norm(numpy.cross(ab, ac), axis=1)

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
out_dir = "output256_v3/"

for meshFile in meshFiles:
    tris, tris_min, tris_max = load_and_preprocess(meshFile)
    print(meshFile)

    file = open(out_dir + meshFile.split('/')[-1].split('.')[0] + ".txt", "w")

    tris = torch.tensor(tris, dtype=torch.float32, device='cuda:0')
    torch.cuda.synchronize()
    start = time.perf_counter()
    start1 = time.perf_counter()
    d = torchcumesh2sdf.get_sdf(tris, R, band)
    if d.min() >= -0.5 / R:
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

    # divide quads into triangles by traingles area
    area_012 = tri_area(v[f[:,0]], v[f[:,1]], v[f[:,2]])
    area_023 = tri_area(v[f[:,0]], v[f[:,2]], v[f[:,3]])
    area_013 = tri_area(v[f[:,0]], v[f[:,1]], v[f[:,3]])
    area_123 = tri_area(v[f[:,1]], v[f[:,2]], v[f[:,3]])
    faces_1 = f[numpy.abs(area_012-area_023) > numpy.abs(area_013-area_123)]
    faces_2 = f[numpy.abs(area_012-area_023) <= numpy.abs(area_013-area_123)]
    f = numpy.concatenate([faces_1[:, [0, 1, 3, 1, 2, 3]].reshape(-1, 3), faces_2[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)], axis=0)

    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.export(out_dir + meshFile.split('/')[-1].split('.')[0] + ".obj")

    # v, f = mcubes.marching_cubes(d.cpu().numpy(), 0)
    # print(v.max(), v.min())
    # v = ((v+0.5) / R * margin - band) * tris_max + tris_min
    # mcubes.export_obj(v, f, "tmp/test.obj")

