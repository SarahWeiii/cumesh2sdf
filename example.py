import time
import numpy
import torch
import mcubes
import trimesh
import torchcumesh2sdf
import matplotlib.pyplot as plotlib


R = 512
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


tris, tris_min, tris_max = load_and_preprocess("/home/sarahwei/dataset/thingi10k_160/642501.stl")

start = time.perf_counter()
d = torchcumesh2sdf.get_sdf(tris, R, band)
torch.cuda.synchronize()
print("%.3f ms" % ((time.perf_counter() - start) * 1000))

d = d.cpu().numpy()
d = abs(d) - 0.9 / R

# # visualize
# plotlib.ion()
# act = plotlib.imshow(d[:, 0, :], vmin=-3 / R, vmax=3 / R)
# plotlib.colorbar()
# for i in range(0, R):
#     if (d[:, i, :] > 1e8).all():
#         continue
#     act.set_data(d[:, i, :])
#     plotlib.pause(0.01)
#     # plotlib.waitforbuttonpress()

# run marching cubes to reconstruct
v, f = mcubes.marching_cubes(d, 0)
print(v.max(), v.min())
v = ((v+0.5) / R * margin - band) * tris_max + tris_min
mcubes.export_obj(v, f, "tmp/test.obj")
