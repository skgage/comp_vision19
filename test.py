import k_means_funcs as kmf
from scipy.cluster.vq import vq, kmeans, whiten
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py
import plotly
import glob
import os.path
import plotly.graph_objs as go
from collections import Counter

folder_path = "/Users/teopb/Desktop/photo_sample"

files = glob.glob(os.path.join(folder_path, '*.jpg'))

file_idx, codes, means, bins = kmf.bin_photos(files, 2, .25)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(means[:, 0], means[:, 1], means[:, 2])
# plt.show()

trace1 = go.Scatter3d(
    x=means[:, 0],
    y=means[:, 1],
    z=means[:, 2],
    mode='markers',
    marker=dict(
        size=16,
        color = codes, #set color equal to a variable
        colorscale='Jet',
        showscale=True
    )
)

layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

data = [trace1]

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig, filename='simple-3d-scatter')
