import k_means_funcs as kmf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py
import plotly
import plotly.graph_objs as go

means, dists, file_idx = kmf.calc_photo_dists("/Users/teopb/Desktop/PhotoSorter_images", 2)

print(file_idx)
print(dists)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(means[:, 0], means[:, 1], means[:, 2])
plt.show()

trace1 = go.Scatter3d(
    x=means[:, 0],
    y=means[:, 1],
    z=means[:, 2],
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color=means[:, 2],
            width=0.5
        ),
        opacity=0.8
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
