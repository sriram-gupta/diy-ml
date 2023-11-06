from math import pi
import math
import torch
import numpy as np
# plot dataset
import plotly.express as px
import pandas as pd
from utils import generateXGrid
from plotly.subplots import make_subplots
from plotly import graph_objects as go

def generate_dataset(n,r12=2,r34=5):
    half = n / 2
    theta=torch.linspace(0,4*pi,n).reshape(-1,1)


    x1 = torch.cos(theta)
    x2 = torch.sin(theta)
    y12 = torch.zeros(n)
    X1 = torch.tensor(np.array([x1,x2]).T).squeeze()*r12
    Y1 = y12

    x3 = torch.cos(theta)
    x4 = torch.sin(theta)
    y34 = torch.ones(n)
    X2 = torch.tensor(np.array([x3,x4]).T).squeeze()*r34
    Y2 = y34

    # Stack X1 and X2 tensors along with Y1 and Y2 tensors
    X = torch.cat((X1, X2), dim=0)  # Stack X1 and X2 along dimension 0
    Y = torch.cat((Y1, Y2), dim=0).unsqueeze(dim=1)  # Stack Y1 and Y2 along dimension 0


    return X1,Y1,X2,Y2, X,Y


def plot_dataset(X,Y,Z):
    # Convert torch tensors to NumPy arrays
    X_np = X.numpy()
    Y_np = Y.numpy()
    Z_np = Z.numpy()

    print(f" X_np {X_np.shape}, Y_np {Y_np.shape}, Z_np {Z_np.shape}")

    # Convert to pandas DataFrame for Plotly express
    data = {'X': X_np, 'Y': Y_np, 'Z': Z_np}
    df = pd.DataFrame(data)

    # Create a 3D scatter plot
    fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Z')
    # fig.update_traces(marker=dict(size=3))
    fig.show()

def plotAllModels(modelsConfigurations,ncols=3, n_points = 100, plot_width=10, dataframe=pd.DataFrame()):
    
    X, x1, x2 = generateXGrid(n_points=n_points, plot_width=plot_width)

    n_models = modelsConfigurations.__len__()
    ncols = ncols
    nrows = math.ceil(n_models / ncols)

    print(f" rows {nrows} cols {ncols}")

    # Create specs for subplots
    specs = [[{'type': 'scene'} for _ in range(ncols)] for _ in range(nrows)]

    # create figure
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=tuple(obj['desc'] for obj in modelsConfigurations) , specs=specs)

    i=1
    j=1

    for modelConfig in modelsConfigurations:
        print(i,j)
        Z = modelConfig['model'](X).detach().numpy()
        # Z_static = modelConfig['static'](X).detach().numpy()
        fig.add_trace(go.Surface(z=Z.reshape( n_points, n_points), x=x1, y=x2), row=i, col= j)
        # fig.add_trace(go.Surface(z=Z_static.reshape( n_points, n_points), x=x1, y=x2,  colorscale='Greys', opacity=0.5), row=i, col= j)
        if dataframe.__len__():
            fig.add_trace(go.Scatter3d(x=dataframe['X'],y=dataframe['Y'],z=dataframe['Z'],marker={"color":dataframe['Z']}),row=i, col= j)
        if(j%ncols==0):
            i=i+1
            j=0
        j=j+1
        

    fig.show()