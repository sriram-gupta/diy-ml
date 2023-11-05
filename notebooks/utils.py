import torch

def generateXGrid(n_points = 100,plot_width = 10 ):
    x1 = torch.linspace(-plot_width,plot_width, n_points)
    x2 = torch.linspace(-plot_width,plot_width, n_points)

    xx1 , xx2 = torch.meshgrid(x1,x2)

    X1 = xx1.flatten().unsqueeze(dim=1)
    X2 = xx2.flatten().unsqueeze(dim=1)

    X = torch.hstack((X1,X2))
    print(X1.shape, X2.shape, X.shape)
    return X, x1,x2