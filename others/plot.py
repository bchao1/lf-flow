import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def test_3d():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(9)
    Y = np.arange(9)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.load("../err.npy")#np.sin(R)
    Z = Z.reshape(9, 9)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    plt.title("Reconstruction error at different views")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
if __name__ == '__main__':
    test_3d()
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
#
    #dist = np.load("../w.npy")
    #h, w = dist.shape
    #h, w = np.meshgrid(h, w, indexing='ij')
    #surf = ax.plot_surface(w, h, dist, cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=False)
#
    #ax.set_zlim(0, 20)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.show()
    #plt.style.use("ggplot")
    #d = np.load("../w.npy")
    #err = np.load("../err.npy")
    #d2err = {d[i]: err[i] for i in range(len(d))}
    #sorted_d = list(sorted(d2err.keys()))
    #err = [d2err[d] for d in sorted_d]
    #plt.title("View Reconstruction Error against Distance from Stereo")
    #plt.ylabel("Recon. error")
    #plt.xlabel("Distance from given stereo image")
    #plt.plot(sorted_d, err)
    #plt.show()