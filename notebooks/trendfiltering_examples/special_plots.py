import numpy as np
import collections
import patsy # for the B-spline bases

def plot_data_from_bspline(axlist, x, y, knots, degree, G, Gb, H, Hb, 
                           include_intercept = False, show_base_legend = True):
    ax1 = axlist[0]
    ax2 = axlist[1]
    ax3 = axlist[2]
    ax4 = axlist[3]

    ax1.set_title("Contribution from B-Spline bases")
    ax2.set_title("Contribution from Trendfiltering bases")

    kpos = np.array([np.argmin(np.abs(x - k)) for k in knots])
    Gbarr = np.zeros(x.shape[0])
    Gbstr = list()
    icount = 0
    if include_intercept:
        Gbstr.append("Intercept")
        Gbarr[icount] = Gb[icount]
        icount += 1
    for i in range(degree):
        Gbstr.append(f"Degree {i+1}")
        Gbarr[icount] = Gb[icount]
        icount += 1
    for k in kpos:
        Gbstr.append(f"Knot {k}")
        Gbarr[k] = Gb[icount]
        icount += 1

    for i in range(G.shape[1]):
        ax1.plot(x, Gb[i] * G[:, i], label = Gbstr[i])
    if show_base_legend: ax1.legend(title = "Index")

    for i,b in enumerate(Hb):
        if b != 0:
            ax2.plot(x, b * H[:, i], label = f"{i+1}")
    if show_base_legend: ax2.legend(title = "Base index")
    
    
    ax3.scatter(x, y, s = 5, edgecolor = 'black', facecolor='white')
    ax3.plot(x, np.dot(H, Hb), label = "TF basis")

    ax3.plot(x, np.dot(G, Gb), label = "B-Spline")
    ax3.legend(frameon = True, borderpad = 1)
    ax3.set_title("Generated curve")
    ax3.set_ylabel("y")

    ax4.scatter(x[Hb==0], Hb[Hb==0], s = 1, edgecolor = 'black', facecolor='white')
    ax4.scatter(x[Hb!=0], Hb[Hb!=0], label = "TF basis")
    ax4.scatter(x[Gbarr!=0], Gbarr[Gbarr!=0], label = "B-Spline")
    ax4.legend(frameon = True, borderpad = 1)
    ax4.set_title("Coefficients")
    ax4.set_ylabel("b")
    return
