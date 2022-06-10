import numpy as np

plotdir = "."

def saveplot(fig, fileprefix):
    for ext in ['pdf', 'png']:
        fl = f"{plotdir}/{fileprefix}.{ext}"
        fig.savefig(fl, bbox_inches = 'tight')
    return
