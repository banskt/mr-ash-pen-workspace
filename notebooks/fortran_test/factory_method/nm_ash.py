
from normal_means import NormalMeans

@NormalMeans.register_nm('ash')
class NMAsh(NormalMeans):
    
    def __init__(self, y, prior, s2, dj = None):
        print("Initializing NMAsh")
        self.y = y
        self.prior = prior
        self.s2 = s2
        self.dj = dj
        print ("dj is", dj)
        
    @property
    def logML(self):
        print("Calculating NMAsh")
        return self.prior + self.s2
