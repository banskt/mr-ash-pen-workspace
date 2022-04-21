from normal_means import NormalMeans

@NormalMeans.register_nm('ash_scaled')
class NMAshScaled(NormalMeans):
    
#     def __init__(self, y, prior, s2):
#         print("Initializing NMAshScaled")
#         self.y = y
#         self.prior = prior
#         self.s2 = s2
        
    @property
    def logML(self):
        print("Calculating NMAshScaled")
        return (self.prior + self.s2) / self.y
