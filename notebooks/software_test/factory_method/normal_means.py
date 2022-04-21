
class NormalMeans:

    subclasses = {}

    @classmethod
    def register_nm(cls, prior_type):
        def decorator(subclass):
            cls.subclasses[prior_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, prior_type, params, extparams = {}):
        if prior_type not in cls.subclasses:
            raise ValueError('Bad prior type {}'.format(prior_type))
        return cls.subclasses[prior_type](*params, **extparams)
    
    def __init__(self, y, prior, s2):
        print("Initializing NormalMeans")
        self.y = y
        self.prior = prior
        self.s2 = s2
        
    def logML(self):
        print("Calculating NormalMeans")
        return
    
    def shrinkage_operator(self):
        return self.logML
    
    def penalty_operator(self):
        return self.y * 2
