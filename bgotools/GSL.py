
class GSL:
    def __init__(self,data,model=None):
        def f(x):
            try:
                return np.float(x)
            except:
                return np.nan
        self.data = data
        self.data['formation_energy'] = self.data['formation_energy'].apply(f)
        self.X 
        self.Y
        self.X2 
    def get_convex_hull(self):

        self.True_GSL
        self.GSL_difference


    def GSLError(self):
        #print(np.sqrt(np.trapz(GSL_difference**2)))
        #print(np.sqrt(np.trapz(True_GSL**2)))
        self.GSL_ERROR = np.sqrt(np.trapz(GSL_difference**2))/np.sqrt(np.trapz(True_GSL**2))
        return GSL_ERROR

