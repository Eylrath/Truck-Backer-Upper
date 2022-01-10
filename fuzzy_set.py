import numpy as np
import matplotlib.pyplot as plt

class FuzzySet():
    """class for fuzzy set"""
    def __init__(self, min_x, max_x, precision=0.1):
        """class for fuzzy set (or rather set of fuzzy sets)
            input:
                min_x, max_x - minimum and maximum for all depending sets
                precision - difference between adjacent points"""
#         if min_x < max_x:
        self.X = np.arange(min_x, max_x+precision, precision)
        self.set_names = []
        self.sets = None
        self.precision = precision
        self.outputs = []
        self.zero_set = np.ones(len(self.X))
    def trampf(self, a, b, c, d, name):
        """trapezoidal function, adds new fuzzy set to main dictionary
            input: 
                a,b,c,d - parameters
                name - individual name to add to dictionary"""
        trap_temp = np.array([(self.X - a)/ (b - a), np.ones(len(self.X)), (d - self.X)/(d-c)])
        Y = np.min(trap_temp, axis=0)
        trap_temp = np.array( [Y, np.zeros(len(Y))] )
        Y = np.max(trap_temp, axis=0)
        
        self.add_set(name, Y)
        
    def trimf(self, a, b, c, name):
        """triangular function, adds new fuzzy set to main dictionary
            input: 
                a,b,c - parameters
                name - individual name to add to dictionary"""
        tri_temp = np.array( [(self.X-a)/(b-a), (c-self.X)/(c-b)] )
        Y = np.min(tri_temp, axis=0)
        tri_temp = np.array( [Y, np.zeros(len(Y))] )
        Y = np.max(tri_temp, axis=0)
        self.add_set(name, Y)
        
    def add_set(self, name, Y):
        """adds new fuzzy set (new membership function) determined and used by trimf and trampf"""
        if self.sets is None:
            self.sets = np.array([Y])
        else:
            self.sets = np.vstack([self.sets, Y])
        self.set_names.append(name)
        self.outputs.append(0)
        
    def cut_sets():
        pass
        
    def centroid(self,name):
        set_idx = self.set_names.index(name)
        prod = np.sum( self.sets[set_idx] * self.X )
        return prod / np.sum(self.sets[set_idx])
        
    #todo - pozostale dwie funkcje
    def plot(self, names=None):
        """plots whole set"""
        if names is None:
            for i in range(len(self.sets)):
                plt.plot(self.X, self.sets[i], label=self.set_names[i])
            plt.legend(self.set_names)
        else:
            for i in range(len(self.sets)):
                if self.set_names[i] in names:
                    plt.plot(self.X, self.sets[i])
            plt.legend(names)
        
    def value_of(self, name, x):
        """returns closest value of x through membership function determined by name"""
        closest_idx = np.abs(self.X - x).argmin()
        set_idx = self.set_names.index(name)
        return self.sets[set_idx][closest_idx]
        
    def fuzzify(self, x):
        """fuzzifies x"""
#         sets_output = np.zeros(len(self.sets))
        i = 0
        for name in self.set_names:  
            self.outputs[i] = self.value_of(name, x)
#             print(name, self.value_of(name, x))
            i += 1
        return np.array( self.outputs )
        
    def getNameIdx(self, name):
        return self.set_names.index(name)
    def getNames(self):
        return self.set_names
    def getX(self):
        return self.X
    def getXlen(self):
        return len(self.X)
    def getSets(self):
        return self.sets
    def getSet(self, name):
        if name in self.set_names:
            return self.sets[self.getNameIdx(name)]
        return self.zero_set
    def getLen(self):
        return len(self.sets)