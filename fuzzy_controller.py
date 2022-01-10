from fuzzy_set import *
import numpy as np
import matplotlib.pyplot as plt

class FuzzyController():
    """class for fuzzy controller"""
    def __init__(self):
        """initializes fuzzy controller"""
        self.antecedant_1 = None
        self.ant_name_1 = None
        self.antecedant_2 = None
        self.ant_name_2 = None
        self.rules = []
        self.ant_result = []
        self.consequent = None
        self.consequent_name = None
        self.result = None
    def add_antecedant(self, name, fuzzySet):
        """sets first antecedant or if it exists - second by certain fuzzySet"""
        if self.antecedant_1 is None:
            self.antecedant_1 = fuzzySet
            self.ant_name_1 = name

        else:
            self.antecedant_2 = fuzzySet
            self.ant_name_2 = name
            rules = np.array([["  "]*self.antecedant_2.getLen() for i in range(self.antecedant_1.getLen())])
            self.rules = rules
            result = np.array([[0]*self.antecedant_2.getLen() for i in range(self.antecedant_1.getLen())]) 
            self.result = result


    def set_consequent(self, name, fuzzySet):
        """sets consequent by a certain fuzzy set"""
        self.consequent = fuzzySet
        self.consequent_name = name
#         self.consequent.plot()
        
    def add_rule(self, ant1, ant2, con):
        """adds new rule to rules"""
        i = self.antecedant_1.getNameIdx(ant1)
        j = self.antecedant_2.getNameIdx(ant2)
        self.rules[i,j] = con
        
        
    def set_antecedants(self, ant1, ant2):
        """fuzzifies anti through antecedant_i dimension for i=1,2 and returns result 
        for every membership function """
        res1 = self.antecedant_1.fuzzify(ant1)[:,np.newaxis]
        res2 = self.antecedant_2.fuzzify(ant2)
        results1 = np.zeros(shape=self.rules.shape)
        results2 = np.zeros(shape=self.rules.shape)
        results1 += res1
        results2 += res2
        self.result = np.minimum(results1, results2)
        
    
    def get_consequent(self, plot=False):
        """returns defuzzified consequent determined by set values in set_antecedants"""
        xLen = self.consequent.getXlen()
        X = self.consequent.getX()
        working_rules = self.rules[self.result.nonzero()]
        results = self.result[self.result.nonzero()]
        cons = np.zeros(xLen)
        for name, res in zip(working_rules, results):
            temp = np.ones(xLen) * res

            Y = np.minimum(self.consequent.getSet(name), temp)
#             Y = np.multiply(self.consequent.getSet(name), temp)

            cons = np.maximum(Y, cons)
        if plot:
            plt.plot(X, cons)
        prod = np.sum(cons * X)
        return prod / np.sum(cons)

    def read_rules_from_file(self, file_name='fuzzy_rules.txt'):
        """reads rules from file"""
        f = open(file_name, 'r')
        rules = []
        for line in f:
            rules.append(line.split(';')[:-1])
        f.close()
        for rule in rules:
            self.add_rule(rule[0], rule[1], rule[2])
            
    def print_rules(self):
        """print matrix of rules"""
        ant_names_1 = self.antecedant_1.getNames()
        ant_names_2 = self.antecedant_2.getNames()

        print('{:<3}'.format(''), end='')
        for name in ant_names_2:
            print('{:<3}'.format(name), end='')
        print()
        for i in range(len(self.rules)):
            print('{:<3}'.format(ant_names_1[i]), end='')
            for j in range(len(self.rules[i])):
                print('{:<3}'.format(self.rules[i,j]), end='')
            print()
