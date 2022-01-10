import numpy as np
from truck_system import *
from fuzzy_set import *
from fuzzy_controller import *
import csv

def generate_data(truck_sim, x_num, y_num, phi_num, file_name, disp=False):
    """generates data from fuzzy truck controller (truck_sim) by specified number of dimensions"""
    ε = 0.1
    i=0
    file = open(file_name, 'w')
    for x_ in np.linspace(10,70,x_num):
        for y_ in np.linspace(10,70,y_num):
            for phi_ in np.linspace(-90,270,phi_num):#[-90, -45, 0, 45, 90, 135, 180, 225, 270]:
                # print(i)
                i+=1
                t = Truck(display=disp)
                t.set_position(phi_, x_, y_)
                while t.y < 100 - ε:
                    ϕ = rad_to_deg(t.ϕ)
                    x = t.x
                    truck_sim.set_antecedants(x, ϕ)
                    θ = truck_sim.get_consequent()
                                   
                    file.write(str(x)+';'+str(t.y)+';'+str(ϕ)+';'+str(θ)+'\n')
                    θ = deg_to_rad(θ) 
                    t.step(θ,1)
                    t.draw()
    file.close()

def load_data(file_name):
    """returns loaded data from file"""
    L = []
    with open(file_name,newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for line in reader:
            if None not in line:
                L.append(np.array(line).astype(np.float))
    L = np.array(L)
    return L[:,:3], L[:,3:]
