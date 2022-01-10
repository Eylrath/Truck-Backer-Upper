
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from fuzzy_controller import *
from NeuralNet import *

Π = 3.1415926535
def rad_to_deg(angle):
    """converts angle in radians to degrees"""
    return angle / Π * 180
def deg_to_rad(angle):
    """converts angle in degrees to radians"""
    return angle * Π / 180

class Truck():
    """class implemeting simulation"""
    def __init__(self, display=False):
        """initializes basic properties of simulation, if display=True, truck will be seen, else not"""
        self.width = 3
        self.length = 6
        self.display = display
        self.ax = None
        if self.display:
            self.f = plt.figure(figsize=(3,3))
            self.ax = self.f.add_axes([0.,0.,1.,1.], facecolor='black')
            # ax.axhline()
            self.ax.axvline()
            plt.xlim(0,100)
            plt.ylim(0,100)
            self.patches = list()
            
    def reset(self):
        """resets position by random phi, theta, x, y"""
        self.ϕ = deg_to_rad ( random()* 360 - 90 )
        self.x = random() * 100
        self.y = random() * 100
        self.θ = deg_to_rad( (random() * 60) - 30 )
        if self.display:
            self.draw()
    
    def set_position(self, ϕ=0, x=0, y=0):
        """sets certain x, y, phi"""
        self.x = x
        self.ϕ = deg_to_rad(ϕ)
        self.y = y
        self.θ = deg_to_rad( (random() * 60) - 30 )
        if self.display:
            self.draw()
        
            
    def draw(self):
        """draws actual position"""
        if not self.display:
            return
        self._draw_car()
        self.f.canvas.draw()
    
    def get_attr(self):
        """returns tuple of atributes"""
        return (self.width, self.length, self.x, self.y, self.ϕ, self.θ, self.ax)
    def _draw_car(self):
        """actual method for drawing car"""
        W, L, x, y, ϕ, θ, ax = self.get_attr()
        car = Rectangle(
            (x, y), L, W, ϕ, color='C2', alpha=0.8,
            transform=matplotlib.transforms.Affine2D().rotate_deg_around(x, y, rad_to_deg(ϕ)) +ax.transData
        )
        ax.add_patch(car)
        
    def step(self, θ=0, dt=1):
        """makes step by dt and theta through truck kinematics"""
        self.θ = θ
        W, L, x, y, ϕ, θ, ax = self.get_attr()
        x_new = x + dt*cos(ϕ)
        y_new = y + dt*sin(ϕ)
        ϕ_new = ϕ + θ
        self.x = x_new
        self.y = y_new
        self.ϕ = ϕ_new



def drive(controller, controller_type, initial_position=None, dt=1):
    """function uses controller (indicated by controller_type - string with name of controller)
        drives from initial position (or random if None) by step dt
        possible controller_type:
            neural_1 - unfixed neural net
            neural_2 - fixed neural net
            fuzzy - fuzzy controller"""
    ε = 0.1
    truck = Truck(display=True)
    if initial_position is None:
        truck.reset()
    else:
        truck.set_position(initial_position[0], initial_position[1], initial_position[2])
    if controller_type == 'neural_1':
        while truck.y < 100 - ε:
            ϕ = rad_to_deg(truck.ϕ)
            x = truck.x
            y = truck.y
            inp = np.array([x,y,ϕ])
            θ = sigm_1(controller.forward(inp))
            truck.step(θ,dt)
            truck.draw()
    elif controller_type == "neural_2":
        while truck.y < 100 - ε:
            ϕ = rad_to_deg(truck.ϕ)
            x = truck.x
            y = truck.y
            inp = np.array([x,y,ϕ])
            θ = (sigm_1(controller.forward(inp)) + truck.θ)/4
            truck.step(θ,dt)
            truck.draw()
    elif controller_type == "fuzzy":
        while truck.y < 100 - ε:
            ϕ = rad_to_deg(truck.ϕ)
            x = truck.x
            y = truck.y
            controller.set_antecedants(x, ϕ)
            θ = controller.get_consequent(plot=False)
            θ = deg_to_rad(θ)
            truck.step(θ,dt)
            truck.draw()



