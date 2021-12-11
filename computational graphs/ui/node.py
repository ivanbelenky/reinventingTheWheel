from abc import ABC, abstractmethod
from tkinter import *
from tkinter import font 

global NODES 
NODES = []

def find_node_components(idx):
        global NODES
        for i,node in enumerate(NODES):
            if node['circle'] == idx or node['label'] == idx:
                #this should be generalized just to any component forming the figure in question
                return node['circle'], node['label']

def delete_node(idx):
    global NODES
    for i,node in enumerate(NODES):
        if node['circle'] == idx or node['label'] == idx:
            NODES.remove({'circle':node['circle'], 'label':node['label']})


class Node(ABC):
    def __init__(self, canvas, x=None, y=None):
        self.canvas = canvas
        self.x = x if x else self.canvas.winfo_width()/2
        self.y = y if y else self.canvas.winfo_height()/2
        self.font = font.Font(self.canvas, family='Helvetica', size=16, weight="bold" )
        global NODES
        self.nodes = NODES 

    @abstractmethod        
    def create(self):
        pass

    @abstractmethod
    def edit(self):
        pass

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def delete(self):
        pass


    


VAR_RADII = 25
INPUT_RADII = 20
OP_RADII = 15


class Variable(Node):
    def __init__(self, canvas, x=None, y=None):
        super().__init__(canvas, x=x, y=y)
        self.radii = VAR_RADII

        # TODO IMplement edit
        #_ = self.edit()
        # self.name = _.name
        #  
        self.create()
        

    def edit(self):
        pass
    
    def create(self):
        self.circle_id = self.canvas.create_oval(self.x-self.radii,self.y-self.radii,self.x+self.radii,self.y+self.radii, fill="black", outline="white")        
        self.label_x = self.label_y = int(self.x+self.radii/2)
        #self.edit returns text, for now, all x nodes are the same
        self.label_id = self.canvas.create_text(self.x, self.y, text="W", font=self.font, fill="white")
        global NODES 
        NODES.append({'circle':self.circle_id,'label':self.label_id})
        print(NODES)

    def move(self):
        pass

    def delete(self):
        pass



class Input(Node):
    def __init__(self, canvas, x=None, y=None):
        super().__init__(canvas, x=x, y=y)
        self.radii = INPUT_RADII
        
        # TODO IMplement edit
        #_ = self.edit()
        # self.name = _.name
        #  
        
        self.create()
        
    def edit(self):
        pass
    
    def create(self):
        self.circle_id = self.canvas.create_oval(self.x-self.radii,self.y-self.radii,self.x+self.radii,self.y+self.radii, fill="black", outline="white")        
        self.label_x = self.label_y = int(self.x+self.radii/2)
        #self.edit returns text, for now, all x nodes are the same
        self.label_id = self.canvas.create_text(self.x, self.y, text="x", fill="white")
        global NODES 
        NODES.append({'circle':self.circle_id,'label':self.label_id})
        print(NODES)

    def move(self):
        pass

    def delete(self):
        pass



class Operation(Node):
    def __init__(self, canvas, operation, x=None, y=None):
        super().__init__(canvas, x=x, y=y)
        self.name = operation
        self.radii = OP_RADII
        self.operation = operation        
        
        # TODO IMplement edit
        #_ = self.edit()
        # self.name = _.name
        #  
        
        self.create()
        
    def edit(self):
        pass
    
    def create(self):
        self.circle_id = self.canvas.create_oval(self.x-self.radii,self.y-self.radii,self.x+self.radii,self.y+self.radii, fill="black", outline="white")        
        #self.edit returns text, for now, all operation nodes are the same
        self.label_id = self.canvas.create_text(self.x, self.y, text=self.operation, font=self.font, fill="white")
        global NODES 
        NODES.append({'circle':self.circle_id,'label':self.label_id})
        print(NODES)

    def move(self):
        pass

    def delete(self):
        pass

