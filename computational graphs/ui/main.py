from tkinter import *
from tkinter import font 

root = Tk()

cgraph_menu = Menu(root)
root.config(menu=cgraph_menu)

all_font = font.Font(root, family='Helvetica', size=16, weight="bold" )

#Menu items

file_menu = Menu(cgraph_menu)
cgraph_menu.add_cascade(label="File", menu=file_menu)

edit_menu = Menu(cgraph_menu)
cgraph_menu.add_cascade(label="Edit", menu=edit_menu)

view_menu = Menu(cgraph_menu)
cgraph_menu.add_cascade(label="View", menu=view_menu)

#File Menu commands
def new_command():
    pass

def save_command():
    pass

#Edit menu commands

def edit_file():
    pass

#View Menu commands
def side_msg(flag):
    if flag:
        return "Hide Tools Panel"
    else:
        return "View Tools Panel"


def view_side_panel(flag):
    global VIEW_SIDE_PANEL_STATE
    if flag == False:
        frame.grid(row=0, column=0, padx=1, pady=1,sticky="N")
        VIEW_SIDE_PANEL_STATE = True
        view_menu.entryconfigure(1, label=side_msg(VIEW_SIDE_PANEL_STATE))
    else:
        frame.grid_forget()
        VIEW_SIDE_PANEL_STATE = False
        view_menu.entryconfigure(1, label=side_msg(VIEW_SIDE_PANEL_STATE))


# add file menu commands

file_menu.add_command(label="New", command=new_command)
file_menu.add_separator()
file_menu.add_command(label="Save", command=save_command)

# add edit menu commands

edit_menu.add_command(label="Edit File", command=edit_file)

# add view menu commands
global VIEW_SIDE_PANEL_STATE 
VIEW_SIDE_PANEL_STATE = True

view_menu.add_command(label=side_msg(VIEW_SIDE_PANEL_STATE), command=lambda: view_side_panel(VIEW_SIDE_PANEL_STATE))


# frame  = LabelFrame(root, text="Tools", padx=1, pady=1)
# frame.grid(row=0, column=0, padx=1, pady=1,sticky="N")



import node 


class SidePanel(object):
    def __init__(self, root, text, node_texts, node_commands, **kwargs):
        self.root = root

        self.node_texts = node_texts
        self.node_commands = node_commands

        self.text = text

        self.frame = LabelFrame(self.root, text=self.text, padx = 1, pady = 1)
        
        if kwargs:
            self.frame.grid(**kwargs)
        else:
            self.frame.grid(row=0, column=0, padx=1, pady=1, sticky="N")

        self.font = font.Font(self.root, family='Courier', size=16, weight="bold" )

        self.set_node_buttons()

    def set_node_buttons(self):
        
        self.node_buttons = [Button(self.frame, text=t, font = self.font, command=c) 
        for t,c in zip(self.node_texts, self.node_commands)]
        
        for i,nb in enumerate(self.node_buttons):
            nb.grid(row=int(i/2), column=i%2)


def add_weights():
    node.Variable(canvas)

def add_input():
    node.Input(canvas)

def add_plus():
    node.Operation(canvas, "+")

def add_minus():
    node.Operation(canvas, "-")

def add_tensor_dot():
    node.Operation(canvas, "*")

def add_exponent():
    node.Operation(canvas, "^")

global CANVAS_MOUSE_STATE
CANVAS_MOUSE_STATE = 'SELECT'

def delete_state():
    global CANVAS_MOUSE_STATE
    CANVAS_MOUSE_STATE = 'DELETE'

def select_state():
    global CANVAS_MOUSE_STATE
    CANVAS_MOUSE_STATE = 'SELECT'



nodes_sp = SidePanel(
    root,
    "Tools", 
    ["W","x","+","-","*","^"], 
    [add_weights, add_input, add_plus, add_minus, add_tensor_dot, add_exponent] )

utils_sp = SidePanel(
    root,
    "",
    ["del","sel"],
    [delete_state,select_state],
    row=1,
    column=0,
    padx = 1,
    pady = 1,
    sticky = 'N'
)

### Tools menu


# w = Button(frame, text="W", font=all_font, command=add_weights)
# w.grid(column=0, row=0, sticky="N")
# x = Button(frame, text="x", font=all_font, command=add_input)
# x.grid(column=0, row=1, sticky="N")
# del_button = Button(frame, text="Delete", command=delete_node)




global canvas


canvas = Canvas(root, width=600, height=400, borderwidth=2, background="gray")
canvas.grid(row=0,column=1,rowspan=2)




#oval = canvas.create_rectangle(100, 100, 110, 150, fill="black", outline='white')
#rectangle = canvas.create_rectangle(10, 100, 110, 150, outline="white", fill="black" )


class MouseMover():
    def __init__(self):
        
        self.item = 0; self.previous = (0, 0)

    def select(self, event):
        global CANVAS_MOUSE_STATE
        self.mouse_state = CANVAS_MOUSE_STATE
        widget = event.widget                       # Get handle to canvas 
        xc = widget.canvasx(event.x); yc = widget.canvasx(event.y)
        self.item = widget.find_closest(xc, yc)[0]        # ID for closest
        
        if self.mouse_state == 'SELECT':
            # Convert screen coordinates to canvas coordinates
            
            self.previous = (xc, yc)
            print((xc, yc, self.item))

        if self.mouse_state == 'DELETE':            
            for item in canvas.find_all():
                bbox = canvas.bbox(item)
                if bbox[0] < xc and bbox[2] > xc and bbox[1] < yc and bbox[3] > yc:
                    node_circle, node_text = node.find_node_components(self.item)
                    node.delete_node(item)
                    canvas.delete(node_circle)
                    canvas.delete(node_text)
                    break
        else:
            pass            

                #canvas.itemconfig(item, outline='white')

            
    def drag(self, event):
        widget = event.widget
        xc = widget.canvasx(event.x); yc = widget.canvasx(event.y)
        try:
            bbox = canvas.bbox(self.item)
            if bbox[0] < xc and bbox[2] > xc and bbox[1] < yc and bbox[3] > yc:
                node_circle, node_text = node.find_node_components(self.item) 
                print(node_circle, node_text)
                canvas.move(node_circle, xc-self.previous[0], yc-self.previous[1])
                canvas.move(node_text, xc-self.previous[0], yc-self.previous[1])
                self.previous = (xc, yc)
        
        except Exception as e:
            pass
        
        else:
            pass

    def motion(self, event):
        global CANVAS_MOUSE_STATE
        self.mouse_state = CANVAS_MOUSE_STATE

        if self.mouse_state == 'DELETE':
            canvas.config(cursor="X_cursor")
        if self.mouse_state == 'SELECT':
            canvas.config(cursor="arrow")

        x, y = event.x, event.y
        for item in canvas.find_all():
            bbox = canvas.bbox(item)
            if bbox[0] < x and bbox[2] > x and bbox[1] < y and bbox[3] > y:
                if self.mouse_state == 'SELECT':
                    canvas.config(cursor="hand2")
                #canvas.itemconfig(item, outline='red')
                



# Get an instance of the MouseMover object
mm = MouseMover()

# Bind mouse events to methods (could also be in the constructor)
canvas.bind("<Button-1>", mm.select)
canvas.bind("<B1-Motion>", mm.drag)
canvas.bind('<Motion>', mm.motion)




root.mainloop()

