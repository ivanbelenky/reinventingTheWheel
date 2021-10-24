from tkinter import *
from tkinter import font 
root = Tk()

cgraph_menu = Menu(root)
root.config(menu=cgraph_menu)
font
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


frame  = LabelFrame(root, text="Tools", padx=1, pady=1)
frame.grid(row=0, column=0, padx=1, pady=1,sticky="N")

### Tools menu

def add_weights():
    canvas.create_oval(60,60,100,100, fill="black", outline="white")
    canvas.create_text(80,80, text="W", font=all_font, fill="white")

def add_input():
    canvas.create_oval(15,15,45,45, fill="black", outline="white")



w = Button(frame, text="W", font=all_font, command=add_weights)
w.grid(column=0, row=0, sticky="N")
x = Button(frame, text="x", font=all_font, command=add_input)
x.grid(column=0, row=1, sticky="N")

















global canvas

canvas = Canvas(root, width=400, height=200, borderwidth=2, background="gray")
canvas.grid(row=0,column=1)

#oval = canvas.create_rectangle(100, 100, 110, 150, fill="black", outline='white')
#rectangle = canvas.create_rectangle(10, 100, 110, 150, outline="white", fill="black" )

class MouseMover():
    def __init__(self):
        self.item = 0; self.previous = (0, 0)
    
    def select(self, event):
        widget = event.widget                       # Get handle to canvas 
        # Convert screen coordinates to canvas coordinates
        xc = widget.canvasx(event.x); yc = widget.canvasx(event.y)
        self.item = widget.find_closest(xc, yc)[0]        # ID for closest
        self.previous = (xc, yc)
        print((xc, yc, self.item))
        canvas.config(cursor="hand2")
        

    def drag(self, event):
        widget = event.widget
        xc = widget.canvasx(event.x); yc = widget.canvasx(event.y)
        canvas.move(self.item, xc-self.previous[0], yc-self.previous[1])
        self.previous = (xc, yc)

    def motion(self, event):
        x, y = event.x, event.y
        enclosed = canvas.find_enclosed
        for item in canvas.find_all():
            bbox = canvas.bbox(item)
            print(canvas.type(item))
            if bbox[0] < x and bbox[2] > x and bbox[1] < y and bbox[3] > y:
                canvas.config(cursor="hand2")
                #canvas.itemconfig(item, outline='red')
            else:
                canvas.config(cursor="")
                #canvas.itemconfig(item, outline='white')
                






# Get an instance of the MouseMover object
mm = MouseMover()

# Bind mouse events to methods (could also be in the constructor)
canvas.bind("<Button-1>", mm.select)
canvas.bind("<B1-Motion>", mm.drag)
canvas.bind('<Motion>', mm.motion)

root.mainloop()

