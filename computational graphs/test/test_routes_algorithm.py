import numpy as np
import copy 
    

def get_routes(start, end, routes, actual_route):
    actual_route.append(start.id)

    if start == end:
        routes.append(copy.deepcopy(actual_route))
        actual_route.remove(start.id)
        return 

    elif not start.consumers:
        actual_route.remove(start.id)
        return 

    else:
        for consumer in start.consumers:
            get_routes(consumer, end, routes, actual_route)

    actual_route.remove(start.id)

    return 

class Consumed(object):
    def __init__(self, inputs, id) -> None:
        self.consumers = []
        self.id = id
        for input in inputs:
            input.consumers.append(self)


if __name__ == "__main__":

    var_1 = Consumed([],1)
    var_2 = Consumed([var_1],2)
    var_4 = Consumed([var_2],4)
    var_5 = Consumed([var_2],5)
    var_6 = Consumed([var_4, var_5],6)
    var_7 = Consumed([var_6],7)
    var_8 = Consumed([var_6],8)
    var_10 = Consumed([var_1],10)
    var_11 = Consumed([var_10],11)
    var_12 = Consumed([var_10],12)   
    var_9 = Consumed([var_7,var_12],9)
    
    routes = []
    actual_route = []
    get_routes(var_1, var_9, routes, actual_route) 
    for r in routes:
        print("routes",r)
    print(routes)

