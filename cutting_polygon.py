
def negative_list(lista):
    return_list = []
    #print(f"lista = {lista}")
    for i in range(len(lista)):
        return_list.append((-lista[i][0], -lista[i][1]))
    return return_list


def cut_polygon_with_horizontal_line(polygon, y_line, line_is_min_value = False, log_steps = False):
    new_polygons = []

    if line_is_min_value:
        polygon = negative_list(polygon)   
    
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        if log_steps:
            print(polygon)
            print(f"p1 = {p1}")
            print(f"p2 = {p2}")
        
        if (p1[1] > y_line and p2[1] < y_line):
            x_intersection = p1[0] + (y_line - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
            intersection_point = (x_intersection, y_line)
            new_polygons.append(intersection_point)
            if log_steps:
                print(f"{intersection_point} appended")
            
            if p2 not in new_polygons:
                new_polygons.append(p2)
                if log_steps:
                    print(f"p2={p2} appended")
            

        elif  (p1[1] < y_line and p2[1] > y_line):
            x_intersection = p1[0] + (y_line - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
            intersection_point = (x_intersection, y_line)
            
            if p1 not in new_polygons:
                new_polygons.append(p1)
                if log_steps:
                    print(f"p1={p1} appended")
            
                
            new_polygons.append(intersection_point)
            if log_steps:
                print(f"{intersection_point} appended")

        elif (p1[1] < y_line and p2[1] < y_line):
            if p1 not in new_polygons:
                new_polygons.append(p1)
                if log_steps:
                    print(f"p1={p1} appended")

            if p2 not in new_polygons:
                new_polygons.append(p2)
                if log_steps:
                    print(f"p2={p2} appended")

        if log_steps:    
           print()
    
    if line_is_min_value:
        new_polygons = negative_list(new_polygons)
            
    return new_polygons


def cut_polygon_with_vertical_line(polygon, y_line, line_is_min_value = False, log_steps = False):
    new_polygons = []

    if line_is_min_value:
        polygon = negative_list(polygon)   

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        if log_steps:
            print(polygon)
            print(f"p1 = {p1}")
            print(f"p2 = {p2}")
        
        if (p1[0] > y_line and p2[0] < y_line):
            x_intersection = p1[1] + (y_line - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
            intersection_point = (y_line, x_intersection)
            new_polygons.append(intersection_point)
            if log_steps:
                print(f"{intersection_point} appended")
            
            if p2 not in new_polygons:
                new_polygons.append(p2)
                if log_steps:
                    print(f"p2={p2} appended")
            

        elif  (p1[0] < y_line and p2[0] > y_line):
            x_intersection = p1[1] + (y_line - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
            intersection_point = (y_line, x_intersection)
            
            if p1 not in new_polygons:
                new_polygons.append(p1)
                if log_steps:
                    print(f"p1={p1} appended")
                
            new_polygons.append(intersection_point)
            #print(f"{intersection_point} appended")

        elif (p1[0] < y_line) and (p2[0] < y_line):
            if p1 not in new_polygons:
                new_polygons.append(p1)
                if log_steps:
                    print(f"p1={p1} appended")

            if p2 not in new_polygons:
                new_polygons.append(p2)
                if log_steps:
                    print(f"p2={p2} appended")
        
        if log_steps:
            print()

    if line_is_min_value:
        new_polygons = negative_list(new_polygons)

    return new_polygons

def cut_polygon_between_2_horizontal_line(polygon,y_min,y_max, log_steps=False):
    x = cut_polygon_with_horizontal_line(polygon,-y_min, True, log_steps)
    return cut_polygon_with_horizontal_line(x,y_max, log_steps=log_steps)

def cut_polygon_between_2_vertical_line(polygon,y_min,y_max, log_steps=False):
    x = cut_polygon_with_vertical_line(polygon,-y_min,True, log_steps)
    return cut_polygon_with_vertical_line(x,y_max, log_steps=log_steps)
