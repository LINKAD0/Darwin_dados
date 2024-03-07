import os
import sys
import time
import math
import cv2
import json
from shapely import Polygon
from PIL import Image, ImageDraw
import IPython
import numpy as np
from tqdm.notebook import  tqdm
from numpy import clip
import matplotlib.pyplot as plt 
from math import pi,sin,cos,tan,atan2,hypot,floor


import multiprocessing
NUM_THREADS = max(1, os.cpu_count() - 1)

def return_files(path_input):

    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(path_input):
        # check if current path is a file
        if os.path.isfile(os.path.join(path_input, path)):
            res.append(path)
    return res

def generate_cubmaps(path_input, path_output, multiprocess=False):
    if not os.path.exists(path_output):
        os.makedirs(path_output) 

    # list to store files
    res = return_files(path_input)
    if multiprocess:
        pool = multiprocessing.Pool(NUM_THREADS)

    for i in range (0, len(res)):
        print('image: ', res[i])
        
        if multiprocess:
            pool.apply_async(generate_cubemap_single,
                                args=(path_input + res[i], path_output + res[i], i + 1))
        else:
            generate_cubemap_single(path_input + res[i], path_output + res[i])

    pool.close()
    pool.join()


def generate_cubemap_single(img_path, output_path):
    
    imgIn = Image.open(img_path)
    print(imgIn.size)
    inSize = imgIn.size
    imgOut = Image.new("RGB",(inSize[0], int(inSize[0]*3/4)),"black")
    
    convertBack(imgIn,imgOut)
    print(imgOut.size)
    imgOut.save(output_path)

# get x,y,z coords from out image pixels coords
# i,j are pixel coords
# face is face number
# edge is edge length
def outImgToXYZ(i,j,face,edge):
    a = 2.0*float(i)/edge
    b = 2.0*float(j)/edge
    if face==0: # back
        (x,y,z) = (-1.0, 1.0-a, 3.0 - b)
    elif face==1: # left
        (x,y,z) = (a-3.0, -1.0, 3.0 - b)
    elif face==2: # front
        (x,y,z) = (1.0, a - 5.0, 3.0 - b)
    elif face==3: # right
        (x,y,z) = (7.0-a, 1.0, 3.0 - b)
    elif face==4: # top
        (x,y,z) = (b-1.0, a -5.0, 1.0)
    elif face==5: # bottom
        (x,y,z) = (5.0-b, a-5.0, -1.0)
    return (x,y,z)

# convert using an inverse transformation
def convertBack(imgIn,imgOut):




    start = time.time()
    inSize = imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    edge = inSize[0]/4   # the length of each edge in pixels
    for i in tqdm(range(outSize[0])):
        face = int(i/edge) # 0 - back, 1 - left 2 - front, 3 - right
        #print('face: ', face)
        if face==2:
            rng = range(0,int(edge*3))
        else:
            rng = range(int(edge), int(edge) * 2)

        for j in rng:
            if j<edge:
                face2 = 4 # top
            elif j>=2*edge:
                face2 = 5 # bottom
            else:
                face2 = face

            (x,y,z) = outImgToXYZ(i,j,face2,edge)
            theta = atan2(y,x) # range -pi to pi
            r = hypot(x,y)
            phi = atan2(z,r) # range -pi/2 to pi/2
            # source img coords
            uf = ( 2.0*edge*(theta + pi)/pi )
            vf = ( 2.0*edge * (pi/2 - phi)/pi)
            # Use bilinear interpolation between the four surrounding pixels
            ui = floor(uf)  # coord of pixel to bottom left
            vi = floor(vf)
            u2 = ui+1       # coords of pixel to top right
            v2 = vi+1
            mu = uf-ui      # fraction of way across pixel
            nu = vf-vi
            # Pixel values of four corners
            A = inPix[ui % inSize[0],int(clip(vi,0,inSize[1]-1))]
            B = inPix[u2 % inSize[0],int(clip(vi,0,inSize[1]-1))]
            C = inPix[ui % inSize[0],int(clip(v2,0,inSize[1]-1))]
            D = inPix[u2 % inSize[0],int(clip(v2,0,inSize[1]-1))]
            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            outPix[i,j] = (int(round(r)),int(round(g)),int(round(b)))

    end = time.time()
    print('elapsed time: ', end-start)
    
def split_cubmaps(infile, outfile):
    if not os.path.exists(outfile):
        os.makedirs(outfile)
        
    if len(sys.argv) < 2:
        print("Usage: cubemap-cut.py <filename.jpg|png>")
        sys.exit(-1)

    #infile = root_path + '000009_cub.png'
    filename, original_extension = os.path.splitext(infile)
    file_extension = ".png"

    name_map = [ \
         ["", "", "posy", ""],
         ["negz", "negx", "posz", "posx"],
         ["", "", "negy", ""]]

    try:
        im = Image.open(infile)
        print(infile, im.format, "%dx%d" % im.size, im.mode)

        width, height = im.size

        cube_size = width / 4

        filelist = []
        for row in range(3):
            for col in range(4):
                if name_map[row][col] != "":
                    sx = cube_size * col
                    sy = cube_size * row
                    fn = name_map[row][col] + file_extension
                    filelist.append(fn)
                    print("%s --> %s" % (str((sx, sy, sx + cube_size, sy + cube_size)), fn))
                    im.crop((sx, sy, sx + cube_size, sy + cube_size)).save(outfile + '/' + fn) 
    except IOError:
        pass

def split_cub_imgs(path_input, path_output):
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    
    file_list = return_files(path_input)
    for i in range (0, len(file_list)):
        print(file_list[i])
        infile = file_list[i]
        split_cubmaps(path_input + infile, path_output + infile.split('.')[0])
        
def join_images(path_segmentation):
    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(path_segmentation):
        # check if current path is a file
        if os.path.isfile(os.path.join(path_segmentation, path)):
            res.append(path)
    print(res)

    vec_img = []
    for i in range(0, 6):
        #print(i)
        vec_img.append(Image.open(path_segmentation + res[i]))

    name_map = [ \
         ["", "posy", "", ""],
         ["negz", "negx", "posz", "posx"],
         ["", "negy", "", ""]]

    width, height = np.array(vec_img[0]).shape
    print(width, height)

    cube_size = np.zeros((width * 3 , width * 4))
    print(cube_size.shape)

    cube_size[0:width, width:width*2] = np.array(vec_img[4])
    cube_size[width:width*2, 0:width] = np.array(vec_img[0])
    cube_size[width:width*2, width:width*2] = np.array(vec_img[5])
    cube_size[width:width*2, width*2:width*3] = np.array(vec_img[3])
    cube_size[width:width*2, width*3:width*4] = np.array(vec_img[2])
    cube_size[2*width:width*3, width:width*2] = np.array(vec_img[1])
    return cube_size

def spherical_coordinates(i, j, w, h):
    """ Returns spherical coordinates of the pixel from the output image. """
    theta = 2*float(i)/float(w)-1
    phi = 2*float(j)/float(h)-1
    # phi = lat, theta = long
    return phi*(math.pi/2), theta*math.pi


def vector_coordinates(phi, theta):
    """ Returns 3D vector which points to the pixel location inside a sphere. """
    return (math.cos(phi) * math.cos(theta),  # X
            math.sin(phi),                    # Y
            math.cos(phi) * math.sin(theta))  # Z


# Assign identifiers to the faces of the cube
FACE_Z_POS = 1  # Left
FACE_Z_NEG = 2  # Right
FACE_Y_POS = 3  # Top
FACE_Y_NEG = 4  # Bottom
FACE_X_NEG = 5  # Front
FACE_X_POS = 6  # Back


def get_face(x, y, z):
    """ Uses 3D vector to find which cube face the pixel lies on. """
    largest_magnitude = max(abs(x), abs(y), abs(z))
    if largest_magnitude - abs(x) < 0.00001:
        return FACE_X_POS if x < 0 else FACE_X_NEG
    elif largest_magnitude - abs(y) < 0.00001:
        return FACE_Y_POS if y < 0 else FACE_Y_NEG
    elif largest_magnitude - abs(z) < 0.00001:
        return FACE_Z_POS if z < 0 else FACE_Z_NEG


def raw_face_coordinates(face, x, y, z):
    """
    Return coordinates with necessary sign (- or +) depending on which face they lie on.
    From Open-GL specification (chapter 3.8.10) https://www.opengl.org/registry/doc/glspec41.core.20100725.pdf
    """
    if face == FACE_X_NEG:
        xc = z
        yc = y
        ma = x
        return xc, yc, ma
    elif face == FACE_X_POS:
        xc = -z
        yc = y
        ma = x
        return xc, yc, ma
    elif face == FACE_Y_NEG:
        xc = z
        yc = -x
        ma = y
        return xc, yc, ma
    elif face == FACE_Y_POS:
        xc = z
        yc = x
        ma = y
        return xc, yc, ma
    elif face == FACE_Z_POS:
        xc = x
        yc = y
        ma = z
        return xc, yc, ma
    elif face == FACE_Z_NEG:
        xc = -x
        yc = y
        ma = z
        return xc, yc, ma


def raw_coordinates(xc, yc, ma):
    """ Return 2D coordinates on the specified face relative to the bottom-left corner of the face. Also from Open-GL spec."""
    return (float(xc)/abs(float(ma)) + 1) / 2, (float(yc)/abs(float(ma)) + 1) / 2


def face_origin_coordinates(face, n):
    """ Return bottom-left coordinate of specified face in the input image. """
    if face == FACE_X_NEG:
        return n, n
    elif face == FACE_X_POS:
        return 3*n, n
    elif face == FACE_Z_NEG:
        return 2*n, n
    elif face == FACE_Z_POS:
        return 0, n
    elif face == FACE_Y_POS:
        return n, 0
    elif face == FACE_Y_NEG:
        return n, 2*n


def normalized_coordinates(face, x, y, n):
    """ Return pixel coordinates in the input image where the specified pixel lies. """
    face_coords = face_origin_coordinates(face, n)
    normalized_x = math.floor(x*n)
    normalized_y = math.floor(y*n)

    # Stop out of bound behaviour
    if normalized_x < 0:
        normalized_x = 0
    elif normalized_x >= n:
        normalized_x = n-1
    if normalized_y < 0:
        normalized_x = 0
    elif normalized_y >= n:
        normalized_y = n-1

    return face_coords[0] + normalized_x, face_coords[1] + normalized_y


def find_corresponding_pixel(i, j, w, h, n):
    """
    :param i: X coordinate of output image pixel
    :param j: Y coordinate of output image pixel
    :param w: Width of output image
    :param h: Height of output image
    :param n: Height/Width of each square face
    :return: Pixel coordinates for the input image that a specified pixel in the output image maps to.
    """

    spherical = spherical_coordinates(i, j, w, h)
    vector_coords = vector_coordinates(spherical[0], spherical[1])
    face = get_face(vector_coords[0], vector_coords[1], vector_coords[2])
    raw_face_coords = raw_face_coordinates(face, vector_coords[0], vector_coords[1], vector_coords[2])

    cube_coords = raw_coordinates(raw_face_coords[0], raw_face_coords[1], raw_face_coords[2])

    return normalized_coordinates(face, cube_coords[0], cube_coords[1], n)

def convert_img(infile, outfile):
    inimg = Image.open(infile)
    inimg = Image.fromarray(np.flip(np.array(inimg),1))
    wo, ho = inimg.size
    print('image size: ', wo, ho)

    # Calculate height and width of output image, and size of each square face
    #h = int(wo/3)
    #w = int(2*h)
    h = 2688
    w = 5376
    n = ho/3
    print(w,h)


    progress_bar = tqdm(total=h*w, unit=" iteration")

    # Create new image with width w, and height h
    outimg = Image.new('RGB', (w, h))

    # For each pixel in output image find colour value from input image
    for ycoord in range(0, h):
        for xcoord in range(0, w):
            corrx, corry = find_corresponding_pixel(xcoord, ycoord, w, h, n)

            outimg.putpixel((xcoord, ycoord), inimg.getpixel((corrx, corry)))
            progress_bar.update(1)
        # Print progress percentage
        #print(str(round((float(ycoord)/float(h))*100, 2)) + '%')

    progress_bar.close()
    outimg.save(outfile, 'PNG')



# utils by Pedro Guedes

def imshow(img):
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret,width=1000,height=1000)
    IPython.display.display(i)

    
def classname_to_color_mapper(class_name, alpha_value=150, error_value=(0,0,0,255)):
    if class_name == "Guarda Corpo":
        return (255, 153, 0, alpha_value)
    if class_name == "Escadas":
        return (0, 255, 0, alpha_value)
    if class_name.lower() in "equipamento.":
        return (250, 0, 0, alpha_value)
    if class_name == "Teto":
        return (102, 0, 102, alpha_value)
    if class_name == "Piso":
        return (255, 102, 204, alpha_value)
    if class_name == "Suportes":
        return (0, 51, 0, alpha_value)
    if class_name == "Estruturas":
        return (0, 0, 255, alpha_value)
    if class_name == "TVF":
        return (102, 51, 0, alpha_value)
    if "corrosão" in class_name.lower():
        return (255,0,255, alpha_value)
    print(f"Classe {class_name} não encontrada no catalogo")
    return error_value

FACES_LIST = ['_posx','_negx','_posy','_negy','_posz','_negz']

def visualize_annotations(img_name, json_name=None, dont_show_ground_truth = False):
    img_name = img_name
    if img_name[-5:].lower() in FACES_LIST:
        raise ValueError(f"img_name {img_name} must not end with face_name")

    
    face_path_list = [img_name + FACES_LIST[i] for i in range(len(FACES_LIST))]


    image = Image.open(f"SISTEMAS_UFF/Images/{face_path_list[0]}.png")
    width,height = image.size

    complete_image = np.zeros((height*3,width*4,4))

    image = image.convert("RGBA")

    # Create a new image with an alpha channel
    image_with_alpha = Image.new("RGBA", image.size)

    # Create a drawing context for the image with an alpha channel
    draw = ImageDraw.Draw(image_with_alpha)

    if img_name.endswith("posx"):
        pass
    elif img_name.endswith("negx"):
        pass
    elif img_name.endswith("negy"):
        image = image.rotate(270)
        pass
    elif img_name.endswith("posy"):
        image = image.rotate(90)
        pass
    elif img_name.endswith("posz"):
        pass
    elif img_name.endswith("negz"):
        pass

    alpha_value = 150
    outline_width = 3

    #print(image.size)
    for face_name in face_path_list:
        
        image = Image.open(f"SISTEMAS_UFF/Images/{face_name}.png")
        image = image.convert("RGBA")

        if face_name.endswith("PosX"):
            pass
        elif face_name.endswith("NegX"):
            pass
        elif face_name.endswith("negy"):
            image = image.rotate(270)
            pass
        elif face_name.endswith("posy"):
            image = image.rotate(90)
            pass
        elif face_name.endswith("PosZ"):
            pass
        elif face_name.endswith("NegZ"):
            pass

        img = np.array(image) 
        
        height_pos = 0
        width_pos = 0
        if face_name.endswith("posx"):
            height_pos = 1
            width_pos = 2
            pass
        elif face_name.endswith("negx"):
            height_pos = 1
            width_pos = 0
            pass
        elif face_name.endswith("negy"):
            height_pos = 0
            width_pos = 1
            pass
        elif face_name.endswith("posy"):
            height_pos = 2
            width_pos = 1
            pass
        elif face_name.endswith("posz"):
            height_pos = 1
            width_pos = 1
            pass
        elif face_name.endswith("negz"):
            height_pos = 1
            width_pos = 3
            pass
        complete_image[height*height_pos:height*(height_pos+1),width*width_pos:width*(width_pos+1),:] = img

    complete_image = Image.fromarray(np.uint8(complete_image))

    if json_name is None:
        data = json.load(open(f"SISTEMAS_UFF/export-v3/{img_name[8:]}.json"),encoding='utf-8')
    else:
        data = json.load(open(json_name),encoding='utf-8')
    for object in data["annotations"]:

        # parse the polygon point list to a list of lists
        polygon_point_list = []
        for point_list in object["polygon"]["paths"]:
            polygon_point_list.append([(point_list[i]["x"], point_list[i]["y"]) for i in range(len(point_list))])

        # define class colors
        fill_color = classname_to_color_mapper(object["name"], error_value=(255,0,255,150))
        outline_color = classname_to_color_mapper(object["name"],255)

        # create the layer in which annotation are going to be drawn
        image_with_alpha = Image.new("RGBA", complete_image.size)
        image_with_alpha.putalpha(0)
        draw = ImageDraw.Draw(image_with_alpha)
        
        # create a list of polygons object to calculate holes
        poly_list = [Polygon(polygon_point_list[i]) for i in range(len(polygon_point_list))]
        #print(f"number of polygons in object: {len(poly_list)}")

        # the holes will be drawn at the end
        holes_list_to_be_draw_later = []
        
        # for each polygon inside a object
        for i in range(len(polygon_point_list)):
            polygon_to_draw = polygon_point_list[i]
            polya = poly_list[i]
            area_superposition_count = 0

            # for each polygon, how many polygons is it inside of
            for polyb in poly_list:
                if polya != polyb:
                    try:
                        polyb_contains_polya = polyb.contains(polya)
                        if polyb_contains_polya:
                            area_superposition_count+=1
                    except:
                        print("Some error ocorred with:")
                        print(polyb)
                        print(polya)
                        print("---------------------------------------------------------------")
                        #fill_color = (0,0,0,255)
                        #raise TypeError("Esse erro")
        
            classe = 0
            #if is a hole, drawn later
            if area_superposition_count%2==1:
                classe+=8
                holes_list_to_be_draw_later.append(polygon_to_draw)

                # drawing the main polygons
            else:
                draw.polygon(polygon_to_draw, fill=(fill_color), outline=(outline_color), width=outline_width)
            
        #print(f"number of holes: {len(holes_list_to_be_draw_later)}/{len(poly_list)}")
        #print(f"")

        # drawing the holes
        for hole in holes_list_to_be_draw_later:

            draw.polygon(hole, fill=0, outline=(outline_color), width=outline_width)
        complete_image = Image.alpha_composite(complete_image, image_with_alpha)
                
            #break
            

    complete_image = np.array(complete_image) 

    complete_image[..., :3] = complete_image[..., 2::-1]

    if not dont_show_ground_truth:
        imshow(complete_image)
    return complete_image

def classname_to_color_mapper_mask(class_name, alpha_value=150, error_value=255):
    if class_name.lower() in "equipamento.":
        return 1
    if class_name == "Escadas":
        return 2
    if class_name == "Estruturas":
        return 3
    if class_name == "Guarda Corpo":
        return 4
    if class_name == "Piso":
        return 5
    if class_name == "Suportes":
        return 6
    if class_name == "TVF":
        return 7
    if class_name == "Teto":
        return 8
    print(f"Classe {class_name} não encontrada no catalogo")
    if class_name.lower() == "corrosão":
        return 255
    return None



def draw_prediction_on_face_mask(image, result):
    
    if image is not None:
        if isinstance(image, (np.ndarray, np.generic)):
            image = Image.fromarray(image.astype(np.uint8))
        elif torch.is_tensor(image):
            image = transforms.functional.to_pil_image(image)

        image = image.convert("RGBA")

        # Create a new image with an alpha channel
        image_with_alpha = Image.new("L", image.size)

        # Create a drawing context for the image with an alpha channel
        draw = ImageDraw.Draw(image_with_alpha)
            
        if len(result) > 0:

            masks = result.masks.xy  # Masks object for segmentation masks outputs
            classes = result.boxes.cls  # Masks object for segmentation masks outputs
            for i in range(len(masks)):
                mask = masks[i]
                classe_index = int(classes[i].item())
                classe_name = result.names[classe_index]
    
                fill_color = classname_to_color_mapper_mask(classe_name)
                outline_color = classname_to_color_mapper_mask(classe_name,255)

                polygon_point_list = []
                for point in mask:
                    x,y = point
                    polygon_point_list.append((x,y))
                    
                if len(polygon_point_list) > 0:
                    draw.polygon(polygon_point_list, fill_color, outline_color)

        annotations_only = np.array(image_with_alpha) 
        annotations_only[..., :3] = annotations_only[..., 2::-1]
        return annotations_only



def draw_prediction_on_face(image, result, mask_only=False):
    
    if image is not None:
        if isinstance(image, (np.ndarray, np.generic)):
            image = Image.fromarray(image.astype(np.uint8))
        elif torch.is_tensor(image):
            image = transforms.functional.to_pil_image(image)

        image = image.convert("RGBA")

        # Create a new image with an alpha channel
        image_with_alpha = Image.new("RGBA", image.size)
        image_with_alpha_outline = Image.new("RGBA", image.size)

        # Create a drawing context for the image with an alpha channel
        draw = ImageDraw.Draw(image_with_alpha)
        draw_outlines = ImageDraw.Draw(image_with_alpha_outline)
            
        if len(result) > 0:

            masks = result.masks.xy  # Masks object for segmentation masks outputs
            classes = result.boxes.cls  # Masks object for segmentation masks outputs
            for i in range(len(masks)):
                mask = masks[i]
                classe_index = int(classes[i].item())
                classe_name = result.names[classe_index]

                fill_color = classname_to_color_mapper(classe_name)
                outline_color = classname_to_color_mapper(classe_name,255)

                polygon_point_list = []
                for point in mask:
                    x,y = point
                    polygon_point_list.append((x,y))
                    
                if len(polygon_point_list) > 2:
                    draw.polygon(polygon_point_list, fill_color, outline_color)
                    draw_outlines.polygon(polygon_point_list, None, outline_color,width=3)
        if mask_only:
            annotations_only = np.array(Image.alpha_composite(image_with_alpha, image_with_alpha_outline)) 
        else:
            annotated_face = Image.alpha_composite(image, image_with_alpha)
            annotated_face = Image.alpha_composite(annotated_face, image_with_alpha_outline)
            annotations_only = np.array(annotated_face) 
        annotations_only[..., :3] = annotations_only[..., 2::-1]
        return annotations_only

def try_find_file(file_path:str, default_directory:str):
    """
    returns int(0) when file is not found on neither
    returns int(1) when file is found on the file_path argument
    returns the file path when file is found on the path 
            obtained by concatening both arguments
    returns False when arguments are invalid
    """

    if type(default_directory) != str:
        raise ValueError(f"arg default_directory:{default_directory} not string!")
        return False

    if type(file_path) != str:
        raise ValueError(f"arg file_path:{file_path} not string!")
        return 0

    if os.path.isfile(file_path):
        return 1

    new_path = os.path.join(default_directory,file_path)
    if os.path.isfile(new_path):
        return new_path
        
    return False