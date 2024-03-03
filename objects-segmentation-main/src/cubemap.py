import os
import sys
import time
import math
import zipfile
import imageio
import numpy as np
from PIL import Image
from numpy import clip
import matplotlib.pyplot as plt 
from math import pi,sin,cos,tan,atan2,hypot,floor
from xprojector.objects import GnomonicProjector
import cv2
import pdb
import tqdm

def return_files(path_input):

    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(path_input):
        # check if current path is a file
        if os.path.isfile(os.path.join(path_input, path)):
            res.append(path)
    return res

# ========== Xprojector transform to cubemap
import multiprocessing as mp
from pathlib import Path
def x_generate_cubmaps(path_input, path_output, dims, n_jobs=40):


    # list to store files
    #res = return_files(path_input)

    res = [str(i) for i in Path(path_input).glob('**/*.JPG')]
    res = ignore_already_computed(res, path_output)

    args = []
    for file in res:
        args.append((file, path_input, path_output, dims))
    #print(len(args))
    #print(mp.cpu_count())

    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm.tqdm(pool.imap_unordered(x_generate_cubmap, args), total=len(args)):
        pass

def ignore_already_computed(input_paths, path_output, ext = 'JPG'):
    output_files = os.listdir(path_output)

    input_files = ['_'.join([x.replace('\\', '/').split('/')[-2], x.replace('\\', '/').split('/')[-1]]) for x in input_paths]
    input_files = [x.split('.')[0] for x in input_files]
    output_files = ['_'.join([x.split('_')[1], x.split('_')[2]]) for x in output_files]


    #input_paths_reduced = []
    #for input_path, input_file in zip(input_paths, input_files):
    #    if input_file in output_files:
    #        continue
    #    else:
    #        input_paths_reduced.append(input_path)

    input_paths_reduced = list(set(input_files).difference(output_files))
    input_path = '/'.join(os.path.dirname(input_paths[0]).split('\\')[:-1])
    input_paths_reduced = [x.replace('_', '/') for x in input_paths_reduced]
    input_paths_reduced = [os.path.join(input_path, '{}.{}'.format(x,ext)) for x in input_paths_reduced]

    print('total number of files: {}'.format(len(input_files)))
    print('total of images processed: {}'.format(len(set(output_files))))
    print('total remaining images: {}'.format(len(input_paths_reduced)))

    return input_paths_reduced

def get_filename_cubemaps(filename, platform, keyword='cubemap'):
    face_ids = ['negx', 'negy', 'negz', 'posx', 'posy', 'posz']
    filenames_cubemap = []
    for face_id in face_ids:
        filenames_cubemap.append('_'.join([keyword, platform, filename, face_id]) + '.png')
    return filenames_cubemap

def x_generate_cubmap(args):

    file, path_input, path_output, dims = args

    file = file.replace('\\', '/')
    
    filename = '/'.join(file.split('/')[-2:])
    
    plataforma = file.split('/')[-2]
    
    # print('processing image: ', filename)
    imgIn = Image.open(os.path.join(path_input, filename))
    #print(imgIn.size)
    x_convertBack(imgIn,path_output, file.split('/')[-1].split('.')[0], dims=dims,plat=plataforma)
    

def x_convertBack(imgIn, path_output, filename, dims,plat):
    
    
    proj = GnomonicProjector(dims=dims)
    # print(proj.scanner_shadow_angle)
    angles = {'negx':(0,0),'posz':(np.pi/2,0),'posx':(np.pi,0),
                  'negz':(-np.pi/2,0),'posy':(0,np.pi/2),'negy':(0,-np.pi/2)}

    # angles = [[0,0], [0,np.pi/2], [0,np.pi], [0,-np.pi/2], [np.pi,0], [-np.pi,0]]
    for key, value in angles.items():
        # print(proj.scanner_shadow_angle)
        o_img = proj.forward(np.array(imgIn), value[1], value[0],fov=(1,1)) #=====> ADJUSTED to value[1],value[0] from value[0],value[1]
        o_img = cv2.cvtColor(o_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(path_output + "cubemap_{}_".format(plat)+filename+"_"+key+".png", o_img)

# ========== Custom transform to cubemap

def generate_cubmaps(path_input, path_output):
    if not os.path.exists(path_output):
        os.makedirs(path_output) 

    # list to store files
    res = return_files(path_input)

    for i in range (0, len(res)):
        print('image: ', res[i])
        generate_cubmap(res[i], path_input, path_output)

def generate_cubmap(filename, path_input, path_output):
    imgIn = Image.open(path_input + filename)
    print(imgIn.size)
    inSize = imgIn.size
    imgOut = Image.new("RGB",(inSize[0], int(inSize[0]*3/4)),"black")
    convertBack(imgIn,imgOut)
    print(imgOut.size)
    imgOut.save(path_output + filename)

# ========== ...Custom transform to cubemap

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
    print("Starting...")
    inSize = imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    edge = inSize[0]/4   # the length of each edge in pixels
    for i in range(outSize[0]):
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
import pdb
def split_cubmaps(infile, outfile, keyword='cubemap'):
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    '''
    if len(sys.argv) < 2:
        print("Usage: cubemap-cut.py <filename.jpg|png>")
        sys.exit(-1)
    '''
    #infile = root_path + '000009_cub.png'
    filename, original_extension = os.path.splitext(infile)
    file_extension = ".png"

    name_map = [ \
         ["", "", "posy", ""],
         ["negz", "negx", "posz", "posx"],
         ["", "", "negy", ""]]

    im_name = outfile.split('/')[-1]
    output_folder_path = outfile.split('/')[:-1]
    output_folder_path = '/'.join(output_folder_path) 
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
                    im.crop((sx, sy, sx + cube_size, sy + cube_size)).save(
                        output_folder_path + '/' + keyword + '_' + im_name + '_' + fn) 
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


def join_images_from_name(path_segmentation):
    face_filenames = []
    face_ids = ['negx.png', 'negy.png', 'negz.png', 'posx.png', 'posy.png', 'posz.png']
    vec_img = []

    # Iterate over 6 faces
    for face_id in face_ids:
        face_filename = path_segmentation + '_' + face_id
        vec_img.append(np.array(Image.open(face_filename)))

    # print(face_filenames)


    # name_map = [ \
    #      ["", "posy", "", ""],
    #      ["negz", "negx", "posz", "posx"],
    #      ["", "negy", "", ""]]
    
    width, height, channels = vec_img[0].shape
    # print(width, height, channels)

    cube_size = np.zeros((width * 3 , width * 4, channels), dtype  = np.uint8)

    # cube_size[0:width, width:width*2] = vec_img[4]
    cube_size[width:width*2, 0:width] = vec_img[0]
    cube_size[width:width*2, width:width*2] = vec_img[5]
    cube_size[width:width*2, width*2:width*3] = vec_img[3]
    cube_size[width:width*2, width*3:width*4] = vec_img[2]
    # cube_size[2*width:width*3, width:width*2] = vec_img[1]

    cube_size[0:width, width:width*2] = np.rot90(vec_img[1], axes=(1,0))
    cube_size[2*width:width*3, width:width*2] = np.rot90(vec_img[4], axes=(0,1))

    return cube_size

def split_images_from_name(path):
    cubemap = np.array(Image.open(path))
    face_ids = ['negx.png', 'negy.png', 'negz.png', 'posx.png', 'posy.png', 'posz.png']
    cubemap_height, cubemap_width, channels = cubemap.shape
    height = cubemap_height//3
    width = height
    print(cubemap.shape)
    print(height, width)
    vec_img = []
    vec_img.append(cubemap[width:width*2, 0:width])
    vec_img.append(np.rot90(cubemap[0:width, width:width*2], axes=(0,1)))
    vec_img.append(cubemap[width:width*2, width*3:width*4])
    vec_img.append(cubemap[width:width*2, width*2:width*3])
    vec_img.append(np.rot90(cubemap[2*width:width*3, width:width*2], axes=(1,0)))
    vec_img.append(cubemap[width:width*2, width:width*2])
    
    for img, face_id in zip(vec_img, face_ids):
        imageio.imwrite(
            face_id, img)
    

def cubemap_to_2D(args):
    path_input, cubemap_keyword, filename_360, path_output_2D = args
    cube_prediction = join_images_from_name(
        os.path.join(path_input,
            '_'.join([cubemap_keyword, filename_360]))
    )
    imageio.imwrite(
        os.path.join(path_output_2D,
            '.'.join([filename_360, 'png'])), cube_prediction)    

def cubemaps_to_2D(path_input, cubemap_keyword, filenames_360, path_output_2D):

    args=[]
    for filename_360 in filenames_360:
        args.append((path_input, cubemap_keyword, filename_360, path_output_2D))

    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm.tqdm(pool.imap_unordered(cubemap_to_2D, args), total=len(args)):
        pass

def get_unique_from_cubemaps(filenames_360):
    return ['_'.join(i.split('_')[:-1]) for i in filenames_360]
def get_unique_from_cubemaps2(filenames_360):
    return ['_'.join(i.split('_')[1:3]) for i in filenames_360]

def ignore_incomplete_cubemaps(filenames_360, path_input, faces, path_csv, keyword='cubemap'):
    filenames_cubemap_complete = []
    for filename_360 in filenames_360:
        faces_exist = []
        for face in faces:
            filename_cubemap = os.path.join(path_input, '{}_{}_{}'.format(keyword,filename_360,face))
            if os.path.exists(filename_cubemap):
                faces_exist.append(True) 
            else:
                faces_exist.append(False)
                with open(path_csv, "a") as f:
                    f.write("\n" + os.path.basename(filename_cubemap))

        if all([x == True for x in faces_exist]):
            filenames_cubemap_complete.append(filename_360)
    return filenames_cubemap_complete

def check_complete_cubemap(filename_360, path_input, faces, path_csv, keyword='cubemap'):

    faces_exist = []
    filename_360
    for face in faces:
        filename_cubemap = os.path.join(path_input, '{}_{}_{}'.format(keyword,filename_360,face))
        if os.path.exists(filename_cubemap):
            faces_exist.append(True) 
        else:
            faces_exist.append(False)
            with open(path_csv, "a") as f:
                f.write("\n" + os.path.basename(filename_cubemap))

    if all([x == True for x in faces_exist]):
        return True
    else:
        return False
'''
def check_complete_cubemap(filename_360, path_input, faces, path_csv, keyword='cubemap'):

    for face in faces:
        filename_cubemap = os.path.join(path_input, '{}_{}_{}'.format(keyword,filename_360,face))
        if os.path.exists(filename_cubemap):
            continue 
        else:
            with open(path_csv, "a") as f:
                f.write("\n" + os.path.basename(filename_cubemap))
            return False
    return True
'''
def ignore_already_processed_cubemaps(filenames_360, path_output_360):

    output_filenames = ['_'.join(i.split('.')[0].split('_')[0:2]) for i in os.listdir(path_output_360)]
    filenames_360 = list(set(filenames_360).difference(set([i.split('.')[0] for i in output_filenames])))
    return filenames_360
    
def cubemaps_to_360(path_input, cubemap_keyword, filenames_360, path_output_360, path_csv, n_jobs=1):


    # ignore incomplete cubemaps
    faces = ['negx.png', 'negy.png', 'negz.png', 'posx.png', 'posy.png', 'posz.png']
    path_csv = os.path.join(os.path.dirname(path_csv),"unsuccessful_from_cubemap.txt")

    # filenames_360 = ignore_incomplete_cubemaps(filenames_360, path_input, faces, path_csv)
    # print('number of valid files: {}'.format(len(filenames_360)))

    args = []
    for filename_360 in filenames_360:
        args.append((path_input, cubemap_keyword, filename_360, path_output_360, faces, path_csv))

    print(len(args))
    print(mp.cpu_count())

    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm.tqdm(pool.imap_unordered(cubemap_to_360, args), total=len(args)):
        pass
            
def cubemap_to_360(args):
    try:
        path_input, cubemap_keyword, filename_360, path_output_360, faces, path_csv = args
        if check_complete_cubemap(filename_360, path_input, faces, path_csv):
            pass
        else:
            pass

        path_segmentation = os.path.join(path_input, "{}_{}".format(cubemap_keyword, filename_360))

        face_ids = ['negx.png', 'posy.png', 'posx.png', 'negy.png', 'posz.png', 'negz.png']

        vec_img = []
        for face_id in face_ids:
            face_filename = "{}_{}".format(path_segmentation, face_id)
            try:
                img = np.array(Image.open(face_filename))
                if face_id == 'posx.png':
                    img = np.rot90(img, k=2)
            except:
                img = np.zeros((1024,1024,4),dtype=np.uint8)
            vec_img.append(img) # 

        H,W,c = vec_img[0].shape

        proj = GnomonicProjector(dims=(H,W))

        angles = {'negx':(0,0),'posy':(np.pi/2,0),'posx':(np.pi,0),
                    'negy':(-np.pi/2,0),'posz':(0,np.pi/2),'negz':(0,-np.pi/2)}  

        dims_360 = (1024*2, 1024*4)
        im = np.zeros((dims_360[0], dims_360[1], 4))
        for idx, (key, values) in enumerate(angles.items()):

            im += proj.backward(vec_img[idx], values[0], values[1], dims_360)

        #im[im>128] = 255


        
        im[..., :3] = im[..., 2::-1]
        
        #print(path_output_360)
        #print(filename_360)
        cv2.imwrite(os.path.join(path_output_360, filename_360 + ".png"), np.squeeze(im))
        # pdb.set_trace()
        #print(im.shape)
        # print('Processed file: {}'.format(filename_360 + ".png"))
    except Exception as e:
        print('error: {}'.format(e))

    
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

    wo, ho = inimg.size
    print('image size: ', wo, ho)

    # Calculate height and width of output image, and size of each square face
    #h = int(wo/3)
    #w = int(2*h)
    h = 2688
    w = 5376
    n = ho/3
    print(w,h)


    # Create new image with width w, and height h
    outimg = Image.new('RGB', (w, h))

    # For each pixel in output image find colour value from input image
    for ycoord in range(0, h):
        for xcoord in range(0, w):
            corrx, corry = find_corresponding_pixel(xcoord, ycoord, w, h, n)

            outimg.putpixel((xcoord, ycoord), inimg.getpixel((corrx, corry)))
        # Print progress percentage
        #print(str(round((float(ycoord)/float(h))*100, 2)) + '%')


    outimg.save(outfile, 'PNG')