#%%

from PIL import Image
import numpy as np
from scipy import ndimage
import math
import numpy as np  
import pye57
from xprojector.ico import icosphere,icoangles
from xprojector.utils import *
import pdb
Image.MAX_IMAGE_PIXELS = 933120000


class GnomonicProjector:
    def __init__(self,dims,scanner_shadow_angle=0):
        self.f_projection=None
        self.b_projection=None
        self.dims=dims
        self.scanner_shadow_angle=scanner_shadow_angle
        pass
    
    def point_forward(self,x,y,phi1,lamb0,fov):
        rho=np.sqrt(x**2+y**2)
        c=np.arctan2(rho,1)
        sinc=np.sin(c)
        cosc=np.cos(c)

        phi=np.arcsin(cosc*np.sin(phi1)+(y*sinc*np.cos(phi1)/rho))
        lamb=lamb0+np.arctan2(x*sinc,rho*np.cos(phi1)*cosc-y*np.sin(phi1)*sinc)
        
        phi=np.where(phi<-np.pi/2,np.pi/2-phi,phi)
        lamb=np.where(lamb<-np.pi,2*np.pi+lamb,lamb)

        phi=np.where(phi>np.pi/2,-np.pi/2+phi,phi)
        lamb=np.where(lamb>np.pi,-2*np.pi+lamb,lamb)
        
        return phi,lamb
    

    def forward(self,img,phi1,lamb0,fov=1):
        fov_h,fov_w = fov
        
        H,W=self.dims
        x,y=np.meshgrid(np.linspace(-1,1,W)*fov_w,np.linspace(-1,1,H)*fov_h)
        phi,lamb=self.point_forward(x,y,phi1,lamb0,fov)
        
        mask = (phi>np.pi/3)&(phi<np.pi/2)
        phi=phi/(np.pi/2)
        lamb=lamb/np.pi

        HH,WW,C=img.shape
        phi=(0.5*(phi+1))*HH*((180/(180-self.scanner_shadow_angle)))
        lamb=(0.5*(lamb+1))*WW
        

        o_img=[ndimage.map_coordinates(img[:,:,i], np.stack([phi,lamb]),order=0,prefilter=False,mode="nearest") for i in range(C)]
        o_img=np.stack(o_img,axis=-1)
        #o_img[mask]=0
        return o_img

        self.f_projection=o_img
        self.phi1=phi1
        self.lamb0=lamb0
        self.fov=fov
        pass
    
    def point_backward(self,phi,lamb,phi1,lamb0,fov):
        fov_h,fov_w = fov
        cosc=np.sin(phi1)*np.sin(phi)+np.cos(phi1)*np.cos(phi)*np.cos(lamb-lamb0)

        K=1/cosc
        x=K*np.cos(phi)*np.sin(lamb-lamb0)/fov_w
        y=K*(np.cos(phi1)*np.sin(phi)-np.sin(phi1)*np.cos(phi)*np.cos(lamb-lamb0))/fov_h

        x=0.5*(x+1)
        y=0.5*(y+1)

        HH,WW=self.dims
        x=x*HH
        y=y*WW
        self.cosc=cosc
        return x,y
    
    def backward(self, img, phi1, lamb0, dims_360, fov=(1,1)):
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        H,W = dims_360
        # phi1=self.phi1
        # lamb0=self.lamb0
        # fov_h,fov_w=self.fov
        fov_h,fov_w=fov
        
        u,v=np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
        phi=v*(np.pi/2)
        lamb=u*np.pi

        x,y=self.point_backward(phi,lamb,phi1,lamb0,fov)

        oo=[ndimage.map_coordinates(img[:,:,i].T, np.stack([x,y]),order=5,prefilter=False)*(self.cosc>0) for i in range(img.shape[-1])]
        oo=np.stack(oo,axis=-1)
        
        self.b_projection=oo
        return self.b_projection

    def generator(self,img,fov,npoints, mode,delta_lamb,delta_phi,add_noise=False):
        def sample_sphere(npoints=100):
            if mode=='cubemap':
                points,angles = cubemap_angles(delta_lamb,delta_phi)
            elif mode == 'fibb':
                points,angles = fibonacci_sphere(samples=npoints)
            elif mode =='ico':
                points,angles = icoangles(npoints)
            else:
                raise

            for lamb0,phi1 in angles:
                if add_noise:
                    lamb0_noise = (np.pi/180)*np.random.uniform(low=0.0, high=25.0, size=1).item()   #np.random.randn(1)*(10*(np.pi/180))
                    phi1_noise = (np.pi/180)*np.random.uniform(low=0.0, high=20.0, size=1).item()   #np.random.randn(1)*(10*(np.pi/180))
                    lamb0+=lamb0_noise
                    phi1+=phi1_noise
                    
                self.forward(img,phi1,lamb0,fov=fov)

                yield {'img':self.f_projection,'lamb0':lamb0,'phi1':phi1}
        return sample_sphere(npoints)


class GnomonicProjector2_deprecated:
    def __init__(self,dims,scanner_shadow_angle=30):
        self.f_projection=None
        self.b_projection=None
        self.dims=dims
        self.scanner_shadow_angle=scanner_shadow_angle
        pass
    
    def point_forward(self,x,y,phi1,lamb0,fov):
        rho=np.sqrt(x**2+y**2)
        c=np.arctan2(rho,1)
        sinc=np.sin(c)
        cosc=np.cos(c)

        phi=np.arcsin(cosc*np.sin(phi1)+(y*sinc*np.cos(phi1)/rho))
        lamb=lamb0+np.arctan2(x*sinc,rho*np.cos(phi1)*cosc-y*np.sin(phi1)*sinc)
        
        phi=np.where(phi<-np.pi/2,np.pi/2-phi,phi)
        lamb=np.where(lamb<-np.pi,2*np.pi+lamb,lamb)

        phi=np.where(phi>np.pi/2,-np.pi/2+phi,phi)
        lamb=np.where(lamb>np.pi,-2*np.pi+lamb,lamb)
        
        return phi,lamb
    

    def forward(self,img,depthmap,phi1,lamb0,fov=1):
        fov_h,fov_w = fov
        
        H,W=self.dims
        x,y=np.meshgrid(np.linspace(-1,1,W)*fov_w,np.linspace(-1,1,H)*fov_h)
        phi,lamb=self.point_forward(x,y,phi1,lamb0,fov)
        
        mask = (phi>np.pi/3)&(phi<np.pi/2)
        phi=phi/(np.pi/2)
        lamb=lamb/np.pi

        HH,WW,_=img.shape
        phi=(0.5*(phi+1))*HH*((180/(180-self.scanner_shadow_angle)))
        lamb=(0.5*(lamb+1))*WW
        

        o_img=[ndimage.map_coordinates(img[:,:,i], np.stack([phi,lamb]),prefilter=False,mode='nearest') for i in range(3)]
        o_img=np.stack(o_img,axis=-1)
        o_img[mask]=0

        
        o_dmap=ndimage.map_coordinates(depthmap, np.stack([phi,lamb]),prefilter=False,mode='nearest')#,cval=1.71) 
        o_dmap[mask]=0

        self.f_projection=(o_img,o_dmap)
        self.phi1=phi1
        self.lamb0=lamb0
        self.fov=fov
        pass
    
    def point_backward(self,phi,lamb,phi1,lamb0,fov):
        fov_h,fov_w = fov
        cosc=np.sin(phi1)*np.sin(phi)+np.cos(phi1)*np.cos(phi)*np.cos(lamb-lamb0)

        K=1/cosc
        x=K*np.cos(phi)*np.sin(lamb-lamb0)/fov_w
        y=K*(np.cos(phi1)*np.sin(phi)-np.sin(phi1)*np.cos(phi)*np.cos(lamb-lamb0))/fov_h

        x=0.5*(x+1)
        y=0.5*(y+1)

        HH = self.f_projection.shape[0]
        WW = self.f_projection.shape[1]
        x=x*HH
        y=y*WW
        self.cosc=cosc
        return x,y
    
    def backward(self,img,f_projection=None,phi1=None,lamb0=None,fov=None):
        H,W,_=img.shape
        phi1=phi1 if isinstance(phi1,float) else self.phi1
        lamb0=lamb0  if isinstance(lamb0,float) else self.lamb0
        fov_h,fov_w= fov if isinstance(fov,tuple) else self.fov
        
        u,v=np.meshgrid(np.linspace(-1,1,W),np.linspace(-np.pi/2,np.pi/2-(self.scanner_shadow_angle/180)*np.pi,H))
        #phi=v*(np.pi/2)
        #phi=(0.5*(phi+1))*HH*()
        lamb=u*np.pi
        
        if isinstance(f_projection,np.ndarray):
            self.f_projection = f_projection
        
        x,y=self.point_backward(phi,lamb,phi1,lamb0,self.fov)

        oo=ndimage.map_coordinates(self.f_projection, np.stack([x,y]))*(self.cosc>0)
        #oo=np.stack(oo,axis=-1)
        self.b_projection=oo
        pass

    def generator(self,img,depthmap,fov,npoints, mode,delta_lamb,delta_phi,add_noise=False):
        def sample_sphere(npoints=100):
            if mode=='cubemap':
                points,angles = cubemap_angles(delta_lamb,delta_phi)
            elif mode == 'fibb':
                points,angles = fibonacci_sphere(samples=npoints)
            else:
                raise

            for lamb0,phi1 in angles:
                if add_noise:
                    lamb0_noise = (np.pi/180)*np.random.uniform(low=0.0, high=25.0, size=1).item()   #np.random.randn(1)*(10*(np.pi/180))
                    phi1_noise = (np.pi/180)*np.random.uniform(low=0.0, high=20.0, size=1).item()   #np.random.randn(1)*(10*(np.pi/180))
                    lamb0+=lamb0_noise
                    phi1+=phi1_noise
                    
                self.forward(img,depthmap,phi1,lamb0,fov=fov)

                yield {'img':self.f_projection[0],'depthmap':self.f_projection[1],'lamb0':lamb0,'phi1':phi1}
        return sample_sphere(npoints)

class GnomonicProjector2:
    def __init__(self,dims,scanner_shadow_angle=30):
        self.f_projection=None
        self.b_projection=None
        self.dims=dims
        self.scanner_shadow_angle=scanner_shadow_angle
        pass
    
    def point_forward(self,x,y,phi1,lamb0,fov):
        rho=np.sqrt(x**2+y**2)
        c=np.arctan2(rho,1)
        sinc=np.sin(c)
        cosc=np.cos(c)

        phi=np.arcsin(cosc*np.sin(phi1)+(y*sinc*np.cos(phi1)/rho))
        lamb=lamb0+np.arctan2(x*sinc,rho*np.cos(phi1)*cosc-y*np.sin(phi1)*sinc)
        
        phi=np.where(phi<-np.pi/2,np.pi/2-phi,phi)
        lamb=np.where(lamb<-np.pi,2*np.pi+lamb,lamb)

        phi=np.where(phi>np.pi/2,-np.pi/2+phi,phi)
        lamb=np.where(lamb>np.pi,-2*np.pi+lamb,lamb)
        
        return phi,lamb
    

    def forward(self,img,depthmap,phi1,lamb0,fov=1,order=0,prefilter=False,mode='nearest'):
        fov_h,fov_w = fov
        
        H,W=self.dims
        x,y=np.meshgrid(np.linspace(-1,1,W)*fov_w,np.linspace(-1,1,H)*fov_h)
        phi,lamb=self.point_forward(x,y,phi1,lamb0,fov)
        
        mask = (phi>np.pi/3)&(phi<np.pi/2)
        phi=phi/(np.pi/2)
        lamb=lamb/np.pi

        HH,WW,_=img.shape
        phi=(0.5*(phi+1))*HH*((180/(180-self.scanner_shadow_angle)))
        lamb=(0.5*(lamb+1))*WW
        

        o_img=[ndimage.map_coordinates(img[:,:,i], np.stack([phi,lamb]),prefilter=False,mode='nearest') for i in range(3)]#
        o_img=np.stack(o_img,axis=-1)
        o_img[mask]=0

        
        o_dmap=ndimage.map_coordinates(depthmap, np.stack([phi,lamb]),\
                                       order=order,prefilter=prefilter,mode=mode)#,cval=1.71) 
        o_dmap[mask]=0

        self.f_projection=(o_img,o_dmap)
        self.phi1=phi1
        self.lamb0=lamb0
        self.fov=fov
        pass
    
    def point_backward(self,phi,lamb,phi1,lamb0,fov):
        fov_h,fov_w = fov
        cosc=np.sin(phi1)*np.sin(phi)+np.cos(phi1)*np.cos(phi)*np.cos(lamb-lamb0)

        K=1/cosc
        x=K*np.cos(phi)*np.sin(lamb-lamb0)/fov_w
        y=K*(np.cos(phi1)*np.sin(phi)-np.sin(phi1)*np.cos(phi)*np.cos(lamb-lamb0))/fov_h

        x=0.5*(x+1)
        y=0.5*(y+1)

        HH = self.f_projection.shape[0]
        WW = self.f_projection.shape[1]
        x=x*WW
        y=y*HH
        self.cosc=cosc
        return x,y
    
    def backward(self,img,f_projection=None,phi1=None,lamb0=None,fov=None):
        H,W,_=img.shape
        phi1=phi1 if isinstance(phi1,float) else self.phi1
        lamb0=lamb0  if isinstance(lamb0,float) else self.lamb0
        fov_h,fov_w= fov if isinstance(fov,tuple) else self.fov
        
        
        u,v=np.meshgrid(np.linspace(-1,1,W),np.linspace(-np.pi/2,np.pi*(90-self.scanner_shadow_angle)/180,H))

        phi=v#*(np.pi/2)
        lamb=u*np.pi
        
        if isinstance(f_projection,np.ndarray):
            self.f_projection = f_projection
        
        x,y=self.point_backward(phi,lamb,phi1,lamb0,self.fov)
        #y/=((180/(180-self.scanner_shadow_angle)))

        oo=ndimage.map_coordinates(self.f_projection.T, np.stack([x,y]))*(self.cosc>0)
        #oo=np.stack(oo,axis=-1)
        self.b_projection=oo
        pass

    def generator(self,img,depthmap,fov,npoints, mode,delta_lamb,delta_phi,add_noise=False):
        def sample_sphere(npoints=100):
            if mode=='cubemap':
                points,angles = cubemap_angles(delta_lamb,delta_phi)
            elif mode == 'fibb':
                points,angles = fibonacci_sphere(samples=npoints)
            else:
                raise

            for lamb0,phi1 in angles:
                if add_noise:
                    lamb0_noise = (np.pi/180)*np.random.uniform(low=0.0, high=25.0, size=1).item()   #np.random.randn(1)*(10*(np.pi/180))
                    phi1_noise = (np.pi/180)*np.random.uniform(low=0.0, high=20.0, size=1).item()   #np.random.randn(1)*(10*(np.pi/180))
                    lamb0+=lamb0_noise
                    phi1+=phi1_noise
                    
                self.forward(img,depthmap,phi1,lamb0,fov=fov)

                yield {'img':self.f_projection[0],'depthmap':self.f_projection[1],'lamb0':lamb0,'phi1':phi1}
        return sample_sphere(npoints)


def img_to_scan_img(img,scan_H,scan_W,scan_shadow_angle=60):
    H,W,_=img.shape
    scan_H+=1
    scan_W+=1
    grid = np.meshgrid(np.linspace(0,1,scan_W),np.linspace(0,1,scan_H))
    grid[1]*=(H*(180-scan_shadow_angle)/180)
    grid[0]*=W
    r=[ndimage.map_coordinates(img[:,:,i].T,np.stack(grid,axis=0)) for i in range(3)]
    r=np.stack(r,axis=-1)

    return r

def to_img(rowIndex,columnIndex,a_t):

    H = rowIndex.max()
    W = columnIndex.max()

    if len(a_t.shape)>1:
        img = np.zeros((H+1,W+1))
        img[rowIndex,columnIndex,:] = a_t
        
    else:
        img = np.zeros((H+1,W+1))
        img[rowIndex,columnIndex] = a_t

    return img
    
class sphere3D:
    def __init__(self,e57path):
        self.e57file = e57path
        self.imgpath = '{}.png'.format("".join(e57path.split('.')[0]))
        self.load()
        self.tosphere()


        
    def load(self):
        
        e57 = pye57.E57(self.e57file)
        header = e57.get_header(0)
        self.position_scan_0 = e57.scan_position(0)
        point_count=header.point_count
        self.Rot=header.rotation_matrix
        self.Rot_inv=np.linalg.inv(self.Rot)
        self.T=header.translation[:,np.newaxis]

        data = e57.read_scan(0,colors=True, row_column=True)    
        self.img = np.array(Image.open(self.imgpath))
        self.oH,self.oW,_ = self.img.shape
        
        self.scan_H = data['rowIndex'].max()
        self.scan_W = data['columnIndex'].max()
        self.xyz_t = np.stack([data['cartesianX'],data['cartesianY'],data['cartesianZ']]).T
        self.rgb_t = np.stack([data['colorRed']/255.,data['colorGreen']/255.,data['colorBlue']/255.]).T

        self.rowIndex = data['rowIndex']
        self.columnIndex = data['columnIndex']
        
        del data
        del e57
    def tosphere(self):
        centered = self.xyz_t - self.position_scan_0
        R = np.sqrt(np.sum(centered**2,axis=1))
        normalized = centered/R[:,None]
        normalized = normalized# + position_scan_0

        # https://en.wikipedia.org/wiki/Spherical_coordinate_system
        phi = np.arctan2(normalized[:,0],normalized[:,1])
        theta = np.arctan2( normalized[:,2],np.sqrt(normalized[:,1]**2 + normalized[:,0]**2) )
        scanner_shadow_angle = 90 - np.abs(theta.min())*(180/np.pi) #===> Use 30 as datasheet but store computed

        self.scan_img = img_to_scan_img(self.img,self.scan_H,self.scan_W,scan_shadow_angle=30)#==== HACK #scanner_shadow_angle)
        self.depthmap = to_img(self.rowIndex,self.columnIndex,R)
        self.proj = GnomonicProjector2(dims=(self.scan_H//2,self.scan_W//4),scanner_shadow_angle=30)#==== HACK #scanner_shadow_angle)
        self.scanner_shadow_angle =scanner_shadow_angle 

    

    def generator(self,fov,npoints,mode,delta_lamb,delta_phi,add_noise):
        
        return self.proj.generator(self.scan_img,self.depthmap,mode=mode,delta_lamb=delta_lamb,delta_phi=delta_phi,fov=fov,npoints=npoints,add_noise=add_noise)

    def forward(self,phi,lamb,fov=(1,1)):
        self.proj.forward(self.scan_img,self.depthmap,phi,lamb,fov=fov)
        return self.proj.f_projection

    def to_dict(self):
        xc,yc,zc = self.T
        d1= {'R_{}'.format(k):v for k,v in enumerate(self.Rot.flatten())}
        d1.update({'H':self.scan_H,'W':self.scan_W,'xc':xc.item(),\
            'yc':yc.item(),'zc':zc.item(),'oH':self.oH,'oW':self.oW,'e57':str(self.e57file),'scanner_shadow_angle':self.scanner_shadow_angle})
        return d1


    def to_img(self,a_t):
        img = to_img(self.rowIndex,self.columnIndex,a_t)

        return img