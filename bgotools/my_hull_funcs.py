#!/usr/bin/env python

'''
Find the convex hull of a set of data points.
Andrew's monotone chain algorithm, modified for own use. 
by ds-wen
'''
import numpy as np 
from operator import itemgetter
from scipy.spatial import ConvexHull, Delaunay
import scipy.interpolate as interp
from pymatgen.util.coord import Simplex, in_coord_list
from functools import lru_cache
from sympy import Symbol
from sympy import Matrix,Array
from sympy import MutableDenseNDimArray as MArray

class my_hull_funcs:

    # convex hull and concave hull with alpha

    def __init__(self,X,Y,Z=None,nu_Y=0,nu_Z=0):

        self.X = X #1darray
        self.Y = Y #1d
        if Z is None:
            self.Z = Z
        else:
            Z = Z.flatten()
            self.Z = Z #1d
        
        self.nu_Y = nu_Y #uncertainty on Y
        self.nu_Z = nu_Z #uncertainty on Z # maybe useful for future 
        # uncertainty can be used in EI acquisition functions
        self.tol = 1e-14 # for composition tolerance 
    def get_bottom_hull(self,use_bary=False):
        self.use_bary = use_bary
        # return the indices of the bottom-hull configurations 
        if self.Z is None:
            # 2d 
            # convert X-array, Y-array data to tuples [(x1,y1),(x2,y2),...] and sort them based on X
            # keep track of the index after sorting
            # X is comp(a)
            # Y is energy

            if np.shape(self.nu_Y)==(len(self.Y),len(self.Y)):

                # use full covariance matrix
                self.nu_Y_matrix = np.array(self.nu_Y)
            elif np.shape(self.nu_Y)==(len(self.Y),1):
                # use diagonal
                self.nu_Y_matrix = np.diag(np.array(self.nu_Y).flatten())
            else:
                self.nu_Y_matrix = np.diag(np.zeros(len(self.Y)))

            self.points = np.array(list(map(lambda x, y:[x,y], self.X,self.Y)))
            

            self.hull = ConvexHull(self.points)
            # index end points
            p0 = np.where(self.points[self.hull.vertices,0]==0)[0][0]
            p1 = np.where(self.points[self.hull.vertices,0]==1)[0][0]
            # define the plane 0, any vertices below plane 0 will be the bottom convex hull
            points_0 = self.points[self.hull.vertices[[p0,p1]]].T  
            x0 = points_0[0]
            y0 = points_0[1]
            plane_0_func = interp.interp1d(x0,y0)
            xs = self.points[self.hull.vertices].T[0]
            ys = self.points[self.hull.vertices].T[1]
            # bt_hull_facets is a function that depends on the value of x 
            # use this to evaluate the configuration distance to hull
            self.bt_hull_vertices = self.hull.vertices[plane_0_func(xs) >= ys]
            self.bt_hull_points = self.points[self.hull.vertices[plane_0_func(xs) >= ys]]
            bt_x = self.bt_hull_points.T[0]
            bt_y = self.bt_hull_points.T[1]
            #print(bt_x,bt_y)
            self.bt_hull_facets = interp.interp1d(bt_x,bt_y)

        else:
            # 3d
            # convert X-array, Z-array data to tuples [(x1,y1),(x2,y2),...] and sort them based on X
            # keep track of the index after sorting
            # X is comp(a)
            # Y is comp(b)
            # Z is energy 
            if np.shape(self.nu_Z)==(len(self.Z),len(self.Z)):
                # use full covariance matrix
                self.nu_Z_matrix = np.array(self.nu_Z)
            elif np.shape(self.nu_Z)==(len(self.Z),1):
                # use diagonal
                self.nu_Z_matrix = np.diag(np.array(self.nu_Z).flatten())
            elif type(self.nu_Z)==float:
                self.nu_Z_matrix = np.diag(np.ones(len(self.Z)).flatten())*self.nu_Z
            else:
                self.nu_Z_matrix = np.diag(np.zeros(len(self.Z)))
                
            self.points_all = np.array(list(map(lambda x, y,z:[x,y,z], self.X,self.Y,self.Z)))
            self.find_plane_zero()
            self.points = self.points_all[self.get_mask()]

            self.calc_points_bary()
            self.hull = ConvexHull(self.points)
            
            # define the plane 0, any vertices below plane 0 will be the bottom convex hull
            
            if self.use_bary:
                self.points = self.points_bary
                p0 = np.where((self.points[:,0]==self.bary_tri[0][0])&(self.points[:,1]==self.bary_tri[0][1]))[0][0]
                p1 = np.where((self.points[:,0]==self.bary_tri[1][0])&(self.points[:,1]==self.bary_tri[1][1]))[0][0]
                p2 = np.where((self.points[:,0]==self.bary_tri[2][0])&(self.points[:,1]==self.bary_tri[2][1]))[0][0]
            else: 
                p0 = np.where((self.points[:,0]==0)&(self.points[:,1]==0))[0][0]
                p1 = np.where((self.points[:,0]==1)&(self.points[:,1]==0))[0][0]
                p2 = np.where((self.points[:,0]==0)&(self.points[:,1]==1))[0][0]
            self.points_0_index = np.sort(np.array([p0,p1,p2]))
            # bt_hull_facets is a function that depends on the x and y
            # use this to evaluate the configuration distance to hull
            # use final_simplices to build bottom convex hull
            self.final_simplices = self.get_simplices()
            self.pmg_simplexes = [Simplex(self.points[f, :-1]) for f in self.final_simplices]
            # self.bt_hull_facets = self.calc_energy_at_comp
            hp_index = np.unique(np.array(self.final_simplices).flatten())
            self.bt_hull_points = self.points[hp_index]
            self.bt_hull_vertices = hp_index
            self.bt_hull_index_all = []
            for i in hp_index:# find the index in self.points_all
                for j in range(len(self.points_all)):
                    if np.linalg.norm(self.points_all[j]-self.points[i])==0:
                        self.bt_hull_index_all.append(j)
                        break
    def find_plane_zero(self):
        p0 = np.where((self.points_all[:,0]==0)&(self.points_all[:,1]==0))[0][0]
        p1 = np.where((self.points_all[:,0]==1)&(self.points_all[:,1]==0))[0][0]
        p2 = np.where((self.points_all[:,0]==0)&(self.points_all[:,1]==1))[0][0]

        self.plane_zero_facets = np.sort(np.array([p0,p1,p2])) # index
        self.plane_zero_simplice = Simplex(np.array(self.points_all[self.plane_zero_facets,:-1],dtype='float64')) # points
        self.plane_zero_e = np.array(self.points_all[self.plane_zero_facets,2],dtype='float64').flatten()
    def calc_plane_zero_comp(self,comp):
        comp_ = self.plane_zero_simplice.bary_coords(comp)
        Ef = np.dot(self.plane_zero_e,comp_)
        return Ef 
    def get_mask(self):
        mask = []
        for p in self.points_all:
            c = p[:2]
            e = p[2]
            et = self.calc_plane_zero_comp(c)
            if e<=et:
                mask.append(True)
            else:
                mask.append(False)
        return mask 
    def shoelace_area(self):
        if self.Z is None:
            # 2d 
            # 
            # X is comp(a)
            # Y is energy

            x = np.array(self.bt_hull_points.T[0])
            y = np.array(self.bt_hull_points.T[1])
            self.nu_y_matrix_hull = np.diag(np.zeros(len(y))) 
            for i in range(len(y)):
                for j in range(len(y)):
                    self.nu_y_matrix_hull[i,j] = self.nu_Y_matrix[self.bt_hull_vertices[i],self.bt_hull_vertices[j]]

            # assume x, y are already sorted according to x
            
            x_i_1 = np.roll(x,1)#np.insert(x[:-1],0,x[-1])
            x_i_2 = np.roll(x,-1) #np.append(x[1:],x[0])
            
            self.Delta_X = x_i_2 - x_i_1
            #print(np.dot(Delta_X,y))
            self.Area = .5*np.abs(np.dot(self.Delta_X,y))
            self.nu_Area = .25*np.dot(np.dot(self.Delta_X,self.nu_y_matrix_hull),self.Delta_X)
            #return self.Area,self.nu_Area

        else:
            # 3d 
            # X is comp(a)
            # Y is comp(b)
            # Z is energy
            # apply shoelace formula to 3d hull volume
            x = np.array(self.bt_hull_points.T[0])
            y = np.array(self.bt_hull_points.T[1])
            z = np.array(self.bt_hull_points.T[2])
            z_sym = np.array([Symbol('E_{}'.format(i)) for i in range(len(z))])
            self.z_sym = z_sym
            self.z_vals = z
            points_w_syms = np.array([x,y,z_sym]).T
            points_w_vals = np.array([x,y,z]).T
            
            # covariance matrix of Z-energy
            self.nu_z_matrix_hull = np.diag(np.zeros(len(z))) 
            for i in range(len(z)):
                for j in range(len(z)):
                    self.nu_z_matrix_hull[i,j] = self.nu_Z_matrix[self.bt_hull_vertices[i],self.bt_hull_vertices[j]]
            
            # get tetrahedrons of the 3d hull from Delauney triangulation
            self.delaunay = Delaunay(self.bt_hull_points)
            
            func = 0
            func_val = 0
            for idx_list in self.delaunay.simplices:
                a_sym = points_w_syms[idx_list[0]]
                b_sym = points_w_syms[idx_list[1]]
                c_sym = points_w_syms[idx_list[2]]
                d_sym = points_w_syms[idx_list[3]]
                a_val = points_w_vals[idx_list[0]]
                b_val = points_w_vals[idx_list[1]]
                c_val = points_w_vals[idx_list[2]]
                d_val = points_w_vals[idx_list[3]]
                volume = self.tetrahedron_volume_3(a_sym,b_sym,c_sym,d_sym)
                volume_val = self.tetrahedron_volume_3(a_val,b_val,c_val,d_val)
                if volume_val>=0:
                    func = func+volume
                    func_val = func_val+volume_val
                else:
                    func = func-volume
                    func_val = func_val-volume_val
            self.volume_func = func
            self.volume_val = np.float64(func_val)
            #print(len(tetrahedrons))
            self.F_vector = np.array([self.volume_func.as_coefficients_dict()[self.z_sym[i]] for i in range(len(self.z_sym))])
            
            self.Area = self.volume_val
            self.var_Area = np.dot(np.dot(self.F_vector,self.nu_z_matrix_hull),self.F_vector)
            self.nu_Area = np.sqrt(np.float64(self.var_Area))
            
            
    def tetrahedron_volume_3(self,a, b, c, d,var=None):
        # a, b, c, d are four vertices of a tetrahedron
        
        a = Array([x for x in a])
        b = Array([x for x in b])
        c = Array([x for x in c])
        d = Array([x for x in d])

        l_ab = b-a
        l_ac = c-a
        l_ad = d-a

        A_M = Matrix([l_ab,l_ac,l_ad]).T

        return A_M.det()/6

    def get_simplices(self):
        # simplices of the Convex Hull. partly from pymatgen
        
        tmp_simplices = []
        for facet in self.hull.simplices:
            # Skip facets that include the extra point
            if max(facet) > len(self.hull.points) - 1:
                #print(facet,len(my_hull.bt_hull.points))
                continue
            m = self.hull.points[facet]
            m[:, -1] = 1
            #print(m)
            if abs(np.linalg.det(m)) > 1e-14:
                tmp_simplices.append(facet)
        final_simplices = []
        for facet in tmp_simplices:
            # remove the simplex containing the vertices of [0,0,0],[0,1,0],[1,0,0] 
            if not np.all(self.points_0_index==np.sort(facet)):
                final_simplices.append(facet)
        return final_simplices 

    #@lru_cache(1) try to get this work?
    def get_facet_and_simplex(self,comp):
        # from pymatgen
        """
        Get any facet that a composition falls into. Cached so successive calls. 
        """
        #
        
        #tol = 1e-12
        for f, s in zip(self.final_simplices, self.pmg_simplexes):
            if s.in_simplex(comp, 1e-9):
                return f, s

    def calc_energy_at_comp(self,c):
        if self.use_bary:
            comp = self.barycentric_coordinates_2D(c)
        else:
            comp = c
        # comp: np.array([x1,y1])
        # get the hull energy by interpolation
        facet, simplex = self.get_facet_and_simplex(comp)
        comp_in_simplex = simplex.bary_coords(comp)

        
        verts_e = np.array([self.points[f][2] for f, c_x in zip(facet, comp_in_simplex) if abs(c_x) > 1e-14])
        verts_c = np.array([c_x for f, c_x in zip(facet, comp_in_simplex) if abs(c_x) > 1e-14])
        E_f = np.dot(verts_e,verts_c)
        

        return E_f

    def create_equilateral_triangle(self,side_length=1):
        A = np.array([0, 0])
        B = np.array([side_length, 0])
        height = side_length * np.sqrt(3) / 2
        C = np.array([side_length / 2, height])
        return np.array([A, B, C])
    def barycentric_coordinates_2D(self,points,triangle=None):
        try:
            A, B, C = triangle
            self.bary_tri = triangle
        except:
            self.bary_tri = self.create_equilateral_triangle(side_length=1)
        A,B,C = self.bary_tri
        v0 = B - A
        v1 = C - A
        #v2 = points - A
        u_v = np.dot(points,np.array([v0,v1]))
        #print(u_v)
        return np.array([u_v.T[0], u_v.T[1]]).T#, np.array([u_v.T[0], u_v.T[1], 1 - u_v.T[0] - u_v.T[1]]).T

    def calc_points_bary(self):
        points_bary = self.barycentric_coordinates_2D(self.points[:,:2])
        all_points_bary = self.barycentric_coordinates_2D(self.points_all[:,:2])
        self.all_points_bary = np.array([all_points_bary.T[0],all_points_bary.T[1],self.points_all.T[2]]).T
        self.points_bary = np.array([points_bary.T[0],points_bary.T[1],self.points.T[2]]).T

# class :
class my_nd_hull_funcs:
    
    def __init__(self,points,E_nu=0):
        # points
        # n x m array, n rows of configurations; m columns of coordinates, the last column is Energy
        self.points = points 
        # uncertainties of each point
        self.E_nu = E_nu 
    
    def get_bottom_hull(self):
        self.points.T[-1]
        self.points.T
        