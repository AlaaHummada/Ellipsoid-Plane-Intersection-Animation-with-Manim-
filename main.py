
from manim import *
import numpy as np

class Ellipses(ThreeDScene):
    def construct(self):

        # Create the 3D axes
        axes = ThreeDAxes()
        
        # Set camera orientation to view from the top-right corner (phi ~ 75 degrees, theta ~ -45 degrees)
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        # Note: The RIGHT,UP,OUT correspond to x,y,z.
        # However at the scene the OUT (z) direction is pointing upward.
        # the UP (y) direction is at the right.
        # the RIGHT (x) direction is at the left.
        
        #Elipsoid parameters
        a, b, c = 3, 2, 1
        
        # Create the ellipsoid
        ellipsoid = self.get_ellipsoid(
            a, b, c,  # Axes of the ellipsoid
            color=BLUE
        )
        
        # Position the ellipsoid at the center (origin)
        ellipsoid.move_to(ORIGIN)

        # Add the axes and ellipsoid to the scene
        self.add(axes, ellipsoid)

        # Define a point on the elipsoid surface where the normal vector will be drawn
        u, v = np.pi/4, np.pi/4; # Choose a point in the positive x, y, z section

        # Create the elispoid surface normal vector at the chosen point
        normal_vector = self.get_normal_vector(u, v, a=3, b=2, c=1)

        #Calculate the location of the elipsoid surface point given its uv coordinates
        point=self.get_point_on_surface(u, v, a=3, b=2, c=1)

        # Create a small dot at the elipsoid surface point
        point_on_surface = Dot3D(point, color=YELLOW)
        
        # Creates an arrow object for the elipsoid surface normal vector
        normal_arrow = Arrow3D(start=0, end=normal_vector, color=RED)

        #Shifts the arrow start to the position of the elispoid surface point
        normal_arrow.shift((point))
        
        # Define the plane using parametric equations: z = 0
        plane = Surface(
            lambda u, v: np.array([u, v, 0]),  # Parametric equation for the plane
            u_range=[-5, 5],  # X range
            v_range=[-5, 5],  # Y range
            resolution=(10, 10),
            fill_opacity=0.1,
            checkerboard_colors=[YELLOW_B, YELLOW],  # Color of the surface
        )
        
        # Let the plane normal initially point at the positive OUT axis
        plane_normal=np.array([0,0,1])
        # Create an arrow for the plane normal
        plane_normal_arrow=Arrow3D(start=0,end=plane_normal,color=GREEN)
        
        # Shifts the plane center to the elipsoid surface point 
        plane.shift(point)
        # Shifts the plane normal arrow to the elipsoid surface point
        plane_normal_arrow.shift(point)

        # Get the eular angles from the elipsoid normal vector
        phi, theta = self.direction_to_euler_angles(normal_vector)
        # Which will be used to apply the right rotations such that the plane becomes parallel with the elipsoid normal surface
        
        # Play a create animation of the elipsoid surface point and the elipsoid surface normal as it extends from the point
        self.play(Create(point_on_surface).set_run_time(1.5))
        self.play(Create(normal_arrow).set_run_time(1.5) )
        # set_run_time(1.5) makes each animation run at 1.5s
         
        # Pauses the scene for 1 second
        self.pause(1)

        # Begins an ambient camera rotation at a rate of 22 degrees/s
        self.begin_ambient_camera_rotation(22*DEGREES,about='theta')
        # The camera will rotate for 2 seconds before it stops
        self.wait(2)
        self.stop_ambient_camera_rotation()

        #Pause the scene for 1 sec
        self.pause(1)
        
        # Create Plane Animation
        self.play(Create(plane))

        #Rotate the plane and its normal such that it is paralled to the eliposoid surface
        self.rotateplane(plane,theta,RIGHT,point,plane_normal_arrow)
        self.rotateplane(plane,phi,OUT,point,plane_normal_arrow)

        #Pause the scene
        self.pause(1)

        #Get the plane normal vector from the plane normal arrow after shift and rotation transformations 
        # And normalize it
        plane_normal=plane_normal_arrow.get_center()-point
        plane_normal=plane_normal/np.linalg.norm(plane_normal)

        # Get the elipsoid surface normal vector from the elipsoid normal arrow after shift and rotation transformations 
        # And normalize it
        elipsoid_normal=normal_arrow.get_center()-point
        elipsoid_normal=elipsoid_normal/np.linalg.norm(elipsoid_normal)
        
        
        intersection_curve = self.get_intersection_curve(a, b, c, plane_normal,normal_vector,point)
        
        self.play(Create(intersection_curve))
        self.wait(1)

        self.begin_ambient_camera_rotation(2*PI/3,about='theta')
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.pause(1)

        self.rotateplane(plane,PI/2,normal_vector,point,plane_normal_arrow)
        
        self.pause(1)

        # Get the plane normal vector from the plane normal arrow after shift and rotation transformations 
        # And normalize it
        plane_normal=plane_normal_arrow.get_center()-point
        plane_normal=plane_normal/np.linalg.norm(plane_normal)

        # Get the elipsoid surface normal vector from the elipsoid normal arrow after shift and rotation transformations 
        # And normalize it
        elipsoid_normal=normal_arrow.get_center()-point
        elipsoid_normal=elipsoid_normal/np.linalg.norm(elipsoid_normal)
        
        intersection_curve = self.get_intersection_curve(a, b, c, plane_normal,elipsoid_normal,point)
        
        self.play(Create(intersection_curve))
        
        self.wait(2)
        

    def get_ellipsoid(self, a, b, c, color):
        # Parameterize the ellipsoid using a surface
        return Surface(
            lambda u, v: np.array([
                a * np.cos(u) * np.sin(v),
                b * np.sin(u) * np.sin(v),
                c * np.cos(v)
            ]),
            u_range=[0, TAU],
            v_range=[0, PI],
            resolution=(20, 20),  # Higher resolution for smoother surface
            fill_opacity=0.5,
            checkerboard_colors=[color, color],
        )

    def get_normal_vector(self, u, v, a, b, c):
        # Calculate the parametric derivatives of the ellipsoid
        # Parametric equations: x = a*cos(u)*sin(v), y = b*sin(u)*sin(v), z = c*cos(v)
        
        # Partial derivatives with respect to u and v
        dP_du = np.array([-a * np.sin(u) * np.sin(v), b * np.cos(u) * np.sin(v), 0])
        dP_dv = np.array([a * np.cos(u) * np.cos(v), b * np.sin(u) * np.cos(v), -c * np.sin(v)])

        # Normal vector is the cross product of the two partial derivatives
        normal = np.cross(dP_du, dP_dv)
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Scale the normal vector for visibility (multiply by a factor)
        normal = -normal  # Increase size if needed
        
        # Return the final position of the normal vector
        return normal
    
    def get_point_on_surface(self, u, v, a, b, c):
        point_on_surface = np.array([
            a * np.cos(u) * np.sin(v),
            b * np.sin(u) * np.sin(v),
            c * np.cos(v)
        ])
        return point_on_surface
    
    def direction_to_euler_angles(self, direction):
        
        x, y, z = direction
        phi = -np.arctan2(x, y)
        theta = np.arcsin(z / np.linalg.norm(direction))
        return phi, theta 

    def get_intersection_curve(self, a, b, c, normalplane,normalelipsoid, point):
        
        normalplane = normalplane / np.linalg.norm(normalplane)
        normalelipsoid = normalelipsoid / np.linalg.norm(normalplane)
        A, B, C = normalplane
        D = -np.dot(point, normalplane)
        cpoint=(-D)*normalplane
        cn=cpoint/np.linalg.norm(cpoint)

        
        v_dir=normalelipsoid
        v_dir=v_dir/np.linalg.norm(v_dir)

        u_dir=np.cross(normalplane,v_dir)
        u_dir=u_dir/np.linalg.norm(u_dir)

        def intersection_func(t):
            D1=np.array([1/a,1/b,1/c])
            Dn=D1*cn
            Du=D1*u_dir
            Dv=D1*v_dir
            beta_1=np.dot(Du,Du)
            beta_2=np.dot(Dv,Dv)
            kappa=np.linalg.norm(cpoint)
            d=(kappa**2)/(np.linalg.norm(np.array([a*A,b*B,c*C]))**2)
            omega=(1/2)*np.arctan((2*np.dot(Du,Dv))/(beta_1-beta_2))
            u_dirnew=np.cos(omega)*u_dir+np.sin(omega)*v_dir
            v_dirnew=-np.sin(omega)*u_dir+np.cos(omega)*v_dir
            Dunew=D1*u_dirnew
            Dvnew=D1*v_dirnew
            beta_1new=np.dot(Dunew,Dunew)
            beta_2new=np.dot(Dvnew,Dvnew)

            u_scale=np.sqrt((1-d)/beta_1new)
            v_scale=np.sqrt((1-d)/beta_2new)
            
            u0=-kappa*(np.dot(Dn,Dunew)/beta_1new)
            v0=-kappa*(np.dot(Dn,Dvnew)/beta_2new)

            u = (u_scale)*np.cos(t) + u0
            v = (v_scale)*np.sin(t) + v0
            local_point = u*u_dirnew + v*v_dirnew + cpoint
            x, y, z = local_point
            
            
            
            
            return np.array([x, y, z])
            
            
        return ParametricFunction(
            intersection_func,
            t_range=[0, TAU],
            color=GREEN,
        )
    
    def rotateplane(self,plane,theta,RIGHT,point,plane_normal_arrow):
        self.play(
                Rotate(
                    plane,
                    angle=theta,
                    axis=RIGHT,
                    about_point=point,
                    rate_func=linear).set_run_time(2)
                    )
            
        #Rotate the plane normal arrow with it
        plane_normal_arrow.rotate(theta,RIGHT,point)