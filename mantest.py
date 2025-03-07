from manim import *
import numpy as np

class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle).set_run_time(5))
          # show the circle on screen

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.rotate(PI / 4)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation

class SquareAndCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency

        square = Square()  # create a square
        square.set_fill(BLUE, opacity=0.5)  # set the color and transparency

        square.next_to(circle, RIGHT, buff=0.5)  # set the position
        self.play(Create(circle), Create(square))  # show the shapes on screen

class AnimatedSquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        square = Square()  # create a square

        self.play(Create(square))  # show the square on screen
        self.play(square.animate.rotate(PI / 4))  # rotate the square
        self.play(Transform(square, circle))  # transform the square into a circle
        self.play(
            square.animate.set_fill(PINK, opacity=0.5)
        )  # color the circle on screen

class DifferentRotations(Scene):
    def construct(self):
        left_square = Square(color=BLUE, fill_opacity=0.7).shift(2 * LEFT)
        right_square = Square(color=GREEN, fill_opacity=0.7).shift(2 * RIGHT)
        self.play(
            left_square.animate.rotate(PI), Rotate(right_square, angle=PI), run_time=2
        )
        self.wait()

class TwoTransforms(Scene):
    def transform(self):
        a = Circle()
        b = Square()
        c = Triangle()
        self.play(Transform(a, b))
        self.play(Transform(a, c))
        self.play(FadeOut(a))

    def replacement_transform(self):
        a = Circle()
        b = Square()
        c = Triangle()
        self.play(ReplacementTransform(a, b))
        self.play(ReplacementTransform(b, c))
        self.play(FadeOut(c))

    def construct(self):
        self.transform()
        self.wait(0.5)  # wait for 0.5 seconds
        self.replacement_transform()

class RotatingPlane(ThreeDScene):
    def construct(self):

        axes = ThreeDAxes()

        arrow_direction = np.array([1, 2, -2])  # Example direction
        arrow_direction_normalized = arrow_direction / np.linalg.norm(arrow_direction)
        arrow = Arrow3D(
            start=ORIGIN,
            end=2 * arrow_direction_normalized,
            color=RED,
        )

        # Define the plane using parametric equations: z = 0
        plane = Surface(
            lambda u, v: np.array([u, v, 0]),  # Parametric equation for the plane
            u_range=[-1, 1],  # X range
            v_range=[-1, 1],  # Y range
            resolution=(5, 5),
            checkerboard_colors=[YELLOW_B, YELLOW],  # Color of the surface
        )
        
        

        phi, theta = self.direction_to_euler_angles(arrow_direction_normalized)
        
        
        # Rotate the plane using Euler angles
        """
        self.play(ApplyMethod(plane.rotate(theta, axis=RIGHT),  # Rotate around x-axis
        plane.rotate(phi, axis=OUT)))
        """
        # Add the plane to the scene
        self.add(axes,arrow,plane)
        # Set up the camera to view the 3D scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        self.play(
            Rotate(
                plane,
                angle=theta,
                axis=RIGHT,
                about_point=ORIGIN,
                rate_func=linear).set_run_time(3)
                )
        self.play(
            Rotate(
                plane,
                angle=phi,
                axis=OUT,
                about_point=ORIGIN,
                rate_func=linear
                ).set_run_time(3)
        )
        

        self.begin_ambient_camera_rotation(22*DEGREES,about='theta')
        self.wait(2)
        self.stop_ambient_camera_rotation()

        self.play(
            Rotate(
                plane,
                angle=PI,
                axis=arrow_direction_normalized,
                about_point=ORIGIN,
                rate_func=linear
                ).set_run_time(3)
        )

        self.wait(2)
        

        """
        self.play(
            Rotate(
                plane,
                angle=2*PI,
                about_point=ORIGIN,
                rate_func=linear,
            ).set_run_time(3)
        )
        """
        
    

    def direction_to_euler_angles(self, direction):
        
        x, y, z = direction
        phi = -np.arctan2(x, y)
        theta = np.arcsin(z / np.linalg.norm(direction))
        return phi, theta

class ExampleArrow3D(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        arrow = Arrow3D(
            start=np.array([0, 0, 0]),
            end=np.array([2, 2, 2]),
            resolution=8
        )
        
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes)
       
        self.play(Create(arrow).set_run_time(2))
        arrow.rotate(PI,OUT,ORIGIN)
        
        x,y,z=(arrow.get_center()/np.linalg.norm(arrow.get_center()))*np.linalg.norm(arrow.get_end())
        newarrow = Arrow3D(
            start=ORIGIN,
            end=np.array([x,y,z]),
            resolution=8,
            color=RED
        )
        self.play(Create(newarrow).set_run_time(1))
        self.wait(2)


class EllipsoidScene(ThreeDScene):
    def construct(self):
        # Create the 3D axes
        axes = ThreeDAxes()
        
        # Set camera orientation to view from the top-right corner (phi ~ 75 degrees, theta ~ -45 degrees)
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        
        
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

        # Define a point on the surface where the normal vector will be drawn
        u, v = np.pi/4, np.pi/4; # Choose a point in the positive x, y, z section

        # Create the normal vector at the chosen point
        normal_vector = self.get_normal_vector(u, v, a=3, b=2, c=1)
        
        # Print the normal vector's position for debugging purposes
        print("Normal Vector Position:", normal_vector)

        point=self.get_point_on_surface(u, v, a=3, b=2, c=1)
        # Create a small dot at the point where the normal is
        point_on_surface = Dot3D(point, color=YELLOW)
        
        # Create the normal vector (an arrow)
        normal_arrow = Arrow3D(start=0, end=normal_vector, color=RED)
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
        
        
        phi, theta = self.direction_to_euler_angles(normal_vector)
        plane.shift(point)

        # Animate the normal vector's extension from the surface
        self.play(Create(point_on_surface).set_run_time(1.5))
        self.play( Create(normal_arrow).set_run_time(1.5) )
        self.pause(1)
        self.begin_ambient_camera_rotation(22*DEGREES,about='theta')
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.pause(1)
        self.play(Create(plane))
        self.play(
            Rotate(
                plane,
                angle=theta,
                axis=RIGHT,
                about_point=point,
                rate_func=linear).set_run_time(3)
                )
        self.play(
            Rotate(
                plane,
                angle=phi,
                axis=OUT,
                about_point=point,
                rate_func=linear
                ).set_run_time(3)
        )
        
        self.pause(1)
        elip=Intersection(plane,ellipsoid,color=GREEN, fill_opacity=10)
        self.play(Create(elip))
        
        self.play(
            Rotate(
                plane,
                angle=PI,
                axis=normal_vector,
                about_point=point,
                rate_func=linear
                ).set_run_time(3)
        )
        
        

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
        
        """
        plane_normal_arrow_origin=Arrow3D(0,plane_normal, color=GREEN)
        self.play(Create(plane_normal_arrow).set_run_time(1.5))
        self.play(Create(plane_normal_arrow_origin))
        self.play(Create(Arrow3D(0,point, color=PURPLE)))
        
        plane_normal_arrow_origin=Arrow3D(0,plane_normal, color=GREEN)
        self.play(Create(plane_normal_arrow))
        self.play(Create(plane_normal_arrow_origin))
        intersection_curve = self.get_intersection_curve(a, b, c, plane_normal,elipsoid_normal,point)
        u_dir_arrow=Arrow3D(start=0,end=u_dir,color=GREEN_A)
        v_dir_arrow=Arrow3D(start=0,end=v_dir,color=GREEN_B)
        c_dot=Dot3D(cpoint, color=WHITE)
        c_arrow=Arrow3D(0,cpoint, color=WHITE)
        self.play(Create(u_dir_arrow))
        self.play(Create(v_dir_arrow))
        self.play(Create(c_dot))
        self.play(Create(c_arrow))
        """

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





    

    

class IntersectScene(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # Ellipsoid parameters
        a, b, c = 3, 2, 1
        ellipsoid = self.get_ellipsoid(a, b, c, color=BLUE)
        ellipsoid.move_to(ORIGIN)
        self.add(axes, ellipsoid)

        # Plane parameters
        normal_vector = np.array([1, 1, 1])  # Define normal vector
        normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize it
        point_on_plane = np.array([0, 0, 0.5])  # Define a point on the plane

        # Check if the plane intersects the ellipsoid
        if not self.plane_intersects_ellipsoid(a, b, c, normal_vector, point_on_plane):
            raise ValueError("The plane does not intersect the ellipsoid.")

        # Compute intersection curve
        intersection_curve = self.get_intersection_curve(a, b, c, normal_vector, point_on_plane, color=RED)

        # Add plane
        plane = Surface(
            lambda u, v: point_on_plane + u * np.cross(normal_vector, [0, 0, 1]) + v * np.cross(normal_vector, [1, 0, 0]),
            u_range=[-5, 5],
            v_range=[-5, 5],
            fill_opacity=0.3,
            checkerboard_colors=[YELLOW_B, YELLOW],
        )

        self.play(Create(ellipsoid))
        self.play(Create(plane))
        self.play(Create(intersection_curve))
        self.wait(2)
        self.begin_ambient_camera_rotation(PI,about='theta')
        self.wait(2)

    def get_ellipsoid(self, a, b, c, color):
        return Surface(
            lambda u, v: np.array([
                a * np.cos(u) * np.sin(v),
                b * np.sin(u) * np.sin(v),
                c * np.cos(v)
            ]),
            u_range=[0, TAU],
            v_range=[0, PI],
            resolution=(30, 30),
            fill_opacity=0.5,
            checkerboard_colors=[color, color],
        )

    def plane_intersects_ellipsoid(self, a, b, c, normal, point):
        """
        Checks if the given plane intersects the ellipsoid.
        """
        A, B, C = normal
        D = -np.dot(normal, point)

        # Ellipsoid's semi-principal axis along the plane's normal
        effective_radius = 1 / np.sqrt((A / a)**2 + (B / b)**2 + (C / c)**2)

        # Distance from origin to the plane
        plane_distance = abs(D) / np.linalg.norm(normal)

        return plane_distance <= effective_radius


    def get_intersection_curve(self, a, b, c, normal, point, color):
        """
        Computes the parametric function for the intersection curve.
        """
        # Define a local basis for the plane
        u_direction = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(u_direction) < 1e-6:
            u_direction = np.cross(normal, [0, 1, 0])
        u_direction = u_direction / np.linalg.norm(u_direction)
        v_direction = np.cross(normal, u_direction)

        # Compute parametric curve
        return ParametricFunction(
            lambda t: self.compute_intersection_point(a, b, c, point, t, u_direction, v_direction),
            t_range=[0, TAU],
            color=color,
        )

    def compute_intersection_point(self, a, b, c, point, t, u_dir, v_dir):
        """
        Solves the parametric equations for the intersection of the ellipsoid with the plane.
        """
        u = np.cos(t)
        v = np.sin(t)
        local_point = u * u_dir + v * v_dir + point
        x, y, z = local_point
        # Scale back to the ellipsoid
        x /= a
        y /= b
        z /= c
        norm_factor = np.sqrt(x**2 + y**2 + z**2)
        x *= a / norm_factor
        y *= b / norm_factor
        z *= c / norm_factor
        return np.array([x, y, z])


