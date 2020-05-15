"""
MIT License

Copyright (c) 2017 Cyrille Rossant

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
import PIL

import os
import time
import pdb

w = 256
h = 256
SPP = 4 # should be power of 2

path = "./Images/"
path_texture = "./Texture/"
path_results = "./Results/"
texture_img = plt.imread(path_texture + "texture.jpg") / 255.0

# get pixel for background rectangle
def get_texture_pixel(texture, pos):
    index_w     = np.round((np.array(pos) + 2.5) / 5 * (texture.shape[0] - 1)).astype(int)[0]
    index_h     = (texture.shape[1] - 1) - np.round((np.array(pos) + 2.5) / 5 * (texture.shape[1] - 1)).astype(int)[1]
    pixel       = texture[index_h, index_w, :]
    return pixel


# compute color using four maps and cook-torrance's algorithm
def render_from_maps(diff_map, norm_map, rough_map, spec_map, camera_pos, light_pos, pos):
    index_w     = np.round((np.array(pos) + 0.5) * 255).astype(int)[0]
    index_h     = 255 - np.round((np.array(pos) + 0.5) * 255).astype(int)[1]
    diff_pos    = diff_map[index_h, index_w, :] ** 2.2
    spec_pos    = spec_map[index_h, index_w, :] ** 2.2
    norm_pos    = norm_map[index_h, index_w, :]
    rough_pos   = rough_map[index_h, index_w, 0]

    light_vec   = normalize(light_pos - pos)
    view_vec    = normalize(camera_pos - pos)
    light_view_half_vec = normalize((light_vec + view_vec) / 2.)

    vertical_normal_vec = np.array([0., 0., -1.])
    normal_vec          = normalize(norm_pos * 2 - 1) * np.array([1, 1, -1])

    normal_dot_light    = np.clip(np.dot(normal_vec, light_vec), 0, 1)
    normal_dot_half     = np.clip(np.dot(normal_vec, light_view_half_vec), 0, 1)
    view_dot_half       = np.clip(np.dot(view_vec, light_view_half_vec), 0, 1)
    view_dot_norm       = np.clip(np.dot(view_vec, normal_vec), 0, 1)
    light_dot_vertical  = np.clip(np.dot(light_vec, vertical_normal_vec), 0, 1)

    diff_pos = diff_pos * (1 - spec_pos) / np.pi
    rough_pos2 = rough_pos * rough_pos
    normal_dot_half2 = normal_dot_half * normal_dot_half
    
    deno_D = (rough_pos2 * rough_pos2 - 1) * normal_dot_half2 + 1
    D = (rough_pos2 / deno_D) * (rough_pos2 / deno_D) / np.pi
    
    EPSILON = 0.0001
    G_1 = 1. / (normal_dot_light * (1 - rough_pos2 / 2) + rough_pos2 / 2 + EPSILON)
    G_2 = 1. / (view_dot_norm * (1 - rough_pos2 / 2) + rough_pos2 / 2 + EPSILON)
    G = G_1 * G_2
    
    F = spec_pos + (1 - spec_pos) * 2 ** ((-5.55473 * view_dot_half - 6.98316) * view_dot_half)
    
    specular = G * F * D / (4 + EPSILON)
    
    color = np.clip(np.pi * (diff_pos + specular) * normal_dot_light, 0, 1)

    return color ** (1 / 2.2)


def normalize(x):
    x /= np.linalg.norm(x)
    return x

def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

def intersect_plane2(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

def intersect_rectangle(O, D, P, N, w, h):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    hit_point = O + D * d
    if abs(hit_point[0]) > 0.5 or abs(hit_point[1]) > 0.5: # assume: P=[0., 0., 1.], N=[0., 0, -1], w=h=1
        return np.inf
    return d

def intersect_rectangle2(O, D, P, N, w, h):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    hit_point = O + D * d
    if abs(hit_point[0]) > 2.5 or abs(hit_point[1]) > 2.5: # assume: P=[0., 0., 2.], N=[0., 0, -1], w=h=5
        return np.inf
    return d

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def intersect(O, D, obj):
    # pdb.set_trace()
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'plane2':
        return intersect_plane2(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'rectangle':
        return intersect_rectangle(O, D, obj['position'], obj['normal'], obj['width'], obj['height'])
    elif obj['type'] == 'rectangle2':
        return intersect_rectangle2(O, D, obj['position'], obj['normal'], obj['width'], obj['height'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])

def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'plane2':
        N = obj['normal']
    elif obj['type'] == 'rectangle':
        N = obj['normal']
    elif obj['type'] == 'rectangle2':
        N = obj['normal']
    return N
    
def get_color(obj, M):
    # pdb.set_trace()
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        # pdb.set_trace()
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    N = get_normal(obj, M)


    if obj['type'] != 'rectangle' and obj['type'] != 'rectangle2':
        # Find properties of the object.
        color = get_color(obj, M)
        toL = normalize(L - M)
        toO = normalize(O - M)
        # Start computing the color.
        col_ray = ambient
        # Lambert shading (diffuse).
        col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
        # Blinn-Phong shading (specular).
        col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * color_light
    elif obj['type'] == 'rectangle2':
        col_ray = get_texture_pixel(texture_img, M)
    else:
        col_ray = render_from_maps(diff, norm, rough, spec, O, L, M) # O: camera position, L: light position

    return obj, M, N, col_ray

def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position), 
        radius=np.array(radius), color=np.array(color), reflection=.5)
    
def add_plane(position, normal):
    return dict(type='plane', position=np.array(position), 
        normal=np.array(normal),
        color=lambda M: (color_plane0 
            if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
        diffuse_c=.75, specular_c=.5, reflection=.25)
    
def add_plane2(position, normal, color):
    return dict(type='plane2', position=np.array(position), 
        normal=np.array(normal),
        color=lambda M: (np.array(color) 
            if (int((M[0]) * 2) % 2) == (int(M[1] * 2) % 2) else np.array(color)),
        diffuse_c=.75, specular_c=0., reflection=0.)
    
def add_rectangle(position, normal, width, height, color):
    return dict(type='rectangle', position=np.array(position), 
        normal=np.array(normal),
        width=width,
        height=height,
        color=np.array(color),
        diffuse_c=.75, specular_c=0.3, reflection=0.)
    
def add_rectangle2(position, normal, width, height, color):
    return dict(type='rectangle2', position=np.array(position), 
        normal=np.array(normal),
        width=width,
        height=height,
        color=np.array(color),
        diffuse_c=.75, specular_c=0.3, reflection=0.)


if __name__ == '__main__':
    imglist = sorted(os.listdir(path))
    maps = [] # remove some hidden files, like .DS_Store in mac, from imglist
    for i in imglist:
        if os.path.splitext(i)[1] == '.jpg':
            maps += [i]

    for scene_id in range(int(len(maps) / 4)):
        scene_start_time = time.time()
        diff    = plt.imread(path + maps[scene_id * 4]) / 255.0
        norm    = plt.imread(path + maps[scene_id * 4 + 1]) / 255.0
        rough   = plt.imread(path + maps[scene_id * 4 + 2]) / 255.0
        spec    = plt.imread(path + maps[scene_id * 4 + 3]) / 255.0
        
        # pdb.set_trace()
        # List of objects.
        color_plane0 = 1. * np.ones(3)
        color_plane1 = 0. * np.ones(3)
        scene = [add_rectangle([0., 0., 1.], [0., 0., -1.], 1.0, 1.0, [0., 0., 1.]),
                add_rectangle2([0., 0., 2.], [0., 0., -1.], 1.0, 1.0, [0., 0., 1.]),
            ]

        # Light position and color.
        lights_pos = [np.array([0., 0.5, 0.]), np.array([0., 0., 0.]), np.array([0., -0.5, 0.]),
                      np.array([-0.5, -0.5, 0.]), np.array([-0.5, 0., 0.]), np.array([-0.5, 0.5, 0.]),
                      np.array([-1., 0.5, 0.]), np.array([-1., 0., 0.]), np.array([-1., -0.5, 0.]),
                      np.array([-1.5, -0.5, 0.]), np.array([-1.5, 0., 0.]), np.array([-1.5, 0.5, 0.]),
                     ]
        color_light = np.ones(3)

        # Default light and material parameters.
        ambient = .05
        diffuse_c = 1.
        specular_c = 1.
        specular_k = 50

        depth_max = 1  # Maximum number of light reflections.
        col = np.zeros(3)  # Current color.
        O = np.array([3., 0., -3.])  # Camera.
        Q = np.array([0., 0., 1.])  # Camera pointing to.
        d = 950 # focal distance

        img = np.zeros((h, w, 3))

        # camera space
        up = np.array([-0.5, 1., 0.])
        w_ = normalize(O - Q)
        u_ = normalize(np.cross(up, w_))
        v_ = np.cross(w_, u_)
        def get_direction(x, y):
            dir = normalize(x * u_ + y * v_ - d * w_)
            return dir

        spp = SPP 
        n_ = int(np.sqrt(spp))

        for light_id in range(len(lights_pos)):
            light_start_time = time.time()
            L = lights_pos[light_id]
            # pdb.set_trace()
            # Loop through all pixels.
            for i in range(w):
                if i % 10 == 0:
                    print(i / float(w) * 100, "%")
                for j in range(h):
                    col[:] = 0
                    for p in range(n_):
                        for q in range(n_):
                            x = -(i - 0.5 * w + (p + 0.5) / n_)
                            y = (j - 0.5 * h + (q + 0.5) / n_)
                            direction = get_direction(x, y)
                            depth = 0
                            rayO, rayD = O, direction
                            reflection = 1.
                            # Loop through initial and secondary rays.
                            while depth < depth_max:
                                traced = trace_ray(rayO, rayD)
                                if not traced:
                                    break
                                obj, M, N, col_ray = traced
                                # Reflection: create a new ray.
                                rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
                                depth += 1
                                col += reflection * col_ray
                                reflection *= obj.get('reflection', 1.)
                    img[h - j - 1, i, :] = np.clip(col / spp, 0, 1)

            # pdb.set_trace()
            plt.imsave(path_results + 'scene' + str(int(maps[scene_id * 4].split(';', 1)[0])).zfill(3) + '_light' + str(light_id).zfill(2) + '(' + str(L[0]) + ', ' + str(L[1]) + ', ' + str(L[2]) + ').png', img)
            print("light_time=%.3f"%(time.time()-light_start_time))
            print('scene' + str(int(maps[scene_id * 4].split(';', 1)[0])).zfill(3) + '_light' + str(light_id).zfill(2) + '(' + str(L[0]) + ', ' + str(L[1]) + ', ' + str(L[2]) + ').png')
        
        print("scene_time=%.3f"%(time.time()-scene_start_time))
