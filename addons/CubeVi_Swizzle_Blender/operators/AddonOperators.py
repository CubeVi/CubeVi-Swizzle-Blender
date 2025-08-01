# ##### BEGIN GPL LICENSE BLOCK #####
#
#  Copyright © GJQ, OpenstageAI, Cubestage
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ##### END GPL LICENSE BLOCK #####
import sys
import os
import importlib
import bpy
import subprocess
# import screeninfo

path = bpy.utils.script_paths()
print(path[0])

wheels_folder_path = rf"{path[1]}\addons\CubeVi_Swizzle_Blender\wheels"
# wheels_folder_path = rf"D:\desktop\wheels"

print(wheels_folder_path)
# wheels_folder_path = r'D:\desktop\wheels'
if wheels_folder_path not in sys.path:
    sys.path.append(wheels_folder_path)
# Iterate through all subdirectories in the wheels folder
for folder_name in os.listdir(wheels_folder_path):
    folder_path = os.path.join(wheels_folder_path, folder_name)

    # Check if it is a directory
    if os.path.isdir(folder_path):
        # Assume each subdirectory contains an importable module
        try:
            # Dynamically import the module from the subdirectory
            module_spec = importlib.util.find_spec(folder_name)
            if module_spec is None:
                print(f"Module '{folder_name}' not found.")
            else:
                module = importlib.import_module(folder_name)
                print(f"Module '{folder_name}' successfully loaded.")
        except Exception as e:
            print(f"Error loading module '{folder_name}': {e}")

from screeninfo import get_monitors
import bpy
import gpu
import sys
# from screeninfo import get_monitors
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector
import time
from gpu_extras.presets import draw_texture_2d
import math
import numpy as np
from PIL import Image
import os
from bpy.props import IntProperty


import asyncio
import json
import cv2
import base64
from hashlib import md5
import websocket
import threading
from bpy.props import StringProperty


from Cryptodome import Random
from Cryptodome.Cipher import AES


flag = False
linenumber = None
obliquity = None
deviation = None
is_drawing = False
frustum_draw_handler = None

window_name = "Real time Display"



def initialize_cv_window():
    window_name = "Real time Display"
    monitors = get_monitors()

    # Find monitor with resolution 1440x2560
    for monitor in monitors:
        if monitor.width == 1440 and monitor.height == 2560:
            # First create a window
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # Resize window to screen resolution
            cv2.resizeWindow(window_name, monitor.width, monitor.height)

            # Move to top-left corner of target monitor
            cv2.moveWindow(window_name, monitor.x, monitor.y)

            # Set to fullscreen
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            break


def update_cv_window(window, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, frame)

# Function to decrypt platform device config information
keycode = "3f5e1a2b4c6d7e8f9a0b1c2d3e4f5a6b"


def unpad(data):
    return data[:-(data[-1] if type(data[-1]) == int else ord(data[-1]))]

def bytes_to_key(data, salt, output=48):
    assert len(salt) == 8, len(salt)
    data += salt
    key = md5(data).digest()
    final_key = key
    while len(final_key) < output:
        key = md5(key + data).digest()
        final_key += key
    return final_key[:output]

def decrypt(encrypted, passphrase):
    encrypted = base64.b64decode(encrypted)
    assert encrypted[0:8] == b"Salted__"
    salt = encrypted[8:16]
    key_iv = bytes_to_key(passphrase, salt, 32 + 16)
    key = key_iv[:32]
    iv = key_iv[32:]
    aes = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(aes.decrypt(encrypted[16:]))
    stringData = pt.decode('utf-8')
    data = json.loads(stringData)
    return data

def set_1():
    bpy.context.scene.render.resolution_x = 540
    bpy.context.scene.render.resolution_y = 960



class FrustumOperator(bpy.types.Operator):
    """Show the camera frustum"""
    bl_idname = "object.frustum"
    bl_label = "Show Frustum"
    bl_options = {'REGISTER', 'UNDO'}


    def setupCameraFrustumShader(self):
        """Set up shader for drawing camera frustum"""
        pass

    def drawCameraFrustum(self, context):
        """Draw camera frustum"""
        scene = context.scene
        camera = scene.camera
        clip_near = scene.clip_near
        clip_far = scene.clip_far
        focal_plane = scene.focal_plane

        # Get camera frustum vertices
        coords_local = self.calculate_frustum_coordinates(camera, context, clip_near, clip_far, focal_plane)

        # Create frustum visualization
        self.create_frustum_visualization(context, coords_local, camera)

    def calculate_frustum_coordinates(self, camera, context, clip_near, clip_far, focal_plane):
        """
        Calculate four vertices of camera frustum and transform based on camera position and orientation
        """
        # Get camera world matrix
        view_matrix = camera.matrix_world.copy()
        view_frame = camera.data.view_frame(scene=context.scene)

        # Get frustum vertices
        scale = 1.39
        view_frame_upper_right = view_frame[0]/scale
        view_frame_lower_right = view_frame[1]/scale
        view_frame_lower_left = view_frame[2]/scale
        view_frame_upper_left = view_frame[3]/scale
        view_frame_distance = abs(view_frame_upper_right[2])/scale

        # Transform frustum coordinates using world matrix
        coords_local = [
            # near clipping plane
            (view_matrix @ (view_frame_lower_right * clip_near)),
            (view_matrix @ (view_frame_lower_left * clip_near)),
            (view_matrix @ (view_frame_upper_left * clip_near)),
            (view_matrix @ (view_frame_upper_right * clip_near)),
            # far clipping plane
            (view_matrix @ (view_frame_lower_right * clip_far)),
            (view_matrix @ (view_frame_lower_left * clip_far)),
            (view_matrix @ (view_frame_upper_left * clip_far)),
            (view_matrix @ (view_frame_upper_right * clip_far)),
            # focal plane
            (view_matrix @ (view_frame_lower_right * focal_plane)),
            (view_matrix @ (view_frame_lower_left * focal_plane)),
            (view_matrix @ (view_frame_upper_left * focal_plane)),
            (view_matrix @ (view_frame_upper_right * focal_plane))
        ]
        return coords_local

    def create_frustum_visualization(self, context, coords_local, camera):
        """
        Create frustum visualization (direct drawing)
        """
        # Define frustum boundary lines
        frustum_indices_lines = [
            # near plane
            (0, 1), (1, 2), (2, 3), (3, 0),
            # far plane
            (4, 5), (5, 6), (6, 7), (7, 4),
            # focal plane
            (8, 9), (9, 10), (10, 11), (11, 8),
            # connecting near and far planes
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        frustum_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch_lines = batch_for_shader(frustum_shader, 'LINES', {"pos": coords_local}, indices=frustum_indices_lines)

        # Draw frustum
        frustum_shader.bind()
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)

        # Draw frustum boundaries
        frustum_shader.uniform_float("color", (0.3, 0, 0, 1))  # Set color to red
        batch_lines.draw(frustum_shader)

        gpu.state.depth_mask_set(False)
        gpu.state.blend_set('ALPHA')

        # Fill frustum faces
        frustum_indices_faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 4, 5), (0, 5, 1),
            (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3),
            (3, 7, 4), (3, 4, 0),
        ]
        batch_faces = batch_for_shader(frustum_shader, 'TRIS', {"pos": coords_local}, indices=frustum_indices_faces)

        # Set other faces color to semi-transparent gray
        frustum_shader.uniform_float("color", (0.5, 0.5, 0.5, 0.1))  # Semi-transparent gray
        batch_faces.draw(frustum_shader)

        # Set focal plane color to yellow (yellow is (1, 1, 0))
        focal_plane_indices_faces = [
            (8, 9, 10), (8, 10, 11)
        ]
        focal_plane_faces = batch_for_shader(frustum_shader, 'TRIS', {"pos": coords_local},
                                             indices=focal_plane_indices_faces)
        frustum_shader.uniform_float("color", (1, 1, 0, 0.1))  # Set focal plane color to yellow
        focal_plane_faces.draw(frustum_shader)

        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')

    def start(self, context):
        """Start drawing camera frustum"""
        # Set up camera frustum and shader
        global is_drawing
        global frustum_draw_handler
        self.setupCameraFrustumShader()

        # If no draw handler exists, add it
        if is_drawing:
            bpy.types.SpaceView3D.draw_handler_remove(frustum_draw_handler, 'WINDOW')
            frustum_draw_handler = None
            context.area.tag_redraw()
        else:
            frustum_draw_handler = bpy.types.SpaceView3D.draw_handler_add(self.drawCameraFrustum, (context,),
                                                                          'WINDOW', 'POST_VIEW')
            context.area.tag_redraw()
        is_drawing = not is_drawing



    def execute(self, context):
        """Execute operator"""
        # Start or stop frustum drawing
        self.start(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        """Keep operator running and monitor events"""
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        """Check if frustum has been drawn when button is clicked"""
        self.start(context)
        return {'FINISHED'}



class connectOperator(bpy.types.Operator):
    """Connect to the device"""
    bl_idname = "object.connect"
    bl_label = "Connect"
    bl_options = {'REGISTER', 'UNDO'}
    _handle = None

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    items = []

    def execute(self, context: bpy.types.Context):
        print(get_monitors())
        print(context.scene.my_filepath)
        set_1()
        appdata_path = os.getenv('APPDATA')
        openstage_path = os.path.join(appdata_path, 'OpenstageAI', 'deviceConfig.json')
        cubestage_path = os.path.join(appdata_path, 'Cubestage', 'deviceConfig.json')
        
        if os.path.exists(openstage_path):
            config_path = openstage_path
        elif os.path.exists(cubestage_path):
            config_path = cubestage_path
        else:
            raise FileNotFoundError(f"deviceConfig.json not found in paths:\n{openstage_path}\n{cubestage_path}")
        
        operator_id = 0
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        with open(config_path, 'r', encoding='utf-8') as f:
            device_info = json.load(f)

        if device_info and 'config' in device_info:
            global obliquity, linenumber, deviation
            config = device_info['config']
            password = keycode.encode()
            data = decrypt(config, password)

            configData = data['config']
            linenumber = configData.get('lineNumber', '')
            obliquity = configData.get('obliquity', '')
            deviation = configData.get('deviation', '')
            self.report({"INFO"}, "Connection Successful")
            return {'FINISHED'}


class LFDSaveOperator(bpy.types.Operator):
    """Save the preview lightfield picture"""
    bl_idname = "object.save"
    bl_label = "Save LFD Preview Picture"
    bl_options = {'REGISTER', 'UNDO'}
    _handle = None

    # Prerequisites for operation
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    # Initialize data
    def __init__(self):
        self.offscreen = None  # For storing single camera texture
        self.final_offscreen = None  # For storing combined large texture
        self.display_offscreen = None  # Offscreen buffer for displaying texture
        self.shader = None
        self.clear_shader = None
        self.batch = None
        self.clear_batch = None
        self.display_batch = None  # Added for display_shader
        self.view_matrix = None
        self.projection_matrix = None
        self.render_width = 540  # Width of each texture
        self.render_height = 960  # Height of each texture
        self.grid_rows = 5
        self.grid_cols = 8
        self.final_width = self.render_width * self.grid_cols
        self.final_height = self.render_height * self.grid_rows
        self.display_shader = None  # For displaying final texture

    def setup_offscreen_rendering(self):
        """Set up small and large texture offscreens"""
        try:
            # Single texture offscreen
            self.offscreen = gpu.types.GPUOffScreen(self.render_width, self.render_height)

            # Final combined texture offscreen
            self.final_offscreen = gpu.types.GPUOffScreen(self.final_width, self.final_height)

            # Offscreen for displaying texture
            self.display_offscreen = gpu.types.GPUOffScreen(1440, 2560)

            return True
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create offscreen buffer: {e}")
            print(f"Failed to create offscreen buffer: {e}")
            return False

    def setup_shader(self):
        """Create shader for drawing textures"""
        vertex_shader = '''
            uniform vec2 scale;
            uniform vec2 offset;
            in vec2 pos;
            in vec2 texCoord;
            out vec2 fragTexCoord;

            void main()
            {
                gl_Position = vec4(pos * scale + offset, 0.0, 1.0);
                fragTexCoord = texCoord;
            }
        '''

        fragment_shader = '''
            uniform sampler2D image;
            in vec2 fragTexCoord;
            out vec4 FragColor;

            void main()
            {
                FragColor = texture(image, fragTexCoord);
            }
        '''

        try:
            self.shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
            # print("Shader compilation successful")
        except Exception as e:
            self.report({'ERROR'}, f"Shader compilation failed: {e}")
            print(f"Shader compilation failed: {e}")
            return False
        # Define vertices and indices
        vertices = [
            (-1, -1, 0, 0),  # Bottom-left
            (1, -1, 1, 0),  # Bottom-right
            (1, 1, 1, 1),  # Top-right
            (-1, 1, 0, 1)  # Top-left
        ]

        indices = [(0, 1, 2), (2, 3, 0)]

        self.batch = batch_for_shader(
            self.shader, 'TRIS',
            {"pos": [v[:2] for v in vertices], "texCoord": [v[2:] for v in vertices]},
            indices=indices
        )

        return True

    def setup_display_shader(self):
        """Create dedicated shader for display_offscreen"""
        vertex_shader = '''
            in vec2 pos;
            in vec2 texCoord;
            out vec2 fragTexCoord;

            void main()
            {
                gl_Position = vec4(pos, 0.0, 1.0);
                fragTexCoord = texCoord;
            }
        '''

        fragment_shader = '''
            uniform sampler2D image1;
            uniform float _OutputSizeX;
            uniform float _OutputSizeY;
            uniform float _Slope;
            uniform float _X0;
            uniform float _Interval;
            uniform float _ImgsCountAll;
            uniform float _ImgsCountX;
            uniform float _ImgsCountY;
            in vec2 fragTexCoord;
            out vec4 FragColor;

            float get_choice_float(vec2 pos, float bias) {
                float x = pos.x * _OutputSizeX + 0.5;
                float y = (1- pos.y) * _OutputSizeY + 0.5;
                // float y = pos.y * _OutputSizeY + 0.5;
                float x1 = (x + y * _Slope) * 3.0 + bias;
                float x_local = mod(x1 + _X0, _Interval);
                return (x_local / _Interval);
            }

            vec3 linear_to_srgb(vec3 linear) {
                bvec3 cutoff = lessThan(linear, vec3(0.0031308));
                vec3 higher = vec3(1.055) * pow(linear, vec3(1.0 / 2.4)) - vec3(0.055);
                vec3 lower = linear * vec3(12.92);
                return mix(higher, lower, cutoff);
            }

            vec2 get_uv_from_choice(vec2 pos, float choice_float) {
                float choice = floor(choice_float * _ImgsCountAll);
                vec2 choice_vec = vec2(
                _ImgsCountX - 1.0 - mod(choice, _ImgsCountX),  // Right to left
                // _ImgsCountY - 1.0 - floor(choice / _ImgsCountX) 
                floor(choice / _ImgsCountX) // Bottom to top
                );

                vec2 reciprocals = vec2(1.0 / _ImgsCountX, 1.0 / _ImgsCountY);
                return (choice_vec + pos) * reciprocals;
            }

            vec4 get_color(vec2 pos, float bias) {
                float choice_float = get_choice_float(pos, bias);
                vec2 sel_pos = get_uv_from_choice(pos, choice_float);
                return texture(image1, sel_pos);
            }

            void main() {
                vec4 color = get_color(fragTexCoord, 0.0);
                color.g = get_color(fragTexCoord, 1.0).g;
                color.b = get_color(fragTexCoord, 2.0).b;
                FragColor = vec4(linear_to_srgb(color.rgb), color.a);
            }
        '''

        try:
            self.display_shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
            # print("Display_offscreen shader compilation successful")
        except gpu.types.GPUShaderCompilationError as e:
            self.report({'ERROR'}, f"Display_offscreen shader compilation failed: {e}")
            print(f"Display_offscreen shader compilation failed: {e}")
            self.display_shader = None
            return False
        except Exception as e:
            self.report({'ERROR'}, f"Display_offscreen shader initialization failed: {e}")
            print(f"Display_offscreen shader initialization failed: {e}")
            self.display_shader = None
            return False

        # Define vertices and indices for drawing display texture
        vertices = [
            (-1, -1, 0, 0),  # Bottom-left
            (1, -1, 1, 0),  # Bottom-right
            (1, 1, 1, 1),  # Top-right
            (-1, 1, 0, 1)  # Top-left
        ]

        indices = [(0, 1, 2), (2, 3, 0)]

        self.display_batch = batch_for_shader(
            self.display_shader, 'TRIS',
            {"pos": [v[:2] for v in vertices], "texCoord": [v[2:] for v in vertices]},
            indices=indices
        )

        return True

    def setup_clear_shader(self):
        """Create simple shader for clearing color buffer"""
        # Use built-in 'UNIFORM_COLOR' shader
        self.clear_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.clear_batch = batch_for_shader(
            self.clear_shader, 'TRI_FAN',
            {"pos": [(-1, -1), (1, -1), (1, 1), (-1, 1)]}
        )
        # print("Clear shader and batch created successfully")

    def __update_matrices(self, context):
        """Update view and projection matrices"""
        camera = context.scene.camera
        if camera:
            depsgraph = context.evaluated_depsgraph_get()
            self.view_matrix = camera.matrix_world.inverted()
            self.projection_matrix = camera.calc_matrix_camera(
                depsgraph=depsgraph,
                x=self.render_width,
                y=self.render_height,
                scale_x=1.0,
                scale_y=1.0
            )
            # print("Matrix update successful")
        else:
            self.report({'ERROR'}, "No camera found in scene!")
            # print("No camera found in scene!")

    def render_quilt(self, context):
        """Render and combine textures to final_offscreen"""
        self.__update_matrices(context)
        context.area.tag_redraw()

        # Bind final_offscreen for combining
        with self.final_offscreen.bind():
            # Get current projection matrix
            viewMatrix = gpu.matrix.get_model_view_matrix()
            projectionMatrix = gpu.matrix.get_projection_matrix()

            fov = 2 * math.atan(1 / self.projection_matrix[1][1])
            f = 1 / math.tan(fov / 2)
            near = (self.projection_matrix[2][3] / (self.projection_matrix[2][2] - 1))
            # camera_size = f * math.tan(fov / 2)

            with gpu.matrix.push_pop():
                # Reset matrix -> Use standard device coordinates [-1, 1]
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                # Set blend and depth states
                gpu.state.depth_test_set('GREATER_EQUAL')
                gpu.state.depth_mask_set(True)
                gpu.state.blend_set('ALPHA')

                # Clear final_offscreen color buffer
                self.clear_shader.bind()
                self.clear_shader.uniform_float("color", (0.0, 0.0, 0.0, 1.0))  # Set clear color to black
                self.clear_batch.draw(self.clear_shader)

                start_time = time.time()

                for idx in range(self.grid_rows * self.grid_cols):
                    row = idx // self.grid_cols
                    col = idx % self.grid_cols
                    row = 4 - row
                    x_offset = col * self.render_width
                    y_offset = row * self.render_height

                    # Calculate center position in NDC coordinates
                    center_x = (x_offset + self.render_width / 2) / self.final_width * 2 - 1
                    center_y = (y_offset + self.render_height / 2) / self.final_height * 2 - 1
                    cameraDistance = context.scene.focal_plane
                    # print(cameraDistance)
                    cameraSize = cameraDistance * math.tan(fov / 2)
                    offsetAngle = (0.5 - idx / (40 - 1)) * math.radians(40)
                    # offset = - f * math.tan(offsetAngle)
                    offset = cameraDistance * offsetAngle
                    # Calculate new view matrix
                    # direction = self.view_matrix.col[2].xyz.normalized()
                    # new_offset = direction * offset
                    new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix
                    # new_view_matrix = self.view_matrix.copy()
                    # Calculate new projection matrix
                    new_projection_matrix = self.projection_matrix.copy()
                    new_projection_matrix[0][2] += offset / (cameraSize * (1440 / 2560))

                    near = context.scene.clip_near
                    far = context.scene.clip_far
                    # print(near)
                    # print(far)
                    clip_1 = -(far+near)/(far-near)
                    clip_2 = -(2*far*near)/(far-near)
                    new_projection_matrix[2][2] = clip_1
                    new_projection_matrix[2][3] = clip_2

                    # print(f"fov={fov}, f={f}, near={near},clip1={clip_1},clip2={clip_2} offsetAngle={offsetAngle}, offset={offset}")
                    # print(f"Texture {idx + 1}, viewMatrix={new_view_matrix},projectionMatrix={new_projection_matrix}")
                    # print(f"Rendering texture {idx + 1}, position: ({x_offset}, {y_offset})")

                    # Render to single offscreen
                    with self.offscreen.bind():
                        self.offscreen.draw_view3d(
                            scene=context.scene,
                            view_layer=context.view_layer,
                            view3d=context.space_data,
                            region=context.region,
                            view_matrix=new_view_matrix,
                            projection_matrix=new_projection_matrix
                        )

                    # Draw single texture to specified position in final_offscreen
                    self.shader.bind()
                    self.shader.uniform_sampler("image", self.offscreen.texture_color)
                    self.shader.uniform_float("scale", (
                        self.render_width / self.final_width, self.render_height / self.final_height))
                    self.shader.uniform_float("offset", (center_x, center_y))
                    self.batch.draw(self.shader)
                    gpu.shader.unbind()

                # Reset blend mode and depth states
                gpu.state.blend_set('NONE')
                gpu.state.depth_mask_set(False)
                gpu.state.depth_test_set('NONE')

                # Reload original matrices
                gpu.matrix.load_matrix(viewMatrix)
                gpu.matrix.load_projection_matrix(projectionMatrix)

                end_time = time.time()
                # print(f"Rendering and combining {self.grid_rows * self.grid_cols} textures took: {end_time - start_time:.6f} seconds")

    def save(self,context):
        """Draw combined texture in viewport"""
        if self.display_offscreen:
            # Set viewport drawing area
            draw_x = 0
            draw_y = 0
            draw_width = 1440  # Adjust display size as needed
            draw_height = 2560  # Fixed height

            # Draw final_offscreen texture to display_offscreen
            with self.display_offscreen.bind():
                if self.display_shader:
                    try:
                        self.display_shader.bind()
                        self.display_shader.uniform_sampler("image1", self.final_offscreen.texture_color)
                        self.display_shader.uniform_float("_Slope", obliquity)  # Correct uniform setting
                        self.display_shader.uniform_float("_Interval", linenumber)
                        self.display_shader.uniform_float("_X0", deviation)
                        self.display_shader.uniform_float("_ImgsCountX", 8.0)
                        self.display_shader.uniform_float("_ImgsCountY", 5.0)
                        self.display_shader.uniform_float("_ImgsCountAll", 40.0)
                        self.display_shader.uniform_float("_OutputSizeX", 1440.0)
                        self.display_shader.uniform_float("_OutputSizeY", 2560.0)
                        self.display_batch.draw(self.display_shader)
                        gpu.shader.unbind()
                    except Exception as e:
                        print(f"Error drawing display_shader: {e}")
                else:
                    print("Display_shader not initialized, cannot bind.")

            # Assumed parameters
            np.set_printoptions(threshold=np.inf)
            s = time.time()
            width, height, channels = 1440, 2560, 4
            padding = 0  # 16 bytes padding per row
            stride_row = width * channels + padding  # Actual bytes per row

            # Create simulated non-contiguous buffer
            pixel_data = self.display_offscreen.texture_color.read()

            # Convert non-contiguous buffer to numpy.array
            buffer_array = np.array(pixel_data, dtype=np.uint8)

            # Build logical view using strides
            logical_view = np.lib.stride_tricks.as_strided(
                buffer_array,
                shape=(height, width, channels),
                strides=(stride_row, channels, 1)
            )

            # View logical view
            # print(logical_view.shape)
            image_data = np.flipud(logical_view)
            rgb_data = image_data[:, :, :3]
            # Save image
            image = Image.fromarray(rgb_data, 'RGB')
            output_path = context.scene.my_filepath
            print(output_path)
            output_path += "LFD_preview_result"
            if not output_path.lower().endswith(".png"):
                output_path += ".png"
            image.save(output_path)

            e = time.time()
            # print(f"time is {e - s}")
            self.report({'INFO'}, f"Picture saved successfully, path is {output_path}")

    # Method called after clicking
    def execute(self, context):
        
        operator_id = 1
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        # Check offscreen rendering
        if not self.setup_offscreen_rendering():
            return {'CANCELLED'}

        # Check shader setup
        if not self.setup_shader():
            return {'CANCELLED'}

        # Setup shader for clearing buffer
        self.setup_clear_shader()

        # Setup display_offscreen shader for showing final_offscreen
        if not self.setup_display_shader():
            return {'CANCELLED'}

        # Render and combine textures
        self.render_quilt(context)

        self.save(context)

        return {'FINISHED'}

    # Execute this method before execute for initialization
    def invoke(self, context, event):
        return self.execute(context)


class QuiltSaveOperator(bpy.types.Operator):
    """Save the preview quilt picture"""
    bl_idname = "object.quilt_save"
    bl_label = "Save Quilt Preview Picture"
    bl_options = {'REGISTER', 'UNDO'}
    _handle = None

    # Prerequisites for operation
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    # Initialize data
    def __init__(self):
        self.offscreen = None  # For storing single camera texture
        self.final_offscreen = None  # For storing combined large texture
        self.display_offscreen = None  # Offscreen buffer for displaying texture
        self.shader = None
        self.clear_shader = None
        self.batch = None
        self.clear_batch = None
        self.display_batch = None  # Added for display_shader
        self.view_matrix = None
        self.projection_matrix = None
        self.render_width = 540  # Width of each texture
        self.render_height = 960  # Height of each texture
        self.grid_rows = 5
        self.grid_cols = 8
        self.final_width = self.render_width * self.grid_cols
        self.final_height = self.render_height * self.grid_rows
        self.display_shader = None  # For displaying final texture

    def setup_offscreen_rendering(self):
        """Setup small and large texture offscreens"""
        try:
            # Single texture offscreen
            self.offscreen = gpu.types.GPUOffScreen(self.render_width, self.render_height)
            # print(f"Single Offscreen created successfully: {self.render_width}x{self.render_height}")

            # Final combined texture offscreen
            self.final_offscreen = gpu.types.GPUOffScreen(self.final_width, self.final_height)
            # print(f"Final combined Offscreen created successfully: {self.final_width}x{self.final_height}")

            # Offscreen for displaying texture (display texture)
            self.display_offscreen = gpu.types.GPUOffScreen(1440, 2560)
            # print(f"Display interlaced OffScreen created successfully")

            return True
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create offscreen buffer: {e}")
            print(f"Failed to create offscreen buffer: {e}")
            return False

    def setup_shader(self):
        """Create shader for drawing textures"""
        vertex_shader = '''
            uniform vec2 scale;
            uniform vec2 offset;
            in vec2 pos;
            in vec2 texCoord;
            out vec2 fragTexCoord;

            void main()
            {
                gl_Position = vec4(pos * scale + offset, 0.0, 1.0);
                fragTexCoord = texCoord;
            }
        '''

        fragment_shader = '''
            uniform sampler2D image;
            in vec2 fragTexCoord;
            out vec4 FragColor;
            
            vec3 linear_to_srgb(vec3 linear) {
                bvec3 cutoff = lessThan(linear, vec3(0.0031308));
                vec3 higher = vec3(1.055) * pow(linear, vec3(1.0 / 2.4)) - vec3(0.055);
                vec3 lower = linear * vec3(12.92);
                return mix(higher, lower, cutoff);
            }
            
            void main()
            {
                vec4 color = texture(image, fragTexCoord);
                FragColor = vec4(linear_to_srgb(color.rgb), color.a);
            }
        '''

        try:
            self.shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
            # print("Shader compilation successful")
        except Exception as e:
            self.report({'ERROR'}, f"Shader compilation failed: {e}")
            print(f"Shader compilation failed: {e}")
            return False

        # Define vertices and indices
        vertices = [
            (-1, -1, 0, 0),  # Bottom-left
            (1, -1, 1, 0),  # Bottom-right
            (1, 1, 1, 1),  # Top-right
            (-1, 1, 0, 1)  # Top-left
        ]

        indices = [(0, 1, 2), (2, 3, 0)]

        self.batch = batch_for_shader(
            self.shader, 'TRIS',
            {"pos": [v[:2] for v in vertices], "texCoord": [v[2:] for v in vertices]},
            indices=indices
        )

        return True

    def setup_display_shader(self):
        """Create dedicated shader for display_offscreen"""
        vertex_shader = '''
            in vec2 pos;
            in vec2 texCoord;
            out vec2 fragTexCoord;

            void main()
            {
                gl_Position = vec4(pos, 0.0, 1.0);
                fragTexCoord = texCoord;
            }
        '''

        fragment_shader = '''
            uniform sampler2D image1;
            uniform float _OutputSizeX;
            uniform float _OutputSizeY;
            uniform float _Slope;
            uniform float _X0;
            uniform float _Interval;
            uniform float _ImgsCountAll;
            uniform float _ImgsCountX;
            uniform float _ImgsCountY;
            in vec2 fragTexCoord;
            out vec4 FragColor;

            float get_choice_float(vec2 pos, float bias) {
                float x = pos.x * _OutputSizeX + 0.5;
                float y = (1- pos.y) * _OutputSizeY + 0.5;
                // float y = pos.y * _OutputSizeY + 0.5;
                float x1 = (x + y * _Slope) * 3.0 + bias;
                float x_local = mod(x1 + _X0, _Interval);
                return (x_local / _Interval);
            }

            vec3 linear_to_srgb(vec3 linear) {
                bvec3 cutoff = lessThan(linear, vec3(0.0031308));
                vec3 higher = vec3(1.055) * pow(linear, vec3(1.0 / 2.4)) - vec3(0.055);
                vec3 lower = linear * vec3(12.92);
                return mix(higher, lower, cutoff);
            }

            vec2 get_uv_from_choice(vec2 pos, float choice_float) {
                float choice = floor(choice_float * _ImgsCountAll);
                vec2 choice_vec = vec2(
                _ImgsCountX - 1.0 - mod(choice, _ImgsCountX),  // Right to left
                // _ImgsCountY - 1.0 - floor(choice / _ImgsCountX) 
                floor(choice / _ImgsCountX) // Bottom to top
                );

                vec2 reciprocals = vec2(1.0 / _ImgsCountX, 1.0 / _ImgsCountY);
                return (choice_vec + pos) * reciprocals;
            }

            vec4 get_color(vec2 pos, float bias) {
                float choice_float = get_choice_float(pos, bias);
                vec2 sel_pos = get_uv_from_choice(pos, choice_float);
                return texture(image1, sel_pos);
            }

            void main() {
                vec4 color = get_color(fragTexCoord, 0.0);
                color.g = get_color(fragTexCoord, 1.0).g;
                color.b = get_color(fragTexCoord, 2.0).b;
                FragColor = vec4(linear_to_srgb(color.rgb), color.a);
            }
        '''

        try:
            self.display_shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
            # print("Display_offscreen shader compilation successful")
        except gpu.types.GPUShaderCompilationError as e:
            self.report({'ERROR'}, f"Display_offscreen shader compilation failed: {e}")
            print(f"Display_offscreen shader compilation failed: {e}")
            self.display_shader = None
            return False
        except Exception as e:
            self.report({'ERROR'}, f"Display_offscreen shader initialization failed: {e}")
            print(f"Display_offscreen shader initialization failed: {e}")
            self.display_shader = None
            return False

        # 定义顶点和索引，用于绘制显示纹理
        vertices = [
            (-1, -1, 0, 0),  # Bottom-left
            (1, -1, 1, 0),  # Bottom-right
            (1, 1, 1, 1),  # Top-right
            (-1, 1, 0, 1)  # Top-left
        ]

        indices = [(0, 1, 2), (2, 3, 0)]

        self.display_batch = batch_for_shader(
            self.display_shader, 'TRIS',
            {"pos": [v[:2] for v in vertices], "texCoord": [v[2:] for v in vertices]},
            indices=indices
        )

        return True
    def setup_clear_shader(self):
        """Create a simple shader for clearing the color buffer"""
        # Use built-in 'UNIFORM_COLOR' shader
        self.clear_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.clear_batch = batch_for_shader(
            self.clear_shader, 'TRI_FAN',
            {"pos": [(-1, -1), (1, -1), (1, 1), (-1, 1)]}
        )
        # print("Clear shader and batch created successfully")

    def __update_matrices(self, context):
        """Update view and projection matrices"""
        camera = context.scene.camera
        if camera:
            depsgraph = context.evaluated_depsgraph_get()
            self.view_matrix = camera.matrix_world.inverted()
            self.projection_matrix = camera.calc_matrix_camera(
                depsgraph=depsgraph,
                x=self.render_width,
                y=self.render_height,
                scale_x=1.0,
                scale_y=1.0
            )
            # print("Matrix update successful")
        else:
            self.report({'ERROR'}, "No camera found in scene!")
            # print("No camera found in scene!")

    def render_quilt(self, context):
        """Render and combine textures to final_offscreen"""
        self.__update_matrices(context)
        context.area.tag_redraw()

        # Bind final_offscreen for combining
        with self.final_offscreen.bind():
            # Get current projection matrix
            viewMatrix = gpu.matrix.get_model_view_matrix()
            projectionMatrix = gpu.matrix.get_projection_matrix()

            fov = 2 * math.atan(1 / self.projection_matrix[1][1])
            f = 1 / math.tan(fov / 2)
            near = (self.projection_matrix[2][3] / (self.projection_matrix[2][2] - 1))
            # camera_size = f * math.tan(fov / 2)

            with gpu.matrix.push_pop():
                # Reset matrix -> Use standard device coordinates [-1, 1]
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                # Set blend and depth states
                gpu.state.depth_test_set('GREATER_EQUAL')
                gpu.state.depth_mask_set(True)
                gpu.state.blend_set('ALPHA')

                # Clear final_offscreen color buffer
                self.clear_shader.bind()
                self.clear_shader.uniform_float("color", (0.0, 0.0, 0.0, 1.0))  # Set clear color to black
                self.clear_batch.draw(self.clear_shader)

                start_time = time.time()

                for idx in range(self.grid_rows * self.grid_cols):
                    row = idx // self.grid_cols
                    col = idx % self.grid_cols
                    row = 4 - row
                    x_offset = col * self.render_width
                    y_offset = row * self.render_height

                    # Calculate center position in NDC coordinates
                    center_x = (x_offset + self.render_width / 2) / self.final_width * 2 - 1
                    center_y = (y_offset + self.render_height / 2) / self.final_height * 2 - 1
                    cameraDistance = context.scene.focal_plane
                    # print(cameraDistance)
                    cameraSize = cameraDistance * math.tan(fov / 2)
                    offsetAngle = (0.5 - idx / (40 - 1)) * math.radians(40)
                    # offset = - f * math.tan(offsetAngle)
                    offset = cameraDistance * offsetAngle
                    # Calculate new view matrix
                    # direction = self.view_matrix.col[2].xyz.normalized()
                    # new_offset = direction * offset
                    new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix
                    # new_view_matrix = self.view_matrix.copy()
                    # Calculate new projection matrix
                    new_projection_matrix = self.projection_matrix.copy()
                    new_projection_matrix[0][2] += offset / (cameraSize * (1440 / 2560))

                    near = context.scene.clip_near
                    far = context.scene.clip_far
                    # print(near)
                    # print(far)
                    clip_1 = -(far+near)/(far-near)
                    clip_2 = -(2*far*near)/(far-near)
                    new_projection_matrix[2][2] = clip_1
                    new_projection_matrix[2][3] = clip_2

                    # print(f"fov={fov}, f={f}, near={near},clip1={clip_1},clip2={clip_2} offsetAngle={offsetAngle}, offset={offset}")
                    # print(f"Texture {idx + 1}, viewMatrix={new_view_matrix}, projectionMatrix={new_projection_matrix}")
                    # print(f"Rendering texture {idx + 1}, position: ({x_offset}, {y_offset})")

                    # Render to single offscreen
                    with self.offscreen.bind():
                        self.offscreen.draw_view3d(
                            scene=context.scene,
                            view_layer=context.view_layer,
                            view3d=context.space_data,
                            region=context.region,
                            view_matrix=new_view_matrix,
                            projection_matrix=new_projection_matrix
                        )

                    # Draw single texture to specified position in final_offscreen
                    self.shader.bind()
                    self.shader.uniform_sampler("image", self.offscreen.texture_color)
                    self.shader.uniform_float("scale", (
                        self.render_width / self.final_width, self.render_height / self.final_height))
                    self.shader.uniform_float("offset", (center_x, center_y))
                    self.batch.draw(self.shader)
                    gpu.shader.unbind()

                # Reset blend mode and depth states
                gpu.state.blend_set('NONE')
                gpu.state.depth_mask_set(False)
                gpu.state.depth_test_set('NONE')

                # Reload original matrices
                gpu.matrix.load_matrix(viewMatrix)
                gpu.matrix.load_projection_matrix(projectionMatrix)

                end_time = time.time()
                # print(f"Rendering and combining {self.grid_rows * self.grid_cols} textures took: {end_time - start_time:.6f} seconds")

    def save(self,context):
        """Draw combined texture in viewport"""
        # Convert non-contiguous buffer to numpy.array
        width, height, channels = 4320, 4800, 4
        padding = 0  # 16 bytes padding per row
        stride_row = width * channels + padding  # Actual bytes per row

        # Create simulated non-contiguous buffer
        pixel_data = self.final_offscreen.texture_color.read()

        # Convert non-contiguous buffer to numpy.array
        buffer_array = np.array(pixel_data, dtype=np.uint8)

        # Build logical view using strides
        logical_view = np.lib.stride_tricks.as_strided(
            buffer_array,
            shape=(height, width, channels),
            strides=(stride_row, channels, 1)
        )

        # View logical view
        # print(logical_view.shape)
        image_data = np.flipud(logical_view)
        rgb_data = image_data[:, :, :3]
        image = Image.fromarray(rgb_data, 'RGB')
        output_path = context.scene.my_filepath
        print(output_path)
        output_path += "quilt_preview_result"
        if not output_path.lower().endswith(".png"):
            output_path += ".png"
        image.save(output_path)

        # e = time.time()
        # # print(f"time is {e - s}")
        self.report({'INFO'}, f"Picture saved successfully, path is {output_path}")

    def execute(self, context):
        # Check offscreen rendering
        
        operator_id = 1
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")

        if not self.setup_offscreen_rendering():
            return {'CANCELLED'}

        # Check shader setup
        if not self.setup_shader():
            return {'CANCELLED'}

        # Setup shader for clearing buffer
        self.setup_clear_shader()

        # Setup display_offscreen shader for showing final_offscreen
        if not self.setup_display_shader():
            return {'CANCELLED'}

        # Render and combine textures
        self.render_quilt(context)

        self.save(context)

        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(context)


class LFDPreviewOperator(bpy.types.Operator):
    """Start LightField Preview, Esc to exit"""
    bl_idname = "object.preview"
    bl_label = "Realtime LightField Preview"
    bl_options = {'REGISTER', 'UNDO'}

    _handle = None  # Handle for storing draw handler

    # display_x: IntProperty(
    #     name="x-axis of display",
    #     description="x axis of display",
    #     default=2560,
    # )

    # Prerequisites for operation
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    # Initialize data
    def __init__(self):
        self.offscreen = None  # For storing single camera texture
        self.final_offscreen = None  # For storing combined large texture
        self.display_offscreen = None  # Offscreen buffer for displaying texture
        self.shader = None
        self.clear_shader = None
        self.batch = None
        self.clear_batch = None
        self.display_batch = None  # Added for display_shader
        self.view_matrix = None
        self.projection_matrix = None
        self.render_width = 540  # Width of each texture
        self.render_height = 960  # Height of each texture
        self.grid_rows = 5
        self.grid_cols = 8
        self.final_width = self.render_width * self.grid_cols
        self.final_height = self.render_height * self.grid_rows
        self.display_shader = None  # For displaying final texture
        self.screen = None
        self.fps = 10

    def setup_offscreen_rendering(self):
        """Setup small and large texture offscreens"""
        try:
            # Single texture offscreen
            self.offscreen = gpu.types.GPUOffScreen(self.render_width, self.render_height)

            # Final combined texture offscreen
            self.final_offscreen = gpu.types.GPUOffScreen(self.final_width, self.final_height)

            # Offscreen for displaying texture
            self.display_offscreen = gpu.types.GPUOffScreen(1440, 2560)

            return True
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create offscreen buffer: {e}")
            print(f"Failed to create offscreen buffer: {e}")
            return False

    def setup_shader(self):
        """Create shader for drawing textures"""
        vertex_shader = '''
            uniform vec2 scale;
            uniform vec2 offset;
            in vec2 pos;
            in vec2 texCoord;
            out vec2 fragTexCoord;

            void main()
            {
                gl_Position = vec4(pos * scale + offset, 0.0, 1.0);
                fragTexCoord = texCoord;
            }
        '''

        fragment_shader = '''
            uniform sampler2D image;
            in vec2 fragTexCoord;
            out vec4 FragColor;

            void main()
            {
                FragColor = texture(image, fragTexCoord);
            }
        '''

        try:
            self.shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
        except Exception as e:
            self.report({'ERROR'}, f"Shader compilation failed: {e}")
            print(f"Shader compilation failed: {e}")
            return False

        # Define vertices and indices
        vertices = [
            (-1, -1, 0, 0),  # Bottom-left
            (1, -1, 1, 0),  # Bottom-right
            (1, 1, 1, 1),  # Top-right
            (-1, 1, 0, 1)  # Top-left
        ]

        indices = [(0, 1, 2), (2, 3, 0)]

        self.batch = batch_for_shader(
            self.shader, 'TRIS',
            {"pos": [v[:2] for v in vertices], "texCoord": [v[2:] for v in vertices]},
            indices=indices
        )

        return True

    def setup_display_shader(self):
        """Create dedicated shader for display_offscreen"""
        vertex_shader = '''
            in vec2 pos;
            in vec2 texCoord;
            out vec2 fragTexCoord;

            void main()
            {
                gl_Position = vec4(pos, 0.0, 1.0);
                fragTexCoord = texCoord;
            }
        '''

        fragment_shader = '''
            uniform sampler2D image1;
            uniform float _OutputSizeX;
            uniform float _OutputSizeY;
            uniform float _Slope;
            uniform float _X0;
            uniform float _Interval;
            uniform float _ImgsCountAll;
            uniform float _ImgsCountX;
            uniform float _ImgsCountY;
            in vec2 fragTexCoord;
            out vec4 FragColor;

            float get_choice_float(vec2 pos, float bias) {
                float x = pos.x * _OutputSizeX + 0.5;
                float y = (1- pos.y) * _OutputSizeY + 0.5;
                // float y = pos.y * _OutputSizeY + 0.5;
                float x1 = (x + y * _Slope) * 3.0 + bias;
                float x_local = mod(x1 + _X0, _Interval);
                return (x_local / _Interval);
            }

            vec3 linear_to_srgb(vec3 linear) {
                bvec3 cutoff = lessThan(linear, vec3(0.0031308));
                vec3 higher = vec3(1.055) * pow(linear, vec3(1.0 / 2.4)) - vec3(0.055);
                vec3 lower = linear * vec3(12.92);
                return mix(higher, lower, cutoff);
            }

            vec2 get_uv_from_choice(vec2 pos, float choice_float) {
                float choice = floor(choice_float * _ImgsCountAll);
                vec2 choice_vec = vec2(
                _ImgsCountX - 1.0 - mod(choice, _ImgsCountX),  // 从右到左
                // _ImgsCountY - 1.0 - floor(choice / _ImgsCountX) 
                floor(choice / _ImgsCountX) // 从下到上
                );

                vec2 reciprocals = vec2(1.0 / _ImgsCountX, 1.0 / _ImgsCountY);
                return (choice_vec + pos) * reciprocals;
            }

            vec4 get_color(vec2 pos, float bias) {
                float choice_float = get_choice_float(pos, bias);
                vec2 sel_pos = get_uv_from_choice(pos, choice_float);
                return texture(image1, sel_pos);
            }

            void main() {
                vec4 color = get_color(fragTexCoord, 0.0);
                color.g = get_color(fragTexCoord, 1.0).g;
                color.b = get_color(fragTexCoord, 2.0).b;
                FragColor = vec4(linear_to_srgb(color.rgb), color.a); 
            }
        '''

        try:
            self.display_shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
        except gpu.types.GPUShaderCompilationError as e:
            self.report({'ERROR'}, f"Display offscreen shader compilation failed: {e}")
            print(f"Display offscreen shader compilation failed: {e}")
            self.display_shader = None
            return False
        except Exception as e:
            self.report({'ERROR'}, f"Display offscreen shader initialization failed: {e}")
            print(f"Display offscreen shader initialization failed: {e}")
            self.display_shader = None
            return False

        # Define vertices and indices for displaying texture
        vertices = [
            (-1, -1, 0, 0),  # Bottom-left
            (1, -1, 1, 0),  # Bottom-right
            (1, 1, 1, 1),  # Top-right
            (-1, 1, 0, 1)  # Top-left
        ]

        indices = [(0, 1, 2), (2, 3, 0)]

        self.display_batch = batch_for_shader(
            self.display_shader, 'TRIS',
            {"pos": [v[:2] for v in vertices], "texCoord": [v[2:] for v in vertices]},
            indices=indices
        )

        return True

    def setup_clear_shader(self):
        """Create simple shader for clearing color buffer"""
        # Use built-in 'UNIFORM_COLOR' shader
        self.clear_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.clear_batch = batch_for_shader(
            self.clear_shader, 'TRI_FAN',
            {"pos": [(-1, -1), (1, -1), (1, 1), (-1, 1)]}
        )

    def __update_matrices(self, context):
        """Update view and projection matrices"""
        camera = context.scene.camera
        if camera:
            depsgraph = context.evaluated_depsgraph_get()
            self.view_matrix = camera.matrix_world.inverted()
            self.projection_matrix = camera.calc_matrix_camera(
                depsgraph=depsgraph,
                x=self.render_width,
                y=self.render_height,
                scale_x=1.0,
                scale_y=1.0
            )
        else:
            self.report({'ERROR'}, "No camera found in scene!")
            print("No camera found in scene!")

    def render_quilt(self, context):
        """Render and combine textures to final_offscreen"""
        self.__update_matrices(context)
        context.area.tag_redraw()

        # Bind final_offscreen for texture combination
        with self.final_offscreen.bind():
            # Get current projection matrix
            viewMatrix = gpu.matrix.get_model_view_matrix()
            projectionMatrix = gpu.matrix.get_projection_matrix()

            fov = 2 * math.atan(1 / self.projection_matrix[1][1])
            f = 1 / math.tan(fov / 2)
            # near = (self.projection_matrix[2][3] / (self.projection_matrix[2][2] - 1))
            # camera_size = f * math.tan(fov / 2)

            with gpu.matrix.push_pop():
                # Reset matrix -> Use standard device coordinates [-1, 1]
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                # Set blend and depth states
                gpu.state.depth_test_set('GREATER_EQUAL')
                gpu.state.depth_mask_set(True)
                gpu.state.blend_set('ALPHA')

                # Clear color buffer of final_offscreen
                self.clear_shader.bind()
                self.clear_shader.uniform_float("color", (0.0, 0.0, 0.0, 1.0))  # Set clear color to black
                self.clear_batch.draw(self.clear_shader)

                start_time = time.time()

                for idx in range(self.grid_rows * self.grid_cols):
                    row = idx // self.grid_cols
                    col = idx % self.grid_cols
                    row = 4 - row
                    x_offset = col * self.render_width
                    y_offset = row * self.render_height

                    # Calculate center position in NDC coordinates
                    center_x = (x_offset + self.render_width / 2) / self.final_width * 2 - 1
                    center_y = (y_offset + self.render_height / 2) / self.final_height * 2 - 1
                    cameraDistance = context.scene.focal_plane
                    # print(cameraDistance)
                    cameraSize = cameraDistance * math.tan(fov / 2)
                    offsetAngle = (0.5 - idx / (40 - 1)) * math.radians(40)
                    # offset = - f * math.tan(offsetAngle)
                    offset = cameraDistance * math.tan(offsetAngle)
                    # Calculate new view matrix
                    # direction = self.view_matrix.col[2].xyz.normalized()
                    # new_offset = direction * offset
                    new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix
                    # new_view_matrix = self.view_matrix.copy()
                    # Calculate new projection matrix
                    new_projection_matrix = self.projection_matrix.copy()
                    new_projection_matrix[0][2] += offset / (cameraSize * (1440 / 2560))

                    near = context.scene.clip_near
                    far = context.scene.clip_far
                    # print(near)
                    # print(far)
                    clip_1 = -(far+near)/(far-near)
                    clip_2 = -(2*far*near)/(far-near)
                    new_projection_matrix[2][2] = clip_1
                    new_projection_matrix[2][3] = clip_2

                    # print(f"fov={fov}, f={f}, near={near},clip1={clip_1},clip2={clip_2} offsetAngle={offsetAngle}, offset={offset}")
                    # print(f"Texture {idx + 1}, viewMatrix={new_view_matrix},projectionMatrix={new_projection_matrix}")
                    # print(f"Rendering texture {idx + 1}, position: ({x_offset}, {y_offset})")

                    # Render to single offscreen
                    with self.offscreen.bind():
                        self.offscreen.draw_view3d(
                            scene=context.scene,
                            view_layer=context.view_layer,
                            view3d=context.space_data,
                            region=context.region,
                            view_matrix=new_view_matrix,
                            projection_matrix=new_projection_matrix
                        )

                    # Draw single texture to specified position in final_offscreen
                    self.shader.bind()
                    self.shader.uniform_sampler("image", self.offscreen.texture_color)
                    self.shader.uniform_float("scale", (
                        self.render_width / self.final_width, self.render_height / self.final_height))
                    self.shader.uniform_float("offset", (center_x, center_y))
                    self.batch.draw(self.shader)
                    gpu.shader.unbind()

                # Reset blend mode and depth states
                gpu.state.blend_set('NONE')
                gpu.state.depth_mask_set(False)
                gpu.state.depth_test_set('NONE')

                # Reload original matrices
                gpu.matrix.load_matrix(viewMatrix)
                gpu.matrix.load_projection_matrix(projectionMatrix)

                end_time = time.time()
                # print(f"Rendering and combining {self.grid_rows * self.grid_cols} textures took: {end_time - start_time:.6f} seconds")

    def draw_callback_px(self, context, region):
        """Draw combined texture in viewport"""
        if self.display_offscreen:
            # Set viewport drawing area
            draw_x = 0
            draw_y = 0
            draw_width = 1440  # Adjust display size as needed
            draw_height = 2560  # Fixed height

            # Draw final_offscreen texture to display_offscreen
            with self.display_offscreen.bind():
                if self.display_shader:
                    try:
                        self.display_shader.bind()
                        self.display_shader.uniform_sampler("image1", self.final_offscreen.texture_color)
                        self.display_shader.uniform_float("_Slope", obliquity)  # Correct uniform setting
                        self.display_shader.uniform_float("_Interval", linenumber)
                        self.display_shader.uniform_float("_X0", deviation)
                        self.display_shader.uniform_float("_ImgsCountX", 8.0)
                        self.display_shader.uniform_float("_ImgsCountY", 5.0)
                        self.display_shader.uniform_float("_ImgsCountAll", 40.0)
                        self.display_shader.uniform_float("_OutputSizeX", 1440.0)
                        self.display_shader.uniform_float("_OutputSizeY", 2560.0)
                        self.display_batch.draw(self.display_shader)
                        gpu.shader.unbind()
                    except Exception as e:
                        print(f"Error drawing display_shader: {e}")
                else:
                    print("display_shader not initialized, cannot bind.")

            # Assumed parameters
            np.set_printoptions(threshold=np.inf)
            s = time.time()
            width, height, channels = 1440, 2560, 4
            padding = 0  # 16 bytes padding per row
            stride_row = width * channels + padding  # Actual bytes per row

            # Create simulated non-contiguous buffer
            pixel_data = self.display_offscreen.texture_color.read()

            # Convert non-contiguous buffer to numpy.array
            buffer_array = np.array(pixel_data, dtype=np.uint8)

            # Build logical view using strides
            logical_view = np.lib.stride_tricks.as_strided(
                buffer_array,
                shape=(height, width, channels),
                strides=(stride_row, channels, 1)
            )

            # View logical view
            # print(logical_view.shape)

            # draw_texture_2d(self.final_offscreen.texture_color, (0,0), 540, 960)

            image_data = np.flipud(logical_view)
            rgb_data = image_data[:, :, :3]
            # update_pygame_window(self.screen, rgb_data)
            update_cv_window(window_name, rgb_data)

            # Image saving
            # image = Image.fromarray(rgb_data, 'RGB')
            # image.save("D:/desktop/debug_image.png")
            e = time.time()
            # print(f"time is {e - s}")

    # 点击后调用该方法
    def execute(self, context):
        global flag
        flag = True
        # self.screen = initialize_pygame_window(self.display_x)
        global temp_x
        # monitors = get_monitors()
        x = context.scene.x_axis
        initialize_cv_window()
        
        operator_id = 1
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        # Check offscreen rendering
        if not self.setup_offscreen_rendering():
            return {'CANCELLED'}

        # Check shader setup
        if not self.setup_shader():
            return {'CANCELLED'}

        # Setup shader for clearing buffer
        self.setup_clear_shader()

        # Setup display_offscreen shader for showing final_offscreen
        if not self.setup_display_shader():
            return {'CANCELLED'}

        # Render and combine textures
        self.render_quilt(context)

        # Add draw callback
        args = (self, context)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback_px, args, 'WINDOW', 'POST_PIXEL'
        )

        # Add timer
        self._timer = context.window_manager.event_timer_add((1 / self.fps), window=context.window)  # Refresh every 0.1 seconds

        # Add modal handler
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    #
    def modal(self, context, event):
        if event.type == 'TIMER':  # Update rendering when timer triggers
            self.render_quilt(context)
            context.area.tag_redraw()

        if event.type in {'ESC'}:  # Exit when ESC key is pressed
            global flag
            flag = False
            self.cancel(context)
            cv2.destroyAllWindows()
            # pygame.quit()
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def cancel(self, context):
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        if self.offscreen:
            self.offscreen.free()
            self.offscreen = None

        if self.final_offscreen:
            self.final_offscreen.free()
            self.final_offscreen = None

        if self.display_offscreen:
            self.display_offscreen.free()
            self.display_offscreen = None

        context.area.tag_redraw()





class LFDRenderOperator(bpy.types.Operator):
    """Save LFD Render image"""
    bl_idname = "object.ldf_render"
    bl_label = "Save LFD Render Picture"
    bl_options = {'REGISTER', 'UNDO'}

    # 执行操作的前提
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False


class QuiltRenderOperator(bpy.types.Operator):
    """Save Multiview Render image"""
    bl_idname = "object.quilt_render"
    bl_label = "Save Multiview Render Picture"
    bl_options = {'REGISTER', 'UNDO'}

    # Properties for tracking render progress and parameters
    index: bpy.props.IntProperty(default=0)
    camera = None
    original_path = ""
    original_shift_x = 0.0
    original_clip_start = 0.0
    original_clip_end = 0.0
    original_focus_distance = 0.0
    view_matrix = None
    cameraSize = 0.0
    focal_plane = 0.0
    _timer = None

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    def execute(self, context):
        
        operator_id = 2
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        # Save original parameters
        self.camera = context.scene.camera
        self.original_path = context.scene.my_filepath
        self.original_shift_x = self.camera.data.shift_x
        self.original_clip_start = self.camera.data.clip_start
        self.original_clip_end = self.camera.data.clip_end
        self.original_focus_distance = self.camera.data.dof.focus_distance

        # Set initial camera parameters
        self.camera.data.type = 'PERSP'
        self.camera.data.dof.use_dof = True
        self.camera.data.clip_start = context.scene.clip_near
        self.camera.data.clip_end = context.scene.clip_far
        self.camera.data.dof.focus_distance = context.scene.focal_plane

        # Calculate necessary matrices
        depsgraph = context.evaluated_depsgraph_get()
        self.view_matrix = self.camera.matrix_world.inverted()
        projection_matrix = self.camera.calc_matrix_camera(
            depsgraph=depsgraph,
            x=540,
            y=960,
            scale_x=1.0,
            scale_y=1.0
        )
        fov = 2 * math.atan(1 / projection_matrix[1][1])
        self.cameraSize = context.scene.focal_plane * math.tan(fov / 2)
        self.focal_plane = context.scene.focal_plane

        # Initialize progress bar and modal loop
        self.index = 0
        context.window_manager.progress_begin(0, 40)
        self._timer = context.window_manager.event_timer_add(0.0001, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        """Restore original settings and end rendering"""
        self.camera.data.clip_start = self.original_clip_start
        self.camera.data.clip_end = self.original_clip_end
        self.camera.data.dof.focus_distance = self.original_focus_distance
        self.camera.data.shift_x = self.original_shift_x
        self.camera.matrix_world = self.view_matrix.inverted()

        context.scene.render.filepath = self.original_path
        context.window_manager.progress_end()

    def modal(self, context, event):
        if event.type == 'ESC':
            # Clean up original settings and end rendering
            self.cleanup(context)
            self.report({"INFO"}, "Render Cancelled")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}


        if self.index >= 40:
            # Rendering complete, clean up resources
            self.cleanup(context)
            self.report({'INFO'}, "All pictures saved successfully.")
            return {'FINISHED'}

        # Calculate current view parameters
        offsetangle = (0.5 - self.index / (40 - 1)) * math.radians(40)
        offset = self.focal_plane * offsetangle
        new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix

        # Update camera position and parameters
        self.camera.matrix_world = new_view_matrix.inverted()
        self.camera.data.shift_x = self.original_shift_x + 0.5 * offset / self.cameraSize

        # Set render path and execute rendering
        output_path = f"{self.original_path}_{self.index:03d}.png"
        context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        # Update progress and prepare next frame
        context.window_manager.progress_update(self.index)
        self.index += 1

        # Allow interface refresh
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}


class QuiltSaveOperator1(bpy.types.Operator):
    """Save Quilt image"""
    bl_idname = "object.quilt_1_save"
    bl_label = "Synthesize Quilt Picture"
    bl_options = {'REGISTER', 'UNDO'}

    # Prerequisites for rendering
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    def execute(self, context):
        
        operator_id = 2
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        original_path = context.scene.my_filepath
        original_path = os.path.normpath(original_path)
        print(original_path)
        single_width = 540
        single_height = 960
        rows = 5
        cols = 8
        width = single_width * cols
        height = single_height * rows
        new_image = Image.new('RGB', (width, height))

        for i in range(rows):
            for j in range(cols):
                image_filename = f"_{i*cols + j:03d}.png"
                image_path = os.path.join(original_path, image_filename)
                img = Image.open(image_path)
                #print(image_path)
                left = j * single_width
                upper = i * single_height
                right = left + single_width
                lower = upper + single_height

                new_image.paste(img,(left,upper,right,lower))

        output_path = os.path.join(original_path, "quilt_synthesize_result.png")
        new_image.save(output_path)
        self.report({'INFO'}, f"Quilt synthesize successfully, path is {output_path}")

        return {'FINISHED'}


message_ = "Upload failed"

def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
        return base64.b64encode(file_content).decode('utf-8')

def split_data(data, max_size):
    return [data[i:i + max_size] for i in range(0, len(data), max_size)]

def on_open_choice1(ws, data_chunks, current_chunk_index):
    print("Connected to server")
    send_next_chunk(ws, data_chunks, current_chunk_index)


def on_open_choice2(ws, operator_id, biz_type):

    data_type = "tracking"
    data_to_send = {
        "bizId": operator_id,
        "bizType": biz_type
    }

    final_data = {
        "type": data_type,
        "data": data_to_send
    }

    ws.send(json.dumps(final_data))
    ws.close()

from websocket import WebSocketTimeoutException

def send_next_chunk(ws, data_chunks, current_chunk_index):

    if ws.sock and ws.sock.connected:
        if current_chunk_index < len(data_chunks):
            print(f"Currently at chunk {current_chunk_index+1}, total {len(data_chunks)} chunks")
            chunk = data_chunks[current_chunk_index]

            data_split = {
                "type": "quilt",
                "data": chunk,
                "done": False
            }
            print(str(data_split)[:100])
            print(str(data_split)[-100:])

            ws.send(json.dumps(data_split))
            print(f"Sent chunk {current_chunk_index + 1}")
            current_chunk_index += 1

            time.sleep(0.1)  
            send_next_chunk(ws, data_chunks, current_chunk_index)
        else:
            done_message = {
                "type": "quilt",
                "data": None,
                "done": True
            }
            print("Send end flag")
            print(str(done_message)[:100])
            ws.send(json.dumps(done_message))
            print("Data transfer ended")
            ws.close() 
    else:
        ws.close()
        print("Connection closed")




def send_via_websockets(on_open_handler, **handler_kwargs):
    # OpenstageAI, Cubestage
    ws_urls = ["ws://127.0.0.1:9001", "ws://127.0.0.1:9003"]
    active_connections = []
    connection_timeout = 5
    
    def ws_worker(url):
        timeout_timer = threading.Timer(connection_timeout, lambda: ws.close() if ws.sock else None)
        
        def on_open(ws):
            nonlocal timeout_timer
            timeout_timer.cancel() 
            for conn in active_connections:
                if conn != ws:
                    conn.close()
            active_connections.append(ws)
            on_open_handler(ws, **handler_kwargs)
            
        def on_close(ws, close_status_code, close_msg):
            if ws in active_connections:
                active_connections.remove(ws)
            if 'timeout_timer' in locals():
                timeout_timer.cancel()
                
        def on_error(ws, error):
            print(f"{url}websocket connect error: {error}")
            ws.close()
            
        ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=None,
            on_close=on_close,
            on_error=on_error
        )
        
        timeout_timer.start() 
        ws.run_forever() 
    
    for url in ws_urls:
        threading.Thread(target=ws_worker, args=(url,), daemon=True).start()



def upload_file(self, file_path1 , file_path2, title):
    global message_
    message_ = "Upload failed"
    print(file_path1)
    if not os.path.isfile(file_path1):
        print("Preview image does not exist")
        self.report({'INFO'}, "Preview image does not exist, ensure that _020.png exists in the file path")
        return
    if not os.path.isfile(file_path2):
        print("Synthesized image does not exist")
        self.report({'INFO'}, "Synthesized image does not exist, ensure that quilt_synthesize_result.png exists in the file path")
        return

    # Get the MIME type of the file (only PNG type is supported here)
    mime_type = "image/png"

    # Convert file content to Base64 encoding
    base64_string_1 = file_to_base64(file_path1)
    base64_string_2 = file_to_base64(file_path2)

    # Create data object to send
    data_to_send = {
        "previewFile": f"data:{mime_type};base64,{base64_string_1}",
        "previewFileName": os.path.basename(file_path1),
        "previewFileType": mime_type.split("/")[1],
        "resourceFile": f"data:{mime_type};base64,{base64_string_2}",
        "resourceFileName": os.path.basename(file_path2),
        "resourceFileType": mime_type.split("/")[1],
        "title": title
    }

    json_data = json.dumps(data_to_send)

    data_chunks = split_data(json_data, 1024 * 1024)  # Split size is 1MB

    current_chunk_index = 0

    send_via_websockets(on_open_choice1, data_chunks=data_chunks, current_chunk_index=current_chunk_index)


class ConnectPlatform(bpy.types.Operator):
    """Upload to the platform"""
    bl_idname = "object.platform"
    bl_label = "Upload to the platform"
    bl_options = {'REGISTER', 'UNDO'}

    title: StringProperty(
        name="title of the picture",
        description="set the title of the picture",
        default="default",
    )
    # Prerequisites for rendering
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False


    def execute(self, context):
        
        operator_id = 3
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        print(self.title)
        fp = bpy.context.scene.my_filepath
        fp1 = os.path.join(fp,"_020.png")
        fp2 = os.path.join(fp,"quilt_synthesize_result.png")
        print(fp)
        print(fp1)
        print(fp2)
        upload_file(self, fp1, fp2 , self.title)
        return {'CANCELLED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class RenderAnimation(bpy.types.Operator):
    """Render animation"""
    bl_idname = "object.animation"
    bl_label = "Render Animation"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    def execute(self, context):
        
        operator_id = 4
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        # Get file path
        output_path = context.scene.my_filepath

        # Set output path, ensure directory exists
        if output_path:
            context.scene.render.filepath = output_path

            # Set render as static image, render entire animation
            bpy.ops.render.render(animation=True, write_still=True)

            print("Rendering animation to:", output_path)
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "Output path not set in scene.")
            return {'CANCELLED'}



class RenderImageSequenceToVideo(bpy.types.Operator):
    """Render Image Sequence to Video"""
    bl_idname = "object.render_image_sequence_to_video"
    bl_label = "Render Image Sequence to Video"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    def execute(self, context):
        
        operator_id = 4
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        input_folder = context.scene.my_filepath
        output_video_path = os.path.join(input_folder, "output.mp4")

        # Process existing image sequence...
        image_files = [f"quilt_frame_{i}.png" for i in range(context.scene.frame_s, context.scene.frame_e + 1)]

        # Image validation...
        images = []
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            if os.path.exists(image_path):
                images.append(image_path)
            else:
                self.report({'WARNING'}, f"File {image_file} does not exist, skipping.")

        if not images:
            self.report({'ERROR'}, "No valid image files found!")
            return {'CANCELLED'}

        # FFmpeg path
        # ffmpeg_path = rf"D:\desktop\wheels\ffmpeg\bin\ffmpeg.exe"
        ffmpeg_path = rf"{path[1]}\addons\CubeVi_Swizzle_Blender\wheels\ffmpeg\bin\ffmpeg.exe"

        # Video rendering command
        image_sequence_path = os.path.join(input_folder, f"quilt_frame_%d.png")
        render_command = [
            ffmpeg_path,
            '-f', 'image2',
            '-framerate', str(context.scene.fps),
            '-i', image_sequence_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-threads', '2',
            '-vsync', '0',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_video_path
        ]

        # Execute video rendering
        try:
            subprocess.run(render_command, check=True)
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"Video rendering failed: {str(e)}")
            return {'CANCELLED'}

        # Generate cover image
        cover_path = os.path.join(input_folder, "cover.png")
        cover_command = [
            ffmpeg_path,
            '-i', output_video_path,
            '-vf', 'crop=540:960:0:0',  # Crop the top-left 540x960 region
            '-vframes', '1',           # Process only the first frame
            '-y',                      # Overwrite existing files
            cover_path
        ]

        try:
            subprocess.run(cover_command, check=True)
            if os.path.exists(cover_path):
                self.report({'INFO'}, f"Preview image saved to: {cover_path}")
            else:
                self.report({'WARNING'}, "Preview image generation failed")
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"Preview image generation failed: {str(e)}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Video saved to: {output_video_path}")
        return {'FINISHED'}



class RenderAnimation1(bpy.types.Operator):
    """Render Animation"""
    bl_idname = "object.ani_render"
    bl_label = "Render Animation"
    bl_options = {'REGISTER', 'UNDO'}

    # 用于跟踪渲染进度和参数的属性
    index: bpy.props.IntProperty(default=0)
    camera = None
    original_path = ""
    original_shift_x = 0.0
    original_clip_start = 0.0
    original_clip_end = 0.0
    original_focus_distance = 0.0
    view_matrix = None
    cameraSize = 0.0
    focal_plane = 0.0
    _timer = None
    index_ = 0
    cancel_render = False  # Flag to mark if rendering is canceled

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    def execute(self, context):
        
        operator_id = 4
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        scene = bpy.context.scene
        frame_start = scene.frame_s
        frame_end = scene.frame_e

        if frame_start <= frame_end:

            bpy.context.scene.frame_current = frame_start
            # Save original parameters
            self.camera = context.scene.camera
            self.original_path = context.scene.my_filepath
            self.original_shift_x = self.camera.data.shift_x
            self.original_clip_start = self.camera.data.clip_start
            self.original_clip_end = self.camera.data.clip_end
            self.original_focus_distance = self.camera.data.dof.focus_distance

            self.camera.data.type = 'PERSP'
            self.camera.data.dof.use_dof = True
            self.camera.data.clip_start = context.scene.clip_near
            self.camera.data.clip_end = context.scene.clip_far
            self.camera.data.dof.focus_distance = context.scene.focal_plane

            depsgraph = context.evaluated_depsgraph_get()
            self.view_matrix = self.camera.matrix_world.inverted()
            projection_matrix = self.camera.calc_matrix_camera(
                depsgraph=depsgraph,
                x=540,
                y=960,
                scale_x=1.0,
                scale_y=1.0
            )
            fov = 2 * math.atan(1 / projection_matrix[1][1])
            self.cameraSize = context.scene.focal_plane * math.tan(fov / 2)
            self.focal_plane = context.scene.focal_plane

            # Initialize progress bar and modal loop
            self.index = 0
            context.window_manager.progress_begin(0, 40)  # Render 40 images per frame
            self._timer = context.window_manager.event_timer_add(0.0001, window=context.window)
            context.window_manager.modal_handler_add(self)

            frame_start += 1
            self.index_ += 1

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            # Listen for ESC key press to cancel rendering
            self.cancel_render = True

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        scene = bpy.context.scene

        if self.cancel_render: 
            self.cleanup(context)
            self.report({"INFO"}, "Render was cancelled.")
            return {'CANCELLED'}

        if scene.frame_current > scene.frame_e:
            self.cleanup(context)
            self.report({'INFO'}, "All frames rendered successfully.")
            return {'FINISHED'}

        if self.index >= 40:  
            original_path = context.scene.my_filepath
            original_path = os.path.normpath(original_path)
            print(original_path)
            single_width = 540
            single_height = 960
            rows = 5
            cols = 8
            width = single_width * cols
            height = single_height * rows
            new_image = Image.new('RGB', (width, height))

            for i in range(rows):
                for j in range(cols):
                    image_filename = f"_{i * cols + j:03d}.png"
                    image_path = os.path.join(original_path, image_filename)
                    img = Image.open(image_path)
                    left = j * single_width
                    upper = i * single_height
                    right = left + single_width
                    lower = upper + single_height

                    new_image.paste(img, (left, upper, right, lower))

            output_path = os.path.join(original_path, f"quilt_frame_{scene.frame_current}.png")
            new_image.save(output_path)
            self.index_ += 1
            self.report({'INFO'}, f"Quilt synthesize successfully, path is {output_path}")

            # Delete the 40 rendered single images
            for i in range(40):
                file_name = f"{original_path}\_{i:03d}.png"
                if os.path.exists(file_name):
                    os.remove(file_name)
                    print(f"Deleted: {file_name}")
                else:
                    print(f"File not found: {file_name}")

            scene.frame_current += 1  
            self.index = 0 

            context.window_manager.progress_update(self.index)
            return {'RUNNING_MODAL'}

        offsetangle = (0.5 - self.index / (40 - 1)) * math.radians(40)
        offset = self.focal_plane * offsetangle
        new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix

        self.camera.matrix_world = new_view_matrix.inverted()
        self.camera.data.shift_x = self.original_shift_x + 0.5 * offset / self.cameraSize

        output_path = f"{self.original_path}_{self.index:03d}.png"
        context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        context.window_manager.progress_update(self.index)
        self.index += 1

        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        """Reset camera settings and end rendering"""
        self.camera.data.clip_start = self.original_clip_start
        self.camera.data.clip_end = self.original_clip_end
        self.camera.data.dof.focus_distance = self.original_focus_distance
        self.camera.data.shift_x = self.original_shift_x
        self.camera.matrix_world = self.view_matrix.inverted()

        context.scene.render.filepath = self.original_path
        context.window_manager.progress_end()




# 主函数
def upload_file_video(self, file_path1 , file_path2, title):
    global message_
    message_ = "Upload failed"
    print(file_path1)
    if not os.path.isfile(file_path1):
        print("Preview image does not exist")
        self.report({'INFO'}, "Preview image does not exist, ensure that cover.png exists in the file path")
        return
    if not os.path.isfile(file_path2):
        print("Video does not exist")
        self.report({'INFO'}, "Video does not exist, ensure that output.mp4 exists in the file path")
        return

    # Get the MIME type of the file (only PNG type is supported here)
    mime_type = "video/mp4"

    base64_string_1 = file_to_base64(file_path1)
    base64_string_2 = file_to_base64(file_path2)

    data_to_send = {
        "previewFile": f"data:{mime_type};base64,{base64_string_1}",
        "previewFileName": os.path.basename(file_path1),
        "previewFileType": mime_type.split("/")[1],
        "resourceFile": f"data:{mime_type};base64,{base64_string_2}",
        "resourceFileName": os.path.basename(file_path2),
        "resourceFileType": mime_type.split("/")[1],
        "title": title
    }

    json_data = json.dumps(data_to_send)

    
    data_chunks = split_data(json_data, 1024 * 1024)  # Split size is 1MB

    
    current_chunk_index = 0

    send_via_websockets(on_open_choice1, data_chunks=data_chunks, current_chunk_index=current_chunk_index)


class ConnectVideoPlatform(bpy.types.Operator):
    """Upload video to the platform"""
    bl_idname = "object.vplatfrom"
    bl_label = "Upload video to the platform"
    bl_options = {'REGISTER', 'UNDO'}

    title: StringProperty(
        name="title of the video",
        description="set the title of the video",
        default="default",
    )

    # 执行操作的前提
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False


    def execute(self, context):
        
        operator_id = 5
        send_via_websockets(on_open_choice2, operator_id=operator_id, biz_type="BLENDER_CLICK")
        print(self.title)
        fp = bpy.context.scene.my_filepath
        fp1 = os.path.join(fp,"cover.png")
        fp2 = os.path.join(fp,"output.mp4")
        print(fp)
        print(fp1)
        print(fp2)
        upload_file_video(self, fp1, fp2 , self.title)
        return {'CANCELLED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)