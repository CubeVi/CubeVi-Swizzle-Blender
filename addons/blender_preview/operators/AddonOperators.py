# ##### BEGIN GPL LICENSE BLOCK #####
#
#  Copyright © GJQ, OpenStageAI
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
# 遍历 wheels 文件夹中的所有子文件夹
for folder_name in os.listdir(wheels_folder_path):
    folder_path = os.path.join(wheels_folder_path, folder_name)

    # 检查是否是文件夹
    if os.path.isdir(folder_path):
        # 假设每个子文件夹中都有一个可以导入的模块
        try:
            # 动态导入该子文件夹中的模块
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
from bpy.props import StringProperty


from Cryptodome import Random
from Cryptodome.Cipher import AES


flag = False
linenumber = None
obliquity = None
deviation = None
is_drawing = False
frustum_draw_handler = None
operator_id = 0

window_name = "Real time Display"


# def initialize_pygame_window(dis_x):
#     # 初始化 Pygame
#     # print(f"dis_x = {dis_x}")
#     os.environ['SDL_VIDEO_WINDOW_POS'] = f"{dis_x},0"
#     pygame.init()
#
#     # 设置窗口大小
#     window_width, window_height = 1440, 2560  # 根据需要调整大小
#     screen = pygame.display.set_mode((window_width, window_height),pygame.RESIZABLE)
#
#     # 设置窗口标题
#     pygame.display.set_caption("LFD Viewer")
#
#     return screen

# def update_pygame_window(screen, numpy_array):
#     # 将 NumPy 数组转换为 Pygame 的 Surface
#     numpy_array = numpy_array.transpose(1, 0, 2)
#     surface = pygame.surfarray.make_surface(numpy_array)
#
#     # 更新窗口显示内容
#     screen.blit(surface, (0, 0))
#     pygame.display.flip()

def initialize_cv_window():
    window_name = "Real time Display"
    monitors = get_monitors()

    # 寻找满足分辨率为 1440x2560 的显示器
    for monitor in monitors:
        if monitor.width == 1440 and monitor.height == 2560:
            # 首先创建一个窗口
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # 调整窗口大小为屏幕的分辨率
            cv2.resizeWindow(window_name, monitor.width, monitor.height)

            # 先移动到对应显示器的左上角
            cv2.moveWindow(window_name, monitor.x, monitor.y)

            # 再设置全屏
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            break


def update_cv_window(window, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, frame)


# def func to decrypt platform device config information

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

def on_open_choice2(ws):

    data_type = "tracking"
    data_to_send = {
        "bizId": operator_id,
        "bizType": "BLENDER_CLICK"
    }

    final_data = {
        "type": data_type,
        "data": data_to_send
    }

    ws.send(json.dumps(final_data))
    ws.close()

def c_p():
    ws_url = "ws://127.0.0.1:9001"
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open_choice2,
                                )
    ws.run_forever()


class FrustumOperator(bpy.types.Operator):
    """Show the camera frustum"""
    bl_idname = "object.frustum"
    bl_label = "Show Frustum"
    bl_options = {'REGISTER', 'UNDO'}


    def setupCameraFrustumShader(self):
        """设置绘制相机视锥的着色器"""
        pass

    def drawCameraFrustum(self, context):
        """绘制相机视锥"""
        scene = context.scene
        camera = scene.camera
        clip_near = scene.clip_near
        clip_far = scene.clip_far
        focal_plane = scene.focal_plane

        # 获取相机视锥的顶点
        coords_local = self.calculate_frustum_coordinates(camera, context, clip_near, clip_far, focal_plane)

        # 创建视锥可视化
        self.create_frustum_visualization(context, coords_local, camera)

    def calculate_frustum_coordinates(self, camera, context, clip_near, clip_far, focal_plane):
        """
        计算相机视锥的四个顶点，并根据相机的位置和朝向进行转换
        """
        # 获取相机的世界矩阵
        view_matrix = camera.matrix_world.copy()
        view_frame = camera.data.view_frame(scene=context.scene)

        # 获取视锥体的四个顶点
        scale = 1.39
        view_frame_upper_right = view_frame[0]/scale
        view_frame_lower_right = view_frame[1]/scale
        view_frame_lower_left = view_frame[2]/scale
        view_frame_upper_left = view_frame[3]/scale
        view_frame_distance = abs(view_frame_upper_right[2])/scale

        # 使用世界矩阵对视锥体坐标进行转换
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
        创建视锥可视化 (直接绘制)
        """
        # 定义视锥体的边界线
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

        # 绘制视锥
        frustum_shader.bind()
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)

        # 绘制视锥的边界
        frustum_shader.uniform_float("color", (0.3, 0, 0, 1))  # 设置颜色为红色
        batch_lines.draw(frustum_shader)

        gpu.state.depth_mask_set(False)
        gpu.state.blend_set('ALPHA')

        # 填充视锥的面
        frustum_indices_faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 4, 5), (0, 5, 1),
            (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3),
            (3, 7, 4), (3, 4, 0),
        ]
        batch_faces = batch_for_shader(frustum_shader, 'TRIS', {"pos": coords_local}, indices=frustum_indices_faces)

        # 设置其他面的颜色为半透明灰色
        frustum_shader.uniform_float("color", (0.5, 0.5, 0.5, 0.1))  # 半透明灰色
        batch_faces.draw(frustum_shader)

        # 设置焦平面的颜色为黄色 (黄色是 (1, 1, 0))
        focal_plane_indices_faces = [
            (8, 9, 10), (8, 10, 11)
        ]
        focal_plane_faces = batch_for_shader(frustum_shader, 'TRIS', {"pos": coords_local},
                                             indices=focal_plane_indices_faces)
        frustum_shader.uniform_float("color", (1, 1, 0, 0.1))  # 设置焦平面的颜色为黄色
        focal_plane_faces.draw(frustum_shader)

        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')

    def start(self, context):
        """开始绘制相机视锥"""
        # 设置相机视锥和着色器
        global is_drawing
        global frustum_draw_handler
        self.setupCameraFrustumShader()

        # 如果没有绘制处理程序，添加它
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
        """执行操作符"""
        # 启动或停止视锥的绘制
        self.start(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        """保持操作符运行并监控事件"""
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        """在点击按钮时检查是否已经绘制过视锥体"""
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
        config_path = os.path.join(os.getenv('APPDATA'), 'OpenstageAI', 'deviceConfig.json')
        global operator_id
        operator_id = 0
        c_p()
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

    # 执行操作的前提
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    # 数据初始化
    def __init__(self):
        self.offscreen = None  # 用于单相机纹理的存储
        self.final_offscreen = None  # 用于存储拼接后的大纹理
        self.display_offscreen = None  # 用于显示纹理的离屏缓冲区
        self.shader = None
        self.clear_shader = None
        self.batch = None
        self.clear_batch = None
        self.display_batch = None  # 新增，用于 display_shader
        self.view_matrix = None
        self.projection_matrix = None
        self.render_width = 540  # 每张纹理的宽度
        self.render_height = 960  # 每张纹理的高度
        self.grid_rows = 5
        self.grid_cols = 8
        self.final_width = self.render_width * self.grid_cols
        self.final_height = self.render_height * self.grid_rows
        self.display_shader = None  # 用于显示最终纹理的着色器

    def setup_offscreen_rendering(self):
        """设置小纹理和大纹理的 Offscreen"""
        try:
            # 单个纹理的 Offscreen
            self.offscreen = gpu.types.GPUOffScreen(self.render_width, self.render_height)
            # print(f"单个 Offscreen 创建成功: {self.render_width}x{self.render_height}")

            # 最终拼接的大纹理 Offscreen
            self.final_offscreen = gpu.types.GPUOffScreen(self.final_width, self.final_height)
            # print(f"最终拼接 Offscreen 创建成功: {self.final_width}x{self.final_height}")

            # 用于显示纹理的 Offscreen（展示纹理）
            self.display_offscreen = gpu.types.GPUOffScreen(1440, 2560)
            # print(f"展示的交织 OffScreen 创建成功")

            return True
        except Exception as e:
            self.report({'ERROR'}, f"创建离屏缓冲区失败: {e}")
            print(f"创建离屏缓冲区失败: {e}")
            return False

    def setup_shader(self):
        """创建用于绘制纹理的着色器"""
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
            # print("着色器编译成功")
        except Exception as e:
            self.report({'ERROR'}, f"着色器编译失败: {e}")
            print(f"着色器编译失败: {e}")
            return False

        # 定义顶点和索引
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
        """为 display_offscreen 创建专门的着色器"""
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
            # print("display_offscreen 着色器编译成功")
        except gpu.types.GPUShaderCompilationError as e:
            self.report({'ERROR'}, f"display_offscreen 着色器编译失败: {e}")
            print(f"display_offscreen 着色器编译失败: {e}")
            self.display_shader = None
            return False
        except Exception as e:
            self.report({'ERROR'}, f"display_offscreen 着色器初始化失败: {e}")
            print(f"display_offscreen 着色器初始化失败: {e}")
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
        """创建用于清除颜色缓冲区的简单着色器"""
        # 使用内置的 'UNIFORM_COLOR' 着色器
        self.clear_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.clear_batch = batch_for_shader(
            self.clear_shader, 'TRI_FAN',
            {"pos": [(-1, -1), (1, -1), (1, 1), (-1, 1)]}
        )
        # print("清除着色器和批处理创建成功")

    def __update_matrices(self, context):
        """更新视图和投影矩阵"""
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
            # print("矩阵更新成功")
        else:
            self.report({'ERROR'}, "场景中未找到相机！")
            # print("场景中未找到相机！")

    def render_quilt(self, context):
        """渲染并拼接纹理到 final_offscreen"""
        self.__update_matrices(context)
        context.area.tag_redraw()

        # 绑定 final_offscreen 以进行拼接
        with self.final_offscreen.bind():
            # 获取当前的投影矩阵
            viewMatrix = gpu.matrix.get_model_view_matrix()
            projectionMatrix = gpu.matrix.get_projection_matrix()

            fov = 2 * math.atan(1 / self.projection_matrix[1][1])
            f = 1 / math.tan(fov / 2)
            near = (self.projection_matrix[2][3] / (self.projection_matrix[2][2] - 1))
            # camera_size = f * math.tan(fov / 2)

            with gpu.matrix.push_pop():
                # 重置矩阵 -> 使用标准设备坐标 [-1, 1]
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                # 设置混合和深度状态
                gpu.state.depth_test_set('GREATER_EQUAL')
                gpu.state.depth_mask_set(True)
                gpu.state.blend_set('ALPHA')

                # 清除 final_offscreen 的颜色缓冲区
                self.clear_shader.bind()
                self.clear_shader.uniform_float("color", (0.0, 0.0, 0.0, 1.0))  # 设置清除颜色为黑色
                self.clear_batch.draw(self.clear_shader)

                start_time = time.time()

                for idx in range(self.grid_rows * self.grid_cols):
                    row = idx // self.grid_cols
                    col = idx % self.grid_cols
                    row = 4 - row
                    x_offset = col * self.render_width
                    y_offset = row * self.render_height

                    # 计算中心位置在 NDC 中的坐标
                    center_x = (x_offset + self.render_width / 2) / self.final_width * 2 - 1
                    center_y = (y_offset + self.render_height / 2) / self.final_height * 2 - 1
                    cameraDistance = context.scene.focal_plane
                    # print(cameraDistance)
                    cameraSize = cameraDistance * math.tan(fov / 2)
                    offsetAngle = (0.5 - idx / (40 - 1)) * math.radians(40)
                    # offset = - f * math.tan(offsetAngle)
                    offset = cameraDistance * offsetAngle
                    # 计算新的view matrix
                    # direction = self.view_matrix.col[2].xyz.normalized()
                    # new_offset = direction * offset
                    new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix
                    # new_view_matrix = self.view_matrix.copy()
                    # 计算新的projection matrix
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
                    # print(f"第{idx + 1}个纹理，viewMatrix为{new_view_matrix},projectionMatrix为{new_projection_matrix}")
                    # print(f"渲染第 {idx + 1} 张纹理，位置: ({x_offset}, {y_offset})")

                    # 渲染到单个 offscreen
                    with self.offscreen.bind():
                        self.offscreen.draw_view3d(
                            scene=context.scene,
                            view_layer=context.view_layer,
                            view3d=context.space_data,
                            region=context.region,
                            view_matrix=new_view_matrix,
                            projection_matrix=new_projection_matrix
                        )

                    # 绘制单个纹理到 final_offscreen 的指定位置
                    self.shader.bind()
                    self.shader.uniform_sampler("image", self.offscreen.texture_color)
                    self.shader.uniform_float("scale", (
                        self.render_width / self.final_width, self.render_height / self.final_height))
                    self.shader.uniform_float("offset", (center_x, center_y))
                    self.batch.draw(self.shader)
                    gpu.shader.unbind()

                # 重置混合模式和深度状态
                gpu.state.blend_set('NONE')
                gpu.state.depth_mask_set(False)
                gpu.state.depth_test_set('NONE')

                # 重新加载原始矩阵
                gpu.matrix.load_matrix(viewMatrix)
                gpu.matrix.load_projection_matrix(projectionMatrix)

                end_time = time.time()
                # print(f"渲染并拼接 {self.grid_rows * self.grid_cols} 张纹理耗时: {end_time - start_time:.6f} 秒")

    def save(self,context):
        """在视口中绘制拼接后的纹理"""
        if self.display_offscreen:
            # 设置视口绘制区域
            draw_x = 0
            draw_y = 0
            draw_width = 1440  # 根据需要调整显示大小
            draw_height = 2560  # 直接设定为固定高度

            # 绘制 final_offscreen 的纹理到 display_offscreen
            with self.display_offscreen.bind():
                if self.display_shader:
                    try:
                        self.display_shader.bind()
                        self.display_shader.uniform_sampler("image1", self.final_offscreen.texture_color)
                        self.display_shader.uniform_float("_Slope", obliquity)  # 正确的 uniform 设置
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
                        print(f"绘制 display_shader 时出错: {e}")
                else:
                    print("display_shader 未初始化，无法绑定。")

            # 假设参数
            np.set_printoptions(threshold=np.inf)
            s = time.time()
            width, height, channels = 1440, 2560, 4
            padding = 0  # 每行有 16 字节的填充
            stride_row = width * channels + padding  # 每行的实际字节数

            # 创建一个模拟的非连续存储的 buffer
            pixel_data = self.display_offscreen.texture_color.read()

            # 将非连续的 buffer 转换为 numpy.array
            buffer_array = np.array(pixel_data, dtype=np.uint8)

            # 使用 strides 构建逻辑视图
            logical_view = np.lib.stride_tricks.as_strided(
                buffer_array,
                shape=(height, width, channels),
                strides=(stride_row, channels, 1)
            )

            # 查看逻辑视图
            # print(logical_view.shape)
            image_data = np.flipud(logical_view)
            rgb_data = image_data[:, :, :3]
            # 图像保存
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

    # 点击后调用该方法
    def execute(self, context):
        global operator_id
        operator_id = 1
        c_p()
        # 判断离屏渲染
        if not self.setup_offscreen_rendering():
            return {'CANCELLED'}

        # 判断着色器设置
        if not self.setup_shader():
            return {'CANCELLED'}

        # 设置用于清除缓冲区的着色器
        self.setup_clear_shader()

        # 设置用于显示 final_offscreen 的 display_offscreen 着色器
        if not self.setup_display_shader():
            return {'CANCELLED'}

        # 渲染和拼接纹理
        self.render_quilt(context)

        self.save(context)

        return {'FINISHED'}

    # 在execute之前执行这个方法，用作初始化
    def invoke(self, context, event):
        return self.execute(context)


class QuiltSaveOperator(bpy.types.Operator):
    """Save the preview quilt picture"""
    bl_idname = "object.quilt_save"
    bl_label = "Save Quilt Preview Picture"
    bl_options = {'REGISTER', 'UNDO'}
    _handle = None

    # 执行操作的前提
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    # 数据初始化
    def __init__(self):
        self.offscreen = None  # 用于单相机纹理的存储
        self.final_offscreen = None  # 用于存储拼接后的大纹理
        self.display_offscreen = None  # 用于显示纹理的离屏缓冲区
        self.shader = None
        self.clear_shader = None
        self.batch = None
        self.clear_batch = None
        self.display_batch = None  # 新增，用于 display_shader
        self.view_matrix = None
        self.projection_matrix = None
        self.render_width = 540  # 每张纹理的宽度
        self.render_height = 960  # 每张纹理的高度
        self.grid_rows = 5
        self.grid_cols = 8
        self.final_width = self.render_width * self.grid_cols
        self.final_height = self.render_height * self.grid_rows
        self.display_shader = None  # 用于显示最终纹理的着色器

    def setup_offscreen_rendering(self):
        """设置小纹理和大纹理的 Offscreen"""
        try:
            # 单个纹理的 Offscreen
            self.offscreen = gpu.types.GPUOffScreen(self.render_width, self.render_height)
            # print(f"单个 Offscreen 创建成功: {self.render_width}x{self.render_height}")

            # 最终拼接的大纹理 Offscreen
            self.final_offscreen = gpu.types.GPUOffScreen(self.final_width, self.final_height)
            # print(f"最终拼接 Offscreen 创建成功: {self.final_width}x{self.final_height}")

            # 用于显示纹理的 Offscreen（展示纹理）
            self.display_offscreen = gpu.types.GPUOffScreen(1440, 2560)
            # print(f"展示的交织 OffScreen 创建成功")

            return True
        except Exception as e:
            self.report({'ERROR'}, f"创建离屏缓冲区失败: {e}")
            print(f"创建离屏缓冲区失败: {e}")
            return False

    def setup_shader(self):
        """创建用于绘制纹理的着色器"""
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
            # print("着色器编译成功")
        except Exception as e:
            self.report({'ERROR'}, f"着色器编译失败: {e}")
            print(f"着色器编译失败: {e}")
            return False

        # 定义顶点和索引
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
        """为 display_offscreen 创建专门的着色器"""
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
            # print("display_offscreen 着色器编译成功")
        except gpu.types.GPUShaderCompilationError as e:
            self.report({'ERROR'}, f"display_offscreen 着色器编译失败: {e}")
            print(f"display_offscreen 着色器编译失败: {e}")
            self.display_shader = None
            return False
        except Exception as e:
            self.report({'ERROR'}, f"display_offscreen 着色器初始化失败: {e}")
            print(f"display_offscreen 着色器初始化失败: {e}")
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
        """创建用于清除颜色缓冲区的简单着色器"""
        # 使用内置的 'UNIFORM_COLOR' 着色器
        self.clear_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.clear_batch = batch_for_shader(
            self.clear_shader, 'TRI_FAN',
            {"pos": [(-1, -1), (1, -1), (1, 1), (-1, 1)]}
        )
        # print("清除着色器和批处理创建成功")

    def __update_matrices(self, context):
        """更新视图和投影矩阵"""
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
            # print("矩阵更新成功")
        else:
            self.report({'ERROR'}, "场景中未找到相机！")
            # print("场景中未找到相机！")

    def render_quilt(self, context):
        """渲染并拼接纹理到 final_offscreen"""
        self.__update_matrices(context)
        context.area.tag_redraw()

        # 绑定 final_offscreen 以进行拼接
        with self.final_offscreen.bind():
            # 获取当前的投影矩阵
            viewMatrix = gpu.matrix.get_model_view_matrix()
            projectionMatrix = gpu.matrix.get_projection_matrix()

            fov = 2 * math.atan(1 / self.projection_matrix[1][1])
            f = 1 / math.tan(fov / 2)
            near = (self.projection_matrix[2][3] / (self.projection_matrix[2][2] - 1))
            # camera_size = f * math.tan(fov / 2)

            with gpu.matrix.push_pop():
                # 重置矩阵 -> 使用标准设备坐标 [-1, 1]
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                # 设置混合和深度状态
                gpu.state.depth_test_set('GREATER_EQUAL')
                gpu.state.depth_mask_set(True)
                gpu.state.blend_set('ALPHA')

                # 清除 final_offscreen 的颜色缓冲区
                self.clear_shader.bind()
                self.clear_shader.uniform_float("color", (0.0, 0.0, 0.0, 1.0))  # 设置清除颜色为黑色
                self.clear_batch.draw(self.clear_shader)

                start_time = time.time()

                for idx in range(self.grid_rows * self.grid_cols):
                    row = idx // self.grid_cols
                    col = idx % self.grid_cols
                    row = 4 - row
                    x_offset = col * self.render_width
                    y_offset = row * self.render_height

                    # 计算中心位置在 NDC 中的坐标
                    center_x = (x_offset + self.render_width / 2) / self.final_width * 2 - 1
                    center_y = (y_offset + self.render_height / 2) / self.final_height * 2 - 1
                    cameraDistance = context.scene.focal_plane
                    # print(cameraDistance)
                    cameraSize = cameraDistance * math.tan(fov / 2)
                    offsetAngle = (0.5 - idx / (40 - 1)) * math.radians(40)
                    # offset = - f * math.tan(offsetAngle)
                    offset = cameraDistance * offsetAngle
                    # 计算新的view matrix
                    # direction = self.view_matrix.col[2].xyz.normalized()
                    # new_offset = direction * offset
                    new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix
                    # new_view_matrix = self.view_matrix.copy()
                    # 计算新的projection matrix
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
                    # print(f"第{idx + 1}个纹理，viewMatrix为{new_view_matrix},projectionMatrix为{new_projection_matrix}")
                    # print(f"渲染第 {idx + 1} 张纹理，位置: ({x_offset}, {y_offset})")

                    # 渲染到单个 offscreen
                    with self.offscreen.bind():
                        self.offscreen.draw_view3d(
                            scene=context.scene,
                            view_layer=context.view_layer,
                            view3d=context.space_data,
                            region=context.region,
                            view_matrix=new_view_matrix,
                            projection_matrix=new_projection_matrix
                        )

                    # 绘制单个纹理到 final_offscreen 的指定位置
                    self.shader.bind()
                    self.shader.uniform_sampler("image", self.offscreen.texture_color)
                    self.shader.uniform_float("scale", (
                        self.render_width / self.final_width, self.render_height / self.final_height))
                    self.shader.uniform_float("offset", (center_x, center_y))
                    self.batch.draw(self.shader)
                    gpu.shader.unbind()

                # 重置混合模式和深度状态
                gpu.state.blend_set('NONE')
                gpu.state.depth_mask_set(False)
                gpu.state.depth_test_set('NONE')

                # 重新加载原始矩阵
                gpu.matrix.load_matrix(viewMatrix)
                gpu.matrix.load_projection_matrix(projectionMatrix)

                end_time = time.time()
                # print(f"渲染并拼接 {self.grid_rows * self.grid_cols} 张纹理耗时: {end_time - start_time:.6f} 秒")

    def save(self,context):
        """在视口中绘制拼接后的纹理"""
        # 将非连续的 buffer 转换为 numpy.array
        width, height, channels = 4320, 4800, 4
        padding = 0  # 每行有 16 字节的填充
        stride_row = width * channels + padding  # 每行的实际字节数

        # 创建一个模拟的非连续存储的 buffer
        pixel_data = self.final_offscreen.texture_color.read()

        # 将非连续的 buffer 转换为 numpy.array
        buffer_array = np.array(pixel_data, dtype=np.uint8)

        # 使用 strides 构建逻辑视图
        logical_view = np.lib.stride_tricks.as_strided(
            buffer_array,
            shape=(height, width, channels),
            strides=(stride_row, channels, 1)
        )

        # 查看逻辑视图
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

    # 点击后调用该方法
    def execute(self, context):
        # 判断离屏渲染
        global operator_id
        operator_id = 1
        c_p()

        if not self.setup_offscreen_rendering():
            return {'CANCELLED'}

        # 判断着色器设置
        if not self.setup_shader():
            return {'CANCELLED'}

        # 设置用于清除缓冲区的着色器
        self.setup_clear_shader()

        # 设置用于显示 final_offscreen 的 display_offscreen 着色器
        if not self.setup_display_shader():
            return {'CANCELLED'}

        # 渲染和拼接纹理
        self.render_quilt(context)

        self.save(context)

        return {'FINISHED'}

    # 在execute之前执行这个方法，用作初始化
    def invoke(self, context, event):
        return self.execute(context)


class LFDPreviewOperator(bpy.types.Operator):
    """Start LightField Preview, Esc to exit"""
    bl_idname = "object.preview"
    bl_label = "Realtime LightField Preview"
    bl_options = {'REGISTER', 'UNDO'}

    _handle = None  # 用于存储绘制句柄

    # display_x: IntProperty(
    #     name="x-axis of display",
    #     description="x axis of display",
    #     default=2560,
    # )

    # 执行操作的前提
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    # 数据初始化
    def __init__(self):
        self.offscreen = None  # 用于单相机纹理的存储
        self.final_offscreen = None  # 用于存储拼接后的大纹理
        self.display_offscreen = None  # 用于显示纹理的离屏缓冲区
        self.shader = None
        self.clear_shader = None
        self.batch = None
        self.clear_batch = None
        self.display_batch = None  # 新增，用于 display_shader
        self.view_matrix = None
        self.projection_matrix = None
        self.render_width = 540  # 每张纹理的宽度
        self.render_height = 960  # 每张纹理的高度
        self.grid_rows = 5
        self.grid_cols = 8
        self.final_width = self.render_width * self.grid_cols
        self.final_height = self.render_height * self.grid_rows
        self.display_shader = None  # 用于显示最终纹理的着色器
        self.screen = None
        self.fps = 10

    def setup_offscreen_rendering(self):
        """设置小纹理和大纹理的 Offscreen"""
        try:
            # 单个纹理的 Offscreen
            self.offscreen = gpu.types.GPUOffScreen(self.render_width, self.render_height)
            # print(f"单个 Offscreen 创建成功: {self.render_width}x{self.render_height}")

            # 最终拼接的大纹理 Offscreen
            self.final_offscreen = gpu.types.GPUOffScreen(self.final_width, self.final_height)
            # print(f"最终拼接 Offscreen 创建成功: {self.final_width}x{self.final_height}")

            # 用于显示纹理的 Offscreen（展示纹理）
            self.display_offscreen = gpu.types.GPUOffScreen(1440, 2560)
            # print(f"展示的交织 OffScreen 创建成功")

            return True
        except Exception as e:
            self.report({'ERROR'}, f"创建离屏缓冲区失败: {e}")
            print(f"创建离屏缓冲区失败: {e}")
            return False

    def setup_shader(self):
        """创建用于绘制纹理的着色器"""
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
            # print("着色器编译成功")
        except Exception as e:
            self.report({'ERROR'}, f"着色器编译失败: {e}")
            print(f"着色器编译失败: {e}")
            return False

        # 定义顶点和索引
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
        """为 display_offscreen 创建专门的着色器"""
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
            # print("display_offscreen 着色器编译成功")
        except gpu.types.GPUShaderCompilationError as e:
            self.report({'ERROR'}, f"display_offscreen 着色器编译失败: {e}")
            print(f"display_offscreen 着色器编译失败: {e}")
            self.display_shader = None
            return False
        except Exception as e:
            self.report({'ERROR'}, f"display_offscreen 着色器初始化失败: {e}")
            print(f"display_offscreen 着色器初始化失败: {e}")
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
        """创建用于清除颜色缓冲区的简单着色器"""
        # 使用内置的 'UNIFORM_COLOR' 着色器
        self.clear_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.clear_batch = batch_for_shader(
            self.clear_shader, 'TRI_FAN',
            {"pos": [(-1, -1), (1, -1), (1, 1), (-1, 1)]}
        )
        # print("清除着色器和批处理创建成功")

    def __update_matrices(self, context):
        """更新视图和投影矩阵"""
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
            # print("矩阵更新成功")
        else:
            self.report({'ERROR'}, "场景中未找到相机！")
            print("场景中未找到相机！")

    def render_quilt(self, context):
        """渲染并拼接纹理到 final_offscreen"""
        self.__update_matrices(context)
        context.area.tag_redraw()

        # 绑定 final_offscreen 以进行拼接
        with self.final_offscreen.bind():
            # 获取当前的投影矩阵
            viewMatrix = gpu.matrix.get_model_view_matrix()
            projectionMatrix = gpu.matrix.get_projection_matrix()

            fov = 2 * math.atan(1 / self.projection_matrix[1][1])
            f = 1 / math.tan(fov / 2)
            # near = (self.projection_matrix[2][3] / (self.projection_matrix[2][2] - 1))
            # camera_size = f * math.tan(fov / 2)

            with gpu.matrix.push_pop():
                # 重置矩阵 -> 使用标准设备坐标 [-1, 1]
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                # 设置混合和深度状态
                gpu.state.depth_test_set('GREATER_EQUAL')
                gpu.state.depth_mask_set(True)
                gpu.state.blend_set('ALPHA')

                # 清除 final_offscreen 的颜色缓冲区
                self.clear_shader.bind()
                self.clear_shader.uniform_float("color", (0.0, 0.0, 0.0, 1.0))  # 设置清除颜色为黑色
                self.clear_batch.draw(self.clear_shader)

                start_time = time.time()

                for idx in range(self.grid_rows * self.grid_cols):
                    row = idx // self.grid_cols
                    col = idx % self.grid_cols
                    row = 4 - row
                    x_offset = col * self.render_width
                    y_offset = row * self.render_height

                    # 计算中心位置在 NDC 中的坐标
                    center_x = (x_offset + self.render_width / 2) / self.final_width * 2 - 1
                    center_y = (y_offset + self.render_height / 2) / self.final_height * 2 - 1
                    cameraDistance = context.scene.focal_plane
                    # print(cameraDistance)
                    cameraSize = cameraDistance * math.tan(fov / 2)
                    offsetAngle = (0.5 - idx / (40 - 1)) * math.radians(40)
                    # offset = - f * math.tan(offsetAngle)
                    offset = cameraDistance * math.tan(offsetAngle)
                    # 计算新的view matrix
                    # direction = self.view_matrix.col[2].xyz.normalized()
                    # new_offset = direction * offset
                    new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix
                    # new_view_matrix = self.view_matrix.copy()
                    # 计算新的projection matrix
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
                    # print(f"第{idx + 1}个纹理，viewMatrix为{new_view_matrix},projectionMatrix为{new_projection_matrix}")
                    # print(f"渲染第 {idx + 1} 张纹理，位置: ({x_offset}, {y_offset})")

                    # 渲染到单个 offscreen
                    with self.offscreen.bind():
                        self.offscreen.draw_view3d(
                            scene=context.scene,
                            view_layer=context.view_layer,
                            view3d=context.space_data,
                            region=context.region,
                            view_matrix=new_view_matrix,
                            projection_matrix=new_projection_matrix
                        )

                    # 绘制单个纹理到 final_offscreen 的指定位置
                    self.shader.bind()
                    self.shader.uniform_sampler("image", self.offscreen.texture_color)
                    self.shader.uniform_float("scale", (
                        self.render_width / self.final_width, self.render_height / self.final_height))
                    self.shader.uniform_float("offset", (center_x, center_y))
                    self.batch.draw(self.shader)
                    gpu.shader.unbind()

                # 重置混合模式和深度状态
                gpu.state.blend_set('NONE')
                gpu.state.depth_mask_set(False)
                gpu.state.depth_test_set('NONE')

                # 重新加载原始矩阵
                gpu.matrix.load_matrix(viewMatrix)
                gpu.matrix.load_projection_matrix(projectionMatrix)

                end_time = time.time()
                # print(f"渲染并拼接 {self.grid_rows * self.grid_cols} 张纹理耗时: {end_time - start_time:.6f} 秒")

    def draw_callback_px(self, context, region):
        """在视口中绘制拼接后的纹理"""
        if self.display_offscreen:
            # 设置视口绘制区域
            draw_x = 0
            draw_y = 0
            draw_width = 1440  # 根据需要调整显示大小
            draw_height = 2560  # 直接设定为固定高度

            # 绘制 final_offscreen 的纹理到 display_offscreen
            with self.display_offscreen.bind():
                if self.display_shader:
                    try:
                        self.display_shader.bind()
                        self.display_shader.uniform_sampler("image1", self.final_offscreen.texture_color)
                        self.display_shader.uniform_float("_Slope", obliquity)  # 正确的 uniform 设置
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
                        print(f"绘制 display_shader 时出错: {e}")
                else:
                    print("display_shader 未初始化，无法绑定。")

            # 假设参数
            np.set_printoptions(threshold=np.inf)
            s = time.time()
            width, height, channels = 1440, 2560, 4
            padding = 0  # 每行有 16 字节的填充
            stride_row = width * channels + padding  # 每行的实际字节数

            # 创建一个模拟的非连续存储的 buffer
            pixel_data = self.display_offscreen.texture_color.read()

            # 将非连续的 buffer 转换为 numpy.array
            buffer_array = np.array(pixel_data, dtype=np.uint8)

            # 使用 strides 构建逻辑视图
            logical_view = np.lib.stride_tricks.as_strided(
                buffer_array,
                shape=(height, width, channels),
                strides=(stride_row, channels, 1)
            )

            # 查看逻辑视图
            # print(logical_view.shape)

            # draw_texture_2d(self.final_offscreen.texture_color, (0,0), 540, 960)

            image_data = np.flipud(logical_view)
            rgb_data = image_data[:, :, :3]
            # update_pygame_window(self.screen, rgb_data)
            update_cv_window(window_name, rgb_data)

            # 图像保存
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
        global operator_id
        operator_id = 1
        c_p()
        # 判断离屏渲染
        if not self.setup_offscreen_rendering():
            return {'CANCELLED'}

        # 判断着色器设置
        if not self.setup_shader():
            return {'CANCELLED'}

        # 设置用于清除缓冲区的着色器
        self.setup_clear_shader()

        # 设置用于显示 final_offscreen 的 display_offscreen 着色器
        if not self.setup_display_shader():
            return {'CANCELLED'}

        # 渲染和拼接纹理
        self.render_quilt(context)

        # 添加绘制回调
        args = (self, context)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback_px, args, 'WINDOW', 'POST_PIXEL'
        )

        # 添加定时器
        self._timer = context.window_manager.event_timer_add((1 / self.fps), window=context.window)  # 每 0.1 秒刷新一次

        # 添加 modal 处理器
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    #
    def modal(self, context, event):
        if event.type == 'TIMER':  # 定时器触发时更新渲染
            self.render_quilt(context)
            context.area.tag_redraw()

        if event.type in {'ESC'}:  # 按下 ESC 键时退出
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


    # 在execute之前执行这个方法，用作初始化
    # def invoke(self, context, event):
    #     # return self.execute(context), context.window_manager.invoke_props_dialog(self)
    #     return context.window_manager.invoke_props_dialog(self)



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

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    def execute(self, context):
        global operator_id
        operator_id = 2
        c_p()
        # 保存原始参数
        self.camera = context.scene.camera
        self.original_path = context.scene.my_filepath
        self.original_shift_x = self.camera.data.shift_x
        self.original_clip_start = self.camera.data.clip_start
        self.original_clip_end = self.camera.data.clip_end
        self.original_focus_distance = self.camera.data.dof.focus_distance

        # 设置初始相机参数
        self.camera.data.type = 'PERSP'
        self.camera.data.dof.use_dof = True
        self.camera.data.clip_start = context.scene.clip_near
        self.camera.data.clip_end = context.scene.clip_far
        self.camera.data.dof.focus_distance = context.scene.focal_plane

        # 计算必要矩阵
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

        # 初始化进度条和模态循环
        self.index = 0
        context.window_manager.progress_begin(0, 40)
        self._timer = context.window_manager.event_timer_add(0.0001, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        """恢复原始设置并结束渲染"""
        self.camera.data.clip_start = self.original_clip_start
        self.camera.data.clip_end = self.original_clip_end
        self.camera.data.dof.focus_distance = self.original_focus_distance
        self.camera.data.shift_x = self.original_shift_x
        self.camera.matrix_world = self.view_matrix.inverted()

        context.scene.render.filepath = self.original_path
        context.window_manager.progress_end()

    def modal(self, context, event):
        if event.type == 'ESC':
            # 清理原始设置并结束渲染
            self.cleanup(context)
            self.report({"INFO"}, "Render Cancelled")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}


        if self.index >= 40:
            # 渲染完成，清理资源
            self.cleanup(context)
            self.report({'INFO'}, "All pictures saved successfully.")
            return {'FINISHED'}

        # 计算当前视角参数
        offsetangle = (0.5 - self.index / (40 - 1)) * math.radians(40)
        offset = self.focal_plane * offsetangle
        new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix

        # 更新相机位置和参数
        self.camera.matrix_world = new_view_matrix.inverted()
        self.camera.data.shift_x = self.original_shift_x + 0.5 * offset / self.cameraSize

        # 设置渲染路径并执行渲染
        output_path = f"{self.original_path}_{self.index:03d}.png"
        context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        # 更新进度并准备下一帧
        context.window_manager.progress_update(self.index)
        self.index += 1

        # 允许界面刷新
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}


class QuiltSaveOperator1(bpy.types.Operator):
    """Save Quilt image"""
    bl_idname = "object.quilt_1_save"
    bl_label = "Synthesize Quilt Picture"
    bl_options = {'REGISTER', 'UNDO'}

    # 执行操作的前提
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    def execute(self, context):
        global operator_id
        operator_id = 2
        c_p()
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


message_ = "上传失败"

# 将文件内容转换为 Base64 编码
def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
        return base64.b64encode(file_content).decode('utf-8')

# 将大的数据分割成大小为 max_size 的小片
def split_data(data, max_size):
    return [data[i:i + max_size] for i in range(0, len(data), max_size)]

# WebSocket 回调函数
def on_open_choice1(ws):
    print("已连接到服务器")
    # 触发数据发送
    send_next_chunk(ws)

from websocket import WebSocketTimeoutException

def send_next_chunk(ws):
    global current_chunk_index  # 记录当前发送的分片索引

    if current_chunk_index < len(data_chunks):
        print(f"当前在第{current_chunk_index+1}分片,总共{len(data_chunks)}分片")
        chunk = data_chunks[current_chunk_index]
        # print("zheshi chunk")
        # print(chunk)

        data_split = {
            "type": "quilt",
            "data": chunk,
            "done": False  # 最后一片设置 done 为 True
        }
        # 打印前100个和后100个
        print(str(data_split)[:100])
        print(str(data_split)[-100:])

        # 发送分片数据
        ws.send(json.dumps(data_split))
        print(f"发送了第 {current_chunk_index + 1} 片数据")
        current_chunk_index += 1

        # 继续发送下一个分片，延迟少许时间
        time.sleep(0.1)  # 可根据需要调整延迟时间，避免过快发送影响网络

        # 继续发送下一片数据
        send_next_chunk(ws)
    else:
        # 所有数据片发送完毕后，发送结束标识
        done_message = {
            "type": "quilt",
            "data": None,
            "done": True
        }
        print("发送结束标识")
        print(str(done_message)[:100])
        ws.send(json.dumps(done_message))
        print("数据传输结束")
        ws.close()  # 发送结束后关闭 WebSocket 连接


# def on_close(ws, close_status_code, close_msg):
#     # global message_
#     # message_ = "连接已关闭"
#     print("连接已关闭")
#
#
# def on_error(ws, error):
#     global message_
#     message_ = f"发生错误: {error}"
#     print(f"发生错误: {error}")


# 主函数
def upload_file(self, file_path1 , file_path2, title):
    global message_
    message_ = "上传失败"
    print(file_path1)
    if not os.path.isfile(file_path1):
        print("封面图片不存在")
        self.report({'INFO'}, "封面图片不存在，确保文件路径下存在_020.png")
        return
    if not os.path.isfile(file_path2):
        print("合成图片不存在")
        self.report({'INFO'}, "source图片不存在,确保文件路径下存在quilt_synthesize_result.png")
        return

    # 获取文件的 MIME 类型（这里只支持 PNG 类型）
    mime_type = "image/png"

    # 将文件内容转为 Base64 编码
    base64_string_1 = file_to_base64(file_path1)
    base64_string_2 = file_to_base64(file_path2)

    # 创建要发送的数据对象
    data_to_send = {
        "previewFile": f"data:{mime_type};base64,{base64_string_1}",
        "previewFileName": os.path.basename(file_path1),
        "previewFileType": mime_type.split("/")[1],
        "resourceFile": f"data:{mime_type};base64,{base64_string_2}",
        "resourceFileName": os.path.basename(file_path2),
        "resourceFileType": mime_type.split("/")[1],
        "title": title
    }

    # 序列化整个 data_to_send 对象为 JSON 字符串
    json_data = json.dumps(data_to_send)

    # 获取 Base64 字符串并分片
    global data_chunks
    data_chunks = split_data(json_data, 1024 * 1024)  # 分片大小为 1MB

    # 设置分片的索引
    global current_chunk_index
    current_chunk_index = 0

    # 连接到 WebSocket 服务器
    ws_url = "ws://127.0.0.1:9001"  # 服务器地址
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open_choice1,
                                on_message=None,
                                on_close=None,
                                on_error=None)

    # 启动 WebSocket 客户端
    ws.run_forever()


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
    # 执行操作的前提
    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False


    def execute(self, context):
        global operator_id
        operator_id = 3
        c_p()
        print(self.title)
        # file_path = "D:/desktop/temp/quilt_result.png"  # 替换为你的 PNG 文件路径
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
        global operator_id
        operator_id = 4
        c_p()
        # 获取文件路径
        output_path = context.scene.my_filepath

        # 设置输出路径，确保目录存在
        if output_path:
            context.scene.render.filepath = output_path

            # 设置渲染为静态图像，渲染整个动画
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
        global operator_id
        operator_id = 4
        c_p()
        input_folder = context.scene.my_filepath
        output_video_path = os.path.join(input_folder, "output.mp4")

        # 原有图像序列处理部分...
        image_files = [f"quilt_frame_{i}.png" for i in range(context.scene.frame_s, context.scene.frame_e + 1)]

        # 图像验证部分...
        images = []
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            if os.path.exists(image_path):
                images.append(image_path)
            else:
                self.report({'WARNING'}, f"文件 {image_file} 不存在，跳过。")

        if not images:
            self.report({'ERROR'}, "没有找到有效的图像文件！")
            return {'CANCELLED'}

        # FFmpeg路径
        # ffmpeg_path = rf"D:\desktop\wheels\ffmpeg\bin\ffmpeg.exe"
        ffmpeg_path = rf"{path[1]}\addons\CubeVi_Swizzle_Blender\wheels\ffmpeg\bin\ffmpeg.exe"

        # 视频渲染命令
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

        # 执行视频渲染
        try:
            subprocess.run(render_command, check=True)
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"视频渲染失败: {str(e)}")
            return {'CANCELLED'}

        # 生成封面图
        cover_path = os.path.join(input_folder, "cover.png")
        cover_command = [
            ffmpeg_path,
            '-i', output_video_path,
            '-vf', 'crop=540:960:0:0',  # 裁剪左上角540x960区域
            '-vframes', '1',           # 只处理第一帧
            '-y',                      # 覆盖已有文件
            cover_path
        ]

        try:
            subprocess.run(cover_command, check=True)
            if os.path.exists(cover_path):
                self.report({'INFO'}, f"封面图已保存到: {cover_path}")
            else:
                self.report({'WARNING'}, "封面图生成失败")
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"封面图生成失败: {str(e)}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"视频已保存到: {output_video_path}")
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
    cancel_render = False  # 标记是否取消渲染

    @classmethod
    def poll(cls, context: bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    def execute(self, context):
        global operator_id
        operator_id = 4
        c_p()
        scene = bpy.context.scene
        frame_start = scene.frame_s
        frame_end = scene.frame_e

        if frame_start <= frame_end:

            bpy.context.scene.frame_current = frame_start
            # 保存原始参数
            self.camera = context.scene.camera
            self.original_path = context.scene.my_filepath
            self.original_shift_x = self.camera.data.shift_x
            self.original_clip_start = self.camera.data.clip_start
            self.original_clip_end = self.camera.data.clip_end
            self.original_focus_distance = self.camera.data.dof.focus_distance

            # 设置初始相机参数
            self.camera.data.type = 'PERSP'
            self.camera.data.dof.use_dof = True
            self.camera.data.clip_start = context.scene.clip_near
            self.camera.data.clip_end = context.scene.clip_far
            self.camera.data.dof.focus_distance = context.scene.focal_plane

            # 计算必要矩阵
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

            # 初始化进度条和模态循环
            self.index = 0
            context.window_manager.progress_begin(0, 40)  # 渲染每帧 40 张图
            self._timer = context.window_manager.event_timer_add(0.0001, window=context.window)
            context.window_manager.modal_handler_add(self)

            frame_start += 1
            self.index_ += 1

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            # 监听 ESC 键按下，取消渲染
            self.cancel_render = True

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        scene = bpy.context.scene

        if self.cancel_render:  # 检查是否需要取消渲染
            self.cleanup(context)
            self.report({"INFO"}, "Render was cancelled.")
            return {'CANCELLED'}

        if scene.frame_current > scene.frame_e:  # 如果当前帧大于结束帧，则结束渲染
            self.cleanup(context)
            self.report({'INFO'}, "All frames rendered successfully.")
            return {'FINISHED'}

        if self.index >= 40:  # 每帧渲染 40 张图像后，合成并保存
            # 合成所有的 40 张图片成一张大图
            original_path = context.scene.my_filepath
            original_path = os.path.normpath(original_path)
            print(original_path)
            single_width = 540
            single_height = 960
            rows = 5  # 5 行
            cols = 8  # 8 列
            width = single_width * cols
            height = single_height * rows
            new_image = Image.new('RGB', (width, height))

            # 遍历所有图片并合成
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

            # 保存合成后的图像
            output_path = os.path.join(original_path, f"quilt_frame_{scene.frame_current}.png")
            new_image.save(output_path)
            self.index_ += 1
            self.report({'INFO'}, f"Quilt synthesize successfully, path is {output_path}")

            # 删除已渲染的 40 张单张图片
            for i in range(40):
                file_name = f"{original_path}\_{i:03d}.png"
                if os.path.exists(file_name):
                    os.remove(file_name)
                    print(f"Deleted: {file_name}")
                else:
                    print(f"File not found: {file_name}")

            # 跳转到下一帧
            scene.frame_current += 1  # 跳到下一帧
            self.index = 0  # 重置索引以开始渲染新的 40 张图像

            context.window_manager.progress_update(self.index)
            return {'RUNNING_MODAL'}

        # 计算当前视角参数
        offsetangle = (0.5 - self.index / (40 - 1)) * math.radians(40)
        offset = self.focal_plane * offsetangle
        new_view_matrix = Matrix.Translation((offset, 0, 0)) @ self.view_matrix

        # 更新相机位置和参数
        self.camera.matrix_world = new_view_matrix.inverted()
        self.camera.data.shift_x = self.original_shift_x + 0.5 * offset / self.cameraSize

        # 设置渲染路径并执行渲染
        output_path = f"{self.original_path}_{self.index:03d}.png"
        context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        # 更新进度并准备下一帧
        context.window_manager.progress_update(self.index)
        self.index += 1

        # 允许界面刷新
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        """恢复原始设置并结束渲染"""
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
    message_ = "上传失败"
    print(file_path1)
    if not os.path.isfile(file_path1):
        print("封面图片不存在")
        self.report({'INFO'}, "封面图片不存在，确保文件路径下存在cover.png")
        return
    if not os.path.isfile(file_path2):
        print("视频不存在")
        self.report({'INFO'}, "视频不存在,确保文件路径下存在output.mp4")
        return

    # 获取文件的 MIME 类型（这里只支持 PNG 类型）
    mime_type = "video/mp4"

    # 将文件内容转为 Base64 编码
    base64_string_1 = file_to_base64(file_path1)
    base64_string_2 = file_to_base64(file_path2)

    # 创建要发送的数据对象
    data_to_send = {
        "previewFile": f"data:{mime_type};base64,{base64_string_1}",
        "previewFileName": os.path.basename(file_path1),
        "previewFileType": mime_type.split("/")[1],
        "resourceFile": f"data:{mime_type};base64,{base64_string_2}",
        "resourceFileName": os.path.basename(file_path2),
        "resourceFileType": mime_type.split("/")[1],
        "title": title
    }

    # 序列化整个 data_to_send 对象为 JSON 字符串
    json_data = json.dumps(data_to_send)

    # 获取 Base64 字符串并分片
    global data_chunks
    data_chunks = split_data(json_data, 1024 * 1024)  # 分片大小为 1MB

    # 设置分片的索引
    global current_chunk_index
    current_chunk_index = 0

    # 连接到 WebSocket 服务器
    ws_url = "ws://127.0.0.1:9001"  # 服务器地址
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open_choice1,
                                on_message=None,
                                on_close=None,
                                on_error=None)

    # 启动 WebSocket 客户端
    ws.run_forever()


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
        global operator_id
        operator_id = 5
        c_p()
        print(self.title)
        # file_path = "D:/desktop/temp/quilt_result.png"  # 替换为你的 PNG 文件路径
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