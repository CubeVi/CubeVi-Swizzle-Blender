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
import bpy
import gpu
import sys
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix
import time
from gpu_extras.presets import draw_texture_2d
import math
import numpy as np
from PIL import Image
import os
from bpy.props import IntProperty
try:
    from win32 import win32file, win32pipe
except ImportError:
    print("未找到 win32 模块，请重启 Blender 以确保正确加载 pywin32 库。")

import asyncio
import json
import cv2
import base64
from hashlib import md5
try:
    from Cryptodome import Random
    from Cryptodome.Cipher import AES
except ImportError as e:
    print(f"无法导入Cryptodome模块: {e}")

flag = False
linenumber = None
obliquity = None
deviation = None

window_name = "Real_time Display"

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

def initialize_cv_window(dis_x):
    # window_name = "Real_time Display"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, dis_x, 0)

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
    key_iv = bytes_to_key(passphrase, salt, 32+16)
    key = key_iv[:32]
    iv = key_iv[32:]
    aes = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(aes.decrypt(encrypted[16:]))
    stringData = pt.decode('utf-8')
    data = json.loads(stringData)
    return data

class connectOperator(bpy.types.Operator):
    bl_idname = "object.connect"
    bl_label = "connect"
    bl_options = {'REGISTER', 'UNDO'}
    _handle = None

    @classmethod
    def poll(cls, context:bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    items = []

    def execute(self, context:bpy.types.Context):
        try:
            config_path = os.path.join(os.getenv('APPDATA'), 'OpenstageAI', 'deviceConfig.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                device_info = json.load(f)
            
            if device_info and 'config' in device_info:
                global obliquity, linenumber, deviation
                config = device_info['config']
                password = keycode.encode()
                data = decrypt(config, password)

                configData = data['config']
                linenumber = configData.get('lineNumber','')
                obliquity = configData.get('obliquity','')
                deviation = configData.get('deviation','')
                self.report({"INFO"},"Connection Successful")
                return {'FINISHED'}
        except Exception as e:
            self.report({"INFO"},"Connection Falied")
            # print("连接失败")
            return {'FINISHED'}




class LFDSaveOperator(bpy.types.Operator):
    """目前仅支持保存png"""
    bl_idname = "object.save"
    bl_label = "Save LFD Picture"
    bl_options = {'REGISTER', 'UNDO'}
    _handle = None

    # 执行操作的前提
    @classmethod
    def poll(cls, context:bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    # 数据初始化
    def __init__(self):
        self.offscreen = None      # 用于单相机纹理的存储
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
                FragColor = color;
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
            camera_size = f * math.tan(fov / 2)

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

                    offsetAngle = (0.5 - idx / (40 - 1)) * fov / 2
                    # offset = - f * math.tan(offsetAngle)
                    offset = - f * offsetAngle * 3
                    # 计算新的view matrix
                    # direction = self.view_matrix.col[2].xyz.normalized()
                    # new_offset = direction * offset
                    new_view_matrix = Matrix.Translation((-offset, 0, 0)) @ self.view_matrix
                    # new_view_matrix = self.view_matrix.copy()
                    # 计算新的projection matrix
                    new_projection_matrix = self.projection_matrix.copy()
                    new_projection_matrix[0][2] -= offset / (camera_size * (1440 / 2560)) / 3

                    # print(f"fov={fov}, f={f}, near={near}, offsetAngle={offsetAngle}, offset={offset}")
                    print(f"第{idx + 1}个纹理，viewMatrix为{new_view_matrix},projectionMatrix为{new_projection_matrix}")
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

    def save(self):
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
            padding = 0 # 每行有 16 字节的填充
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
            output_path = bpy.context.scene.render.filepath
            # print(output_path)
            output_path += "render_result"
            if not output_path.lower().endswith(".png"):
                output_path += ".png"
            image.save(output_path)

            e = time.time()
            # print(f"time is {e - s}")
            self.report({'INFO'},"Picture saved successfully")

    # 点击后调用该方法
    def execute(self, context):
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

        self.save()

        return {'FINISHED'}

    # 在execute之前执行这个方法，用作初始化
    def invoke(self, context, event):
        return self.execute(context)


class LFDPreviewOperator(bpy.types.Operator):
    """单击后，开启光场实时预览. 单击Esc退出"""
    bl_idname = "object.preview"
    bl_label = "Lightfield rendering"
    bl_options = {'REGISTER', 'UNDO'}

    _handle = None  # 用于存储绘制句柄

    display_x: IntProperty(
        name = "x-axis of display",
        description = "x axis of display",
        default = 2560,
    )

    # 执行操作的前提
    @classmethod
    def poll(cls, context:bpy.types.Context):
        if context.scene.camera is not None and flag is False:
            return True
        return False

    # 数据初始化
    def __init__(self):
        self.offscreen = None      # 用于单相机纹理的存储
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
                FragColor = color;
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
            near = (self.projection_matrix[2][3] / (self.projection_matrix[2][2] - 1))
            camera_size = f * math.tan(fov / 2)

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

                    offsetAngle = (0.5 - idx / (40 - 1)) * fov / 2
                    # offset = - f * math.tan(offsetAngle)
                    offset = - f * offsetAngle * 4.0
                    # 计算新的view matrix
                    # direction = self.view_matrix.col[2].xyz.normalized()
                    # new_offset = direction * offset
                    new_view_matrix = Matrix.Translation((-offset*1.5, 0, 0)) @ self.view_matrix
                    # new_view_matrix = self.view_matrix.copy()
                    # 计算新的projection matrix
                    new_projection_matrix = self.projection_matrix.copy()
                    new_projection_matrix[0][2] -= offset / (camera_size * (1440 / 2560)) / 2.5

                    print(f"fov={fov}, f={f}, near={near}, offsetAngle={offsetAngle}, offset={offset}")
                    print(f"第{idx + 1}个纹理，viewMatrix为{new_view_matrix},projectionMatrix为{new_projection_matrix}")
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
            padding = 0 # 每行有 16 字节的填充
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
        initialize_cv_window(self.display_x)
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
        self._timer = context.window_manager.event_timer_add((1/self.fps), window=context.window)  # 每 0.1 秒刷新一次

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


    # 在execute之前执行这个方法，用作初始化
    def invoke(self, context, event):
        # return self.execute(context), context.window_manager.invoke_props_dialog(self)
        return context.window_manager.invoke_props_dialog(self)

