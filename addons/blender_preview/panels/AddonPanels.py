import bpy

from ..config import __addon_name__
from ..operators.AddonOperators import LFDPreviewOperator
from ..operators.AddonOperators import LFDSaveOperator
from ..operators.AddonOperators import connectOperator
from ..operators.AddonOperators import QuiltSaveOperator
from ..operators.AddonOperators import FrustumOperator
from ..operators.AddonOperators import LFDRenderOperator
from ..operators.AddonOperators import QuiltRenderOperator
from ..operators.AddonOperators import QuiltSaveOperator1
from ..operators.AddonOperators import ConnectPlatform
# from ..operators.AddonOperators import RenderAnimation
from ..operators.AddonOperators import RenderImageSequenceToVideo
from ..operators.AddonOperators import RenderAnimation1
from ..operators.AddonOperators import ConnectVideoPlatform
from ....common.i18n.i18n import i18n
from ....common.types.framework import reg_order


class BasePanel(object):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LFD"

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return True

# Define the property in bpy.types.Scene instead of directly in the panel
def clip_near_property(self: bpy.types.Scene):
    return self.get("clip_near", 2.0)  # Default value if not set

def set_clip_near_property(self: bpy.types.Scene, value: float):
    self["clip_near"] = value

def clip_far_property(self: bpy.types.Scene):
    return self.get("clip_far", 15)  # Default value if not set

def set_clip_far_property(self: bpy.types.Scene, value: float):
    self["clip_far"] = value

def focal_plane_property(self: bpy.types.Scene):
    return self.get("focal_plane", 10)  # Default value if not set

def set_focal_plane_property(self: bpy.types.Scene, value: int):
    self["focal_plane"] = value

def x_axis_property(self: bpy.types.Scene):
    return self.get("x_axis", 2560)  # Default value if not set

def set_x_axis_property(self: bpy.types.Scene, value: int):
    self["x_axis"] = value



def frames_property(self: bpy.types.Scene):
    return self.get("frame_start", 0)  # Default value if not set

def set_frames_property(self: bpy.types.Scene, value: int):
    self["frame_start"] = value

def framee_property(self: bpy.types.Scene):
    return self.get("frame_end", 250)  # Default value if not set

def set_framee_property(self: bpy.types.Scene, value: int):
    self["frame_end"] = value

def fps_property(self: bpy.types.Scene):
    return self.get("fps", 10)  # Default value if not set

def set_fps_property(self: bpy.types.Scene, value: int):
    self["fps"] = value

# def frame_start_property(self: bpy.types.Scene):
#     return self.get("frame_start", 0)  # Default value if not set
#
# def set_frame_start_property(self: bpy.types.Scene, value: int):
#     self["frame_start"] = value
#
# def frame_end_property(self: bpy.types.Scene):
#     return self.get("frame_end", 250)  # Default value if not set
#
# def set_frame_end_property(self: bpy.types.Scene, value: int):
#     self["frame_end"] = value
#
# def fps_property(self: bpy.types.Scene):
#     return self.get("fps", 10)  # Default value if not set
#
# def set_fps_property(self: bpy.types.Scene, value: int):
#     self["fps"] = value
#
# Register the property on bpy.types.Scene

# bpy.types.Scene.frame_start = bpy.props.IntProperty(
#     name="Frame Start",
#     description="Adjust the start frame",
#     default=0,
#     get=frame_start_property,
#     set=set_frame_start_property
# )
#
# # Register the property on bpy.types.Scene
# bpy.types.Scene.frame_end = bpy.props.IntProperty(
#     name="Frame End",
#     description="Adjust the end frame",
#     default=250,
#     get=frame_end_property,
#     set=set_frame_end_property
# )
#
# # Register the property on bpy.types.Scene
# bpy.types.Scene.fps = bpy.props.IntProperty(
#     name="Video fps",
#     description="Adjust the fps",
#     default=10,
#     get=fps_property,
#     set=set_fps_property
# )


# Register the property on bpy.types.Scene
bpy.types.Scene.clip_near = bpy.props.FloatProperty(
    name="Near Clip",
    description="Adjust the near Clip",
    default=2.0,
    min=0.1,
    get=clip_near_property,
    set=set_clip_near_property
)

bpy.types.Scene.clip_far = bpy.props.FloatProperty(
    name="Far Clip",
    description="Adjust the far Clip",
    default=15,
    min=0.1,
    get=clip_far_property,
    set=set_clip_far_property
)

bpy.types.Scene.focal_plane = bpy.props.FloatProperty(
    name="Focal Plane",
    description="Adjust the focal plane",
    default=10,
    min=0.1,
    get=focal_plane_property,
    set=set_focal_plane_property
)

bpy.types.Scene.x_axis = bpy.props.IntProperty(
    name="x axis",
    description="Adjust the x axis",
    default=2560,
    get=x_axis_property,
    set=set_x_axis_property
)

bpy.types.Scene.frame_s = bpy.props.IntProperty(
    name="frame start",
    description="Adjust the start frame",
    default=0,
    get=frames_property,
    set=set_frames_property
)

bpy.types.Scene.frame_e = bpy.props.IntProperty(
    name="frame end",
    description="Adjust the end frame",
    default=250,
    get=framee_property,
    set=set_framee_property
)

bpy.types.Scene.fps = bpy.props.IntProperty(
    name="video fps",
    description="Adjust the fps",
    default=10,
    get=fps_property,
    set=set_fps_property
)

bpy.types.Scene.frame_s = bpy.props.IntProperty(
    name="frame start",
    description="Adjust the start frame",
    default=0,
    get=frames_property,
    set=set_frames_property
)

bpy.types.Scene.my_filepath = bpy.props.StringProperty(
    name="File Path",
    description="Select a file path",
    subtype='FILE_PATH',  # 使其为文件路径选择器
    default = "C:\\"
)


# @reg_order(0)
# class ExampleAddonPanel(BasePanel, bpy.types.Panel):
#     bl_label = "Example Addon Side Bar Panel"
#     bl_idname = "SCENE_PT_sample"
#
#     def draw(self, context: bpy.types.Context):
#         layout = self.layout
#
#
#         # layout.operator(ExampleOperator.bl_idname)
#
#     @classmethod
#     def poll(cls, context: bpy.types.Context):
#         return True


# This panel will be drawn after ExampleAddonPanel since it has a higher order value
# @reg_order(1)
class ExampleAddonPanel2(BasePanel, bpy.types.Panel):
    bl_label = "Light Field Rendering"
    bl_idname = "SCENE_PT_PREVIEW"

    def draw(self, context: bpy.types.Context):
        layout = self.layout

        row = layout.row(align=True)
        row.alignment = 'CENTER'
        row.label(text="Camera Settings")
        layout.prop(context.scene, "clip_near")
        layout.prop(context.scene, "clip_far")
        layout.prop(context.scene, "focal_plane")
        # layout.prop(context.scene, "frame_start")
        layout.operator(FrustumOperator.bl_idname)

        layout.separator()
        layout.prop(context.scene, "my_filepath")
        layout.separator()

        row = layout.row(align=True)
        row.alignment = 'CENTER'
        row.label(text="Platform Connection")
        layout.operator(connectOperator.bl_idname)
        layout.separator()

        row = layout.row(align=True)
        row.alignment = 'CENTER'
        row.label(text="Realtime Preview")
        # layout.prop(context.scene, "x_axis")
        layout.operator(LFDPreviewOperator.bl_idname)
        layout.operator(QuiltSaveOperator.bl_idname)
        layout.operator(LFDSaveOperator.bl_idname)

        layout.separator()
        row = layout.row(align=True)
        row.alignment = 'CENTER'
        row.label(text="Render")
        layout.operator(QuiltRenderOperator.bl_idname)
        layout.operator(QuiltSaveOperator1.bl_idname)
        layout.operator(ConnectPlatform.bl_idname)
        # layout.operator(RenderAnimation.bl_idname)

        layout.separator()
        layout.prop(context.scene, "frame_s")
        layout.prop(context.scene, "frame_e")
        # layout.prop(context.scene, "frame_start")
        # layout.prop(context.scene, "frame_end")
        layout.prop(context.scene, "fps")
        layout.operator(RenderAnimation1.bl_idname)
        layout.operator(RenderImageSequenceToVideo.bl_idname)
        layout.operator(ConnectVideoPlatform.bl_idname)
        # layout.operator(LFDRenderOperator.bl_idname)



