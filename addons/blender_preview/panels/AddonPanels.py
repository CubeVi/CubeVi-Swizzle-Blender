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

def set_focal_plane_property(self: bpy.types.Scene, value: float):
    self["focal_plane"] = value

# Register the property on bpy.types.Scene
bpy.types.Scene.clip_near = bpy.props.FloatProperty(
    name="Near Clip",
    description="Adjust the near Clip",
    default=2.0,
    min=0.0,
    get=clip_near_property,
    set=set_clip_near_property
)

bpy.types.Scene.clip_far = bpy.props.FloatProperty(
    name="Far Clip",
    description="Adjust the far Clip",
    default=15,
    min=0.0,
    get=clip_far_property,
    set=set_clip_far_property
)

bpy.types.Scene.focal_plane = bpy.props.FloatProperty(
    name="Focal Plane",
    description="Adjust the focal plane",
    default=10,
    min=0.0,
    get=focal_plane_property,
    set=set_focal_plane_property
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
        layout.operator(FrustumOperator.bl_idname)
        layout.separator()

        row = layout.row(align=True)
        row.alignment = 'CENTER'
        row.label(text="Platform Connection")
        layout.operator(connectOperator.bl_idname)
        layout.separator()

        row = layout.row(align=True)
        row.alignment = 'CENTER'
        row.label(text="Realtime Preview")
        layout.operator(LFDPreviewOperator.bl_idname)
        layout.operator(QuiltSaveOperator.bl_idname)
        layout.operator(LFDSaveOperator.bl_idname)

        layout.separator()
        row = layout.row(align=True)
        row.alignment = 'CENTER'
        row.label(text="Render")
        layout.operator(QuiltRenderOperator.bl_idname)
        layout.operator(QuiltSaveOperator1.bl_idname)
        # layout.operator(LFDRenderOperator.bl_idname)




