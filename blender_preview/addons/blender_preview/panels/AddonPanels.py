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

from ..config import __addon_name__
from ..operators.AddonOperators import LFDPreviewOperator
from ..operators.AddonOperators import LFDSaveOperator
from ..operators.AddonOperators import connectOperator
from ....common.i18n.i18n import i18n
from ....common.types.framework import reg_order


class BasePanel(object):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LFD"

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return True


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
@reg_order(1)
class ExampleAddonPanel2(BasePanel, bpy.types.Panel):
    bl_label = "Light Field Rendering"
    bl_idname = "SCENE_PT_PREVIEW"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        row = layout.row(align=True)
        row.alignment = 'CENTER'
        row.label(text="Real Time Rendering")
        layout.operator(connectOperator.bl_idname)
        layout.operator(LFDPreviewOperator.bl_idname)
        layout.operator(LFDSaveOperator.bl_idname)



