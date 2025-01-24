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
from blender_preview.common.i18n.dictionary import preprocess_dictionary

dictionary = {
    "zh_CN": {
        ("*", "Real Time Rendering"): "实时渲染",
        ("*", "Light Field Rendering"): "光场渲染",
        ("Operator", "connect"): "设备连接",
        ("Operator", "Lightfield rendering"): "实时渲染预览",
        ("Operator", "Save LFD Picture"): "保存光场图片"
    }
}

dictionary = preprocess_dictionary(dictionary)

dictionary["zh_HANS"] = dictionary["zh_CN"]
