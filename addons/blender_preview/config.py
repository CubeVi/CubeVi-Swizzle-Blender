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
from ...common.types.framework import is_extension

# https://docs.blender.org/manual/en/latest/advanced/extensions/addons.html#extensions-and-namespace
# This is the unique package name of the addon, it is different from the add-on name in bl_info.
# This name is used to identify the add-on in python code. It should also be the same to the package name of the add-on.
__addon_name__ = ".".join(__package__.split(".")[0:3]) if is_extension() else __package__.split(".")[0]
