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

from .addons.blender_preview import register as addon_register, unregister as addon_unregister

bl_info = {
    "name": 'blender_preview',
    "author": 'OpenStageAI',
    "blender": (4, 2, 1),
    "version": (1, 0, 0),
    "description": 'Blender Preview for LFD',
    "doc_url": '',
    "tracker_url": '',
    "support": 'COMMUNITY',
    "category": '3D View'
}

def register():
    addon_register()

def unregister():
    addon_unregister()

    