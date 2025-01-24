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


def is_extension():
    # Blender extension package starts with "bl_ext."
    # https://docs.blender.org/manual/en/latest/advanced/extensions/addons.html#extensions-and-namespace
    return str(__package__).startswith("bl_ext.")


# This is a helper base class for you to expand native Blender UI
class ExpandableUi:
    # ID of the target panel.menu to be expanded to
    target_id: str
    # mode of expansion, either "PREPEND" or "APPEND"
    expand_mode: str = "APPEND"

    def draw(self, context: bpy.types.Context):
        raise NotImplementedError("draw method must be implemented")


def reg_order(order_value: int):
    """
    This decorator is used to specify the relative registration order of a class. The class with lower order value will
    be registered first, class without this decorator will be registered last.
    Notice it still respect the dependencies between classes. Only classes with no dependencies relationship will be
    sorted by order value.
    Notice, for UI classes, updating the order won't take effect in real-time during testing, you need to also update
    the bl_idname of the class to let Blender clean up the drawing cache.
    这个装饰器用于指定类的相对注册顺序。具有较低顺序值的类将首先注册，没有此装饰器的类将最后注册。
    请注意，它仍然尊重类之间的依赖关系。只有没有依赖关系的类才会按顺序值排序。
    请注意，对于UI类，更新顺序不会在测试期间实时生效，您还需要更新类的bl_idname，以便Blender清除绘图缓存。
    """

    def class_decorator(cls):
        cls._reg_order = order_value
        return cls

    return class_decorator
