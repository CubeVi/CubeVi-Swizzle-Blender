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
common_dictionary = {
    "zh_CN": {
        # ("*", "translation"): "翻译",
    }
}

common_dictionary["zh_HANS"] = common_dictionary["zh_CN"]


# preprocess dictionary
def preprocess_dictionary(dictionary):
    for key in dictionary:
        invalid_items = {}
        for translate_key in dictionary[key]:
            if isinstance(translate_key, str):
                invalid_items[translate_key] = dictionary[key][translate_key]
        for invalid_item in invalid_items:
            translation = invalid_items[invalid_item]
            dictionary[key][("*", invalid_item)] = translation
            dictionary[key][("Operator", invalid_item)] = translation
            del dictionary[key][invalid_item]
    return dictionary
