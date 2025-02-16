import bpy
import importlib
import subprocess
import sys
# import pkg_resources
from importlib import metadata

from .config import __addon_name__
from .i18n.dictionary import dictionary
from ...common.class_loader import auto_load
from ...common.class_loader.auto_load import add_properties, remove_properties
from ...common.i18n.dictionary import common_dictionary
from ...common.i18n.i18n import load_dictionary

# Add-on info
bl_info = {
    "name": "blender_preview",
    "author": "CubeVi",
    "blender": (4, 2, 1),
    "version": (1, 0, 1),
    "description": "Blender Preview for LFD",
    "doc_url": "",
    "tracker_url": "",
    "support": "COMMUNITY",
    "category": "3D View"
}

_addon_properties = {}


# You may declare properties like following, framework will automatically add and remove them.
# Do not define your own property group class in the __init__.py file. Define it in a separate file and import it here.
# 注意不要在__init__.py文件中自定义PropertyGroup类。请在单独的文件中定义它们并在此处导入。
# _addon_properties = {
#     bpy.types.Scene: {
#         "property_name": bpy.props.StringProperty(name="property_name"),
#     },
# }

REQUIRED_LIBRARIES = [
    "opencv-python",
    "numpy",
    "Pillow",  # Pillow 替代 PIL
    "pywin32",
    "screeninfo"
    # 包含 win32file 和 win32pipe
]

def ensure_packages():
    """
    检查并安装所需的 Python 包。
    """
    python_executable = sys.executable  # 获取当前 Python 解释器路径



    for package in REQUIRED_LIBRARIES:
        package_name, _, required_version = package.partition("==")  # 分离包名和版本号
        try:
            # 检查包是否已安装以及版本是否匹配
            installed_version = metadata.version(package_name)
            if required_version and installed_version != required_version:
                print(f"{package_name}版本不匹配：已安装 {installed_version}，需要 {required_version}")
                raise metadata.PackageNotFoundError
            print(f"{package_name} 已安装，版本匹配 ({installed_version}).")
        except metadata.PackageNotFoundError:
            # 包未安装或版本不匹配时重新安装
            try:
                print(f"Installing {package}...")
                subprocess.check_call(
                    [python_executable, "-m", "pip", "install", package]
                )
                print(f"{package} installed successfully.")

                # 对于 pywin32，需要运行 post-install 脚本
                if package_name.lower() == "pywin32":
                    print("Running pywin32 post-install script...")
                    subprocess.check_call(
                        [python_executable, "-m", "pywin32_postinstall", "-install"]
                    )
                    requires_restart = True

            except Exception as e:
                print(f"Failed to install {package}: {e}")





def register():
    ensure_packages()
    print("All required packages are ensured.")
    # Register classes
    auto_load.init()
    auto_load.register()
    add_properties(_addon_properties)

    # Internationalization
    load_dictionary(dictionary)
    bpy.app.translations.register(__addon_name__, common_dictionary)
    # set_1()
    print("{} addon is installed.".format(__addon_name__))


def unregister():
    # Internationalization
    bpy.app.translations.unregister(__addon_name__)
    # unRegister classes
    auto_load.unregister()
    remove_properties(_addon_properties)
    print("{} addon is uninstalled.".format(__addon_name__))
