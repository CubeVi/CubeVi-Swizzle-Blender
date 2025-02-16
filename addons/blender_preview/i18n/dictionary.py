from blender_preview.common.i18n.dictionary import preprocess_dictionary

dictionary = {
    "zh_CN": {
        ("*","Light Field Rendering"):"光场渲染",
        ("*","Camera Settings"):"相机设置",
        ("*","Near Clip"):"近剪裁面",
        ("*","Adjust the near Clip"):"调整近剪裁面",
        ("*","Far Clip"):"远剪裁面",
        ("*","Adjust the far Clip"):"调整远剪裁面",
        ("*","Focal Plane"):"焦平面",
        ("*","Adjust the focal plane"):"调整焦平面",
        ("Operator","Show Frustum"):"显示视锥",
        ("*","Show the camera frustum"):"显示/关闭视锥",

        ("*","Platform Connection"):"平台连接",
        ("Operator","Connect"):"连接",
        ("*","Connect to the device"):"连接平台及设备",

        ("*","Realtime Preview"):"实时预览",
        ("Operator","Realtime LightField Preview"):"实时光场预览",
        ("*","Start LightField Preview, Esc to exit"):"开启光场预览，单击Esc取消",
        ("Operator","Save Quilt Preview Picture"):"保存宫格预览图片",
        ("*","Save the preview quilt picture"):"将宫格预览图片保存到指定路径",
        ("Operator","Save LFD Preview Picture"):"保存光场预览图片",
        ("*","Save the preview lightfield picture"):"将光场预览图片保存到指定路径",

        ("Operator","Save Multiview Render Picture"):"保存视点渲染图片",
        ("*","Save Multiview Render image"):"保存视点渲染图片到指定路径",
        ("Operator","Synthetize Quilt Picture"):"合成宫格图片",
        ("*","Save Quilt image"):"从视点渲染图片合成宫格图片到指定路径",

    }
}

dictionary = preprocess_dictionary(dictionary)

dictionary["zh_HANS"] = dictionary["zh_CN"]
