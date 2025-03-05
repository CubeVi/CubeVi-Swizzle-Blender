from common.i18n.dictionary import preprocess_dictionary

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

        ("*","Realtime Preview"):"预览界面",
        ("*","x axis"):"预览x轴分辨率",
        ("*","Adjust the x axis"):"修改预览窗口的x轴位置",
        ("Operator","Realtime LightField Preview"):"实时光场预览",
        ("*","Start LightField Preview, Esc to exit"):"开启光场预览，单击Esc取消",

        ("Operator","Save Quilt Preview Picture"):"保存宫格预览图片",
        ("*","Save the preview quilt picture"):"将宫格预览图片保存到指定路径",

        ("Operator","Save LFD Preview Picture"):"保存光场预览图片",
        ("*","Save the preview lightfield picture"):"将光场预览图片保存到指定路径",



        ("Operator","Save Multiview Render Picture"):"保存视点渲染图片",
        ("*","Save Multiview Render image"):"保存视点渲染图片到指定路径",

        ("Operator", "Synthesize Quilt Picture"): "合成宫格图片",
        ("*","Save Quilt image"):"从视点渲染图片合成宫格图片到指定路径",

        ("Operator","Upload to the platform"):"上传宫格图到3D图库",
        ("*","Upload to the platform"):"将宫格图上传到3D图库",


        ("*", "Render Animation"): "渲染宫格图序列",

        ("Operator", "Render Image Sequence to Video"): "将宫格图序列合成为视频",
        ("*", "Render Image Sequence to Video"): "按照视频帧率合成视频",


        ("*","frame start"):"开始帧",
        ("*","Adjust the start frame"):"调整开始帧",
        ("*","frame end"):"结束帧",
        ("*", "Adjust the end frame"): "调整结束帧",
        ("*","video fps"):"视频帧率",
        ("*", "Adjust the fps"): "调整视频帧率",

        ("*", "Upload video to the platform"): "上传视频到平台",
        ("Operator", "Upload video to the platform"): "上传视频到平台",

    }
}

dictionary = preprocess_dictionary(dictionary)

dictionary["zh_HANS"] = dictionary["zh_CN"]
