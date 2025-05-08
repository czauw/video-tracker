import cv2
import numpy as np
from collections import deque
import time
import glob
import os

# --- 全局配置 ---
VIDEO_PATH = "target.mp4"  # 视频文件路径
TEMPLATE_PATTERN_JPG = "template*.jpg"  # JPG 模板文件匹配模式
TEMPLATE_PATTERN_PNG = "template*.png"  # PNG 模板文件匹配模式
OUTPUT_VIDEO_PATH = "result.mp4"  # 输出视频文件路径

# --- 颜色设置 ---
BOX_COLOR_TRACKING = (0, 255, 0)  # 跟踪时的框颜色 (绿色)
BOX_COLOR_REDETECTION = (0, 165, 255)  # 重新检测时的框颜色 (橙色)
BOX_COLOR_LOST = (0, 0, 255)  # 丢失目标时的框颜色 (红色)
TRAJECTORY_COLOR = (0, 255, 255)  # 轨迹颜色 (青色)

# --- 参数配置 ---
NCC_BASE_THRESHOLD = 0.45  # NCC 基础阈值，用于判断模板匹配是否成功
TEMPLATE_UPDATE_CONF_THRESHOLD = 0.80  # 模板更新的置信度阈值，当匹配得分高于此值时更新模板
MAX_TEMPLATES = 10  # 存储的最大模板数量 (用于自适应模板更新)
MAX_LOST_FRAMES = 10  # 目标丢失的最大帧数，超过此帧数则进入重新检测模式
RE_DETECTION_FRAMES = 20  # 重新检测模式持续的最大帧数
GLOBAL_SEARCH_SCALE_FACTOR = 3.0  # 全局搜索时，搜索区域相对于目标大小的缩放因子
PYRAMID_MAX_LEVEL = 3  # 图像金字塔的最大层数
SEARCH_WINDOW_FACTOR = 2.5  # 搜索窗口相对于预测目标大小的缩放因子
MAX_DISPLAY_DIM = 1024  # 显示窗口的最大尺寸 (宽度或高度)

# --- ORB 特征检测器和匹配器 ---
orb = cv2.ORB_create(nfeatures=500)  # 创建 ORB 特征检测器，最大特征点数为 500
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 创建 BFMatcher (暴力匹配器)，使用汉明距离并进行交叉检查

# --- 卡尔曼滤波器初始化 ---
def init_kalman(bbox, dt):
    """
    初始化卡尔曼滤波器。

    参数:
        bbox (tuple): 目标的初始边界框 (x, y, w, h)。
        dt (float): 时间间隔，即帧之间的时间差。

    返回:
        cv2.KalmanFilter: 初始化后的卡尔曼滤波器对象。
    """
    kf = cv2.KalmanFilter(6, 4)  # 初始化卡尔曼滤波器，状态量为6 (cx, cy, w, h, vx, vy)，观测量为4 (cx, cy, w, h)
    # 状态转移矩阵 A: 定义当前状态如何根据前一个状态进行预测
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, dt, 0],  # cx_k = cx_{k-1} + vx_{k-1}*dt
        [0, 1, 0, 0, 0, dt],  # cy_k = cy_{k-1} + vy_{k-1}*dt
        [0, 0, 1, 0, 0, 0],  # w_k = w_{k-1}
        [0, 0, 0, 1, 0, 0],  # h_k = h_{k-1}
        [0, 0, 0, 0, 1, 0],  # vx_k = vx_{k-1}
        [0, 0, 0, 0, 0, 1]   # vy_k = vy_{k-1}
    ], np.float32)
    # 测量矩阵 H: 定义状态如何映射到测量值
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],  # z_cx = cx
        [0, 1, 0, 0, 0, 0],  # z_cy = cy
        [0, 0, 1, 0, 0, 0],  # z_w = w
        [0, 0, 0, 1, 0, 0]   # z_h = h
    ], np.float32)
    # 过程噪声协方差矩阵 Q: 表示状态转移的不确定性
    kf.processNoiseCov = np.diag([1, 1, 1, 1, 1e2, 1e2]).astype(np.float32) * 1e-3
    # 测量噪声协方差矩阵 R: 表示测量过程的不确定性
    kf.measurementNoiseCov = np.diag([1, 1, 10, 10]).astype(np.float32) * 1e-1
    x, y, w, h = bbox  # 解包边界框坐标
    cx, cy = x + w / 2, y + h / 2  # 计算中心点坐标
    # 设置初始状态 (后验估计)
    kf.statePost = np.array([cx, cy, w, h, 0, 0], np.float32).reshape(6, 1)
    # 设置初始误差协方差矩阵 (后验估计)
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1
    return kf


# --- 全局运动估计 ---
def estimate_global_affine(prev_gray, curr_gray, orb_detector, bf_matcher):
    """
    估计两帧之间的全局仿射变换。

    参数:
        prev_gray (np.ndarray): 前一帧的灰度图像。
        curr_gray (np.ndarray): 当前帧的灰度图像。
        orb_detector (cv2.ORB): ORB 特征检测器。
        bf_matcher (cv2.BFMatcher): BF 特征匹配器。

    返回:
        tuple: (M, mask) 其中 M 是仿射变换矩阵，mask 是内点掩码。如果失败则返回 (None, None)。
    """
    try:
        # 检测前一帧的关键点和描述子
        kp1, des1 = orb_detector.detectAndCompute(prev_gray, None)
        # 检测当前帧的关键点和描述子
        kp2, des2 = orb_detector.detectAndCompute(curr_gray, None)
        # 如果描述子为空或数量不足，则返回失败
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None, None
        # 进行特征匹配
        matches = bf_matcher.match(des1, des2)
        # 按距离对匹配结果进行排序
        matches = sorted(matches, key=lambda x: x.distance)
        # 如果匹配数量不足，则返回失败
        if len(matches) < 6: # 仿射变换至少需要3对点，RANSAC通常需要更多
            return None, None
        # 提取匹配点对的坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # 使用 RANSAC 估计部分仿射变换 (允许缩放、旋转、平移，但不允许剪切)
        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        # 如果变换矩阵为空或内点数量不足，则返回失败
        if M is None or mask is None or np.sum(mask) < 6:
            return None, None
        return M, mask.ravel() == 1 # 返回变换矩阵和布尔型内点掩码
    except (cv2.error, Exception): # 捕获 OpenCV 错误和其他异常
        return None, None

# --- 自定义 NCC 匹配 (替代 cv2.matchTemplate with TM_CCOEFF_NORMED) ---
def custom_ncc_match(template_img, search_img):
    """
    执行归一化互相关 (NCC) 匹配。

    参数:
        template_img (np.ndarray): 模板图像 (灰度图, float32)。
        search_img (np.ndarray): 搜索图像 (灰度图, float32)。

    返回:
        tuple: (max_score, max_loc) 其中 max_score 是最高的 NCC 分数，
               max_loc 是最佳匹配的左上角 (x, y) 坐标。
               出错或无匹配时返回 (-1.0, (0,0))。
    """
    # 检查输入图像是否有效
    if template_img is None or search_img is None or \
       template_img.ndim != 2 or search_img.ndim != 2:
        # print("Debug: Invalid input images for NCC (None or not 2D)") # 调试信息：NCC的输入图像无效（None或非二维）
        return -1.0, (0, 0)

    th, tw = template_img.shape # 获取模板图像的高度和宽度
    sh, sw = search_img.shape # 获取搜索图像的高度和宽度

    # 检查图像尺寸是否为零
    if th == 0 or tw == 0 or sh == 0 or sw == 0:
        # print(f"Debug: Zero dimensions for template ({th}x{tw}) or search ({sh}x{sw})") # 调试信息：模板或搜索图像的尺寸为零
        return -1.0, (0,0)

    # 检查模板是否大于搜索图像
    if th > sh or tw > sw:
        # print(f"Debug: Template ({th}x{tw}) larger than search image ({sh}x{sw})") # 调试信息：模板大于搜索图像
        return -1.0, (0, 0)

    # 确保图像为 float32 类型以进行计算
    template_img = template_img.astype(np.float32)
    search_img = search_img.astype(np.float32)

    # 计算模板的均值和标准差
    template_mean = np.mean(template_img)
    template_std = np.std(template_img)

    # 避免除以零或接近零的标准差
    if template_std < 1e-6:
        # print("Debug: Template standard deviation is too small.") # 调试信息：模板标准差太小
        # 如果模板是平坦的，相关性计算会很棘手。
        # cv2.matchTemplate 通过返回低分来处理这种情况。
        # 我们可以做同样的事情，或者将其视为没有意义的相关性。
        return 0.0, (0,0) # 或者，如果倾向于表示“无匹配”，则返回 -1.0

    # 归一化模板: (T - T_mean)
    # 对于 NCC 公式，更准确地说，我们稍后需要在分母中使用 sum((T - T_mean)^2)
    # 或者，通过 (T - T_mean) / sqrt(sum((T-T_mean)^2)) 进行归一化
    # 现在，我们使用 (T - T_mean) 的形式，并针对每个窗口处理归一化。
    template_norm = template_img - template_mean

    # NCC 模板部分的分母: sqrt(sum((T - T_mean)^2))
    # 如果使用样本标准差，这等效于 template_std * sqrt(模板中的像素数 - 1)
    # 或者更直接地：
    template_denom = np.sqrt(np.sum(template_norm**2))
    if template_denom < 1e-6: # 避免分母过小
        # print("Debug: Template denominator (sqrt sum of squares) is too small.") # 调试信息：模板分母（平方和的平方根）太小
        return 0.0, (0,0)

    # 创建一个结果映射来存储 NCC 分数
    result_h = sh - th + 1
    result_w = sw - tw + 1
    # 此情况应由 th > sh 或 tw > sw 捕获，但最好再次检查
    if result_h <= 0 or result_w <= 0:
        # print("Debug: Result map has non-positive dimensions.") # 调试信息：结果映射的维度非正
        return -1.0, (0,0)

    ncc_scores = np.zeros((result_h, result_w), dtype=np.float32) # 初始化 NCC 分数矩阵

    # 在搜索图像上滑动模板
    for y in range(result_h):
        for x in range(result_w):
            # 从搜索图像中提取当前窗口
            window = search_img[y:y+th, x:x+tw]

            # 计算窗口的均值
            window_mean = np.mean(window)
            # window_std = np.std(window) # 在此 NCC 公式变体中不直接使用

            # if window_std < 1e-6: # 如果窗口是平坦的，避免除以零
            #     ncc_scores[y, x] = 0.0 # 或其他表示低相关性的指标
            #     continue

            # 归一化窗口: (I - I_mean)
            window_norm = window - window_mean

            # NCC 窗口部分的分母: sqrt(sum((I - I_mean)^2))
            window_denom = np.sqrt(np.sum(window_norm**2))

            if window_denom < 1e-6: # 避免窗口分母过小
                ncc_scores[y, x] = 0.0
                continue

            # 计算 NCC 的分子: sum((T - T_mean) * (I - I_mean))
            numerator = np.sum(template_norm * window_norm)

            # 计算 NCC 分数
            # NCC = Numerator / (Template_Denom * Window_Denom)
            score = numerator / (template_denom * window_denom)
            ncc_scores[y, x] = score

    # 找到最大分数及其位置
    # np.unravel_index 给出 (row, col)，即 (y, x)
    max_val = np.max(ncc_scores)
    max_loc_flat = np.argmax(ncc_scores) # 获取扁平化数组中最大值的索引
    max_loc_y, max_loc_x = np.unravel_index(max_loc_flat, ncc_scores.shape) # 将扁平化索引转换为二维索引

    # 由于潜在的浮点不准确性，确保分数在 [-1, 1] 范围内
    max_val = np.clip(max_val, -1.0, 1.0)

    return float(max_val), (int(max_loc_x), int(max_loc_y))


# --- NCC 匹配 (使用自定义函数) ---
def match_ncc(template_img, search_img):
    """
    使用自定义的 NCC 函数进行模板匹配。

    参数:
        template_img (np.ndarray): 模板图像。
        search_img (np.ndarray): 搜索图像。

    返回:
        tuple: (max_score, max_loc) NCC 最高得分和对应位置。
    """
    try:
        # 检查输入图像的有效性
        if template_img is None or search_img is None or \
                template_img.shape[0] > search_img.shape[0] or \
                template_img.shape[1] > search_img.shape[1] or \
                template_img.shape[0] == 0 or template_img.shape[1] == 0 or \
                search_img.shape[0] == 0 or search_img.shape[1] == 0:
            return -1.0, (0, 0) # 如果无效则返回

        # 使用自定义 NCC 函数
        maxv, maxl = custom_ncc_match(template_img, search_img)
        # print(f"Custom NCC: score={maxv}, loc={maxl} for template {template_img.shape} in search {search_img.shape}") # 调试信息

        # 如果你想与 OpenCV 的版本进行比较以进行调试：
        # res_cv = cv2.matchTemplate(search_img.astype(np.float32), template_img.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        # _, maxv_cv, _, maxl_cv = cv2.minMaxLoc(res_cv)
        # print(f"OpenCV NCC: score={maxv_cv}, loc={maxl_cv}") # 调试信息

        return maxv, maxl
    except Exception as e: # 捕获来自 custom_ncc_match 或类型转换的任何其他错误
        print(f"Error in match_ncc (custom): {e}") # 打印自定义NCC匹配中的错误信息
        return -1.0, (0, 0)


# --- 构建图像金字塔 ---
def build_pyramid(image, max_level):
    """
    构建图像金字塔。

    参数:
        image (np.ndarray): 输入图像。
        max_level (int): 金字塔的最大层数。

    返回:
        list: 包含金字塔各层图像的列表，第一层是原始图像。
    """
    pyramid = [image] # 初始化金字塔列表，第一层为原始图像
    for _ in range(max_level): # 迭代构建金字塔
        try:
            # 在 pyrDown 之前确保图像不要太小
            if image is None or image.shape[0] < 10 or image.shape[1] < 10: # 增加了最小尺寸限制
                break # 如果图像过小，则停止构建
            downscaled_image = cv2.pyrDown(image) # 对图像进行下采样
            # 检查下采样后的图像是否有效
            if downscaled_image is None or downscaled_image.shape[0] < 5 or downscaled_image.shape[1] < 5:
                break # 如果下采样图像过小或无效，则停止
            pyramid.append(downscaled_image) # 将下采样图像添加到金字塔列表
            image = downscaled_image # 更新图像以进行下一次迭代
        except cv2.error as e: # 捕获 OpenCV 错误
            # print(f"cv2.error in build_pyramid pyrDown: {e}") # 调试信息：构建金字塔 pyrDown 时发生 cv2.error
            break
        except Exception as e: # 捕获其他通用错误
            # print(f"General error in build_pyramid pyrDown: {e}") # 调试信息：构建金字塔 pyrDown 时发生通用错误
            break
    return pyramid


# --- 帧缩放工具 ---
def resize_frame_if_needed(frame, max_dim):
    """
    如果帧的尺寸超过最大限制，则按比例缩放帧。

    参数:
        frame (np.ndarray): 输入帧。
        max_dim (int): 允许的最大尺寸 (宽度或高度)。

    返回:
        tuple: (resized_frame, scale_factor) 缩放后的帧和缩放比例。
    """
    h_orig, w_orig = frame.shape[:2] # 获取原始帧的高度和宽度
    scale = 1.0 # 初始化缩放比例
    if max(h_orig, w_orig) > max_dim: # 如果原始帧的最大尺寸超过限制
        scale = max_dim / max(h_orig, w_orig) # 计算缩放比例
        # 使用 INTER_AREA 插值进行缩放，适用于缩小图像
        frame_resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return frame_resized, scale
    return frame, scale # 如果不需要缩放，则返回原始帧和比例 1.0


# --- 加载外部模板 ---
def load_external_templates():
    """
    从文件中加载外部模板图像。

    返回:
        list: 包含加载的模板信息 (数据、名称、形状) 的字典列表。
    """
    # 查找匹配模式的 JPG 和 PNG 文件
    template_files = glob.glob(TEMPLATE_PATTERN_JPG) + glob.glob(TEMPLATE_PATTERN_PNG)
    loaded_templates = [] # 初始化加载的模板列表
    for f_path in template_files: # 遍历找到的模板文件
        try:
            tmpl_color = cv2.imread(f_path, cv2.IMREAD_COLOR) # 以彩色模式读取模板图像
            if tmpl_color is not None and tmpl_color.size > 0: # 检查图像是否成功加载且不为空
                tmpl_gray = cv2.cvtColor(tmpl_color, cv2.COLOR_BGR2GRAY) # 将模板转换为灰度图
                # tmpl_processed = cv2.equalizeHist(tmpl_gray) # 考虑直方图均衡化是否总是最佳选择
                tmpl_processed = tmpl_gray # 使用原始灰度图可能对自定义NCC更稳定
                loaded_templates.append(
                    {"data": tmpl_processed, "name": os.path.basename(f_path), "shape": tmpl_processed.shape})
                print(
                    f"Successfully loaded template: {os.path.basename(f_path)} with shape {tmpl_processed.shape}") # 打印成功加载信息
            else:
                print(f"Warning: Could not load or empty template: {f_path}") # 打印加载失败或空模板警告
        except Exception as e:
            print(f"Error loading template {f_path}: {e}") # 打印加载模板时的错误
    return loaded_templates


# --- 在给定帧上匹配外部模板 ---
def match_external_templates_on_frame(frame_gray_processed, external_templates_list):
    """
    在给定的处理过的灰度帧上匹配所有外部模板。

    参数:
        frame_gray_processed (np.ndarray): 处理过的灰度帧。
        external_templates_list (list): 包含外部模板信息的列表。

    返回:
        tuple: (best_template_data, best_bbox, best_score, best_template_name)
               最佳匹配的模板数据、在帧上的边界框、匹配分数和模板名称。
    """
    best_overall_score = -1.0 # 初始化最佳总分
    best_bbox_on_frame = None # 初始化最佳边界框
    best_template_data_matched = None # 初始化最佳匹配的模板数据
    best_template_name_matched = "None" # 初始化最佳匹配的模板名称
    frame_h, frame_w = frame_gray_processed.shape # 获取帧的高度和宽度

    for ext_tmpl_item in external_templates_list: # 遍历外部模板列表
        ext_tmpl_data = ext_tmpl_item["data"] # 获取模板数据
        ext_tmpl_name = ext_tmpl_item["name"] # 获取模板名称
        tmpl_h, tmpl_w = ext_tmpl_data.shape # 获取模板的高度和宽度

        if tmpl_h == 0 or tmpl_w == 0: # 如果模板尺寸为零，则跳过
            continue

        pyramid_frame = build_pyramid(frame_gray_processed, PYRAMID_MAX_LEVEL) # 构建帧的图像金字塔
        current_best_score_for_this_template = -1.0 # 初始化当前模板的最佳分数
        current_best_bbox_for_this_template = None # 初始化当前模板的最佳边界框

        for level, frame_level_img in enumerate(pyramid_frame): # 遍历金字塔的每一层
            scale_py = 2 ** level # 计算当前金字塔层级的缩放比例
            # 确保模板不大于当前金字塔层级的图像
            if ext_tmpl_data.shape[0] > frame_level_img.shape[0] or \
               ext_tmpl_data.shape[1] > frame_level_img.shape[1]:
                continue # 如果模板过大，则跳到下一层

            score, loc = match_ncc(ext_tmpl_data, frame_level_img) # 使用自定义或原始NCC进行匹配

            if score > current_best_score_for_this_template: # 如果当前分数更好
                current_best_score_for_this_template = score # 更新当前模板的最佳分数
                # loc 是相对于 frame_level_img 的
                abs_x_on_level = loc[0] # 匹配位置在当前层级图像上的x坐标
                abs_y_on_level = loc[1] # 匹配位置在当前层级图像上的y坐标

                # 缩放回原始 frame_gray_processed 坐标
                final_x = int(abs_x_on_level * scale_py)
                final_y = int(abs_y_on_level * scale_py)
                # Bbox 大小是模板的原始大小
                final_w = tmpl_w
                final_h = tmpl_h

                # 确保 bbox 在帧边界内
                final_x = max(0, final_x)
                final_y = max(0, final_y)
                final_w = min(frame_w - final_x, final_w) # 如果超出边界，调整宽度
                final_h = min(frame_h - final_y, final_h) # 如果超出边界，调整高度

                if final_w > 0 and final_h > 0: # 确保调整后的宽高有效
                    current_best_bbox_for_this_template = (final_x, final_y, final_w, final_h)

        if current_best_score_for_this_template > best_overall_score: # 如果当前模板的最佳分数优于总最佳分数
            best_overall_score = current_best_score_for_this_template # 更新总最佳分数
            best_bbox_on_frame = current_best_bbox_for_this_template # 更新总最佳边界框
            best_template_data_matched = ext_tmpl_data # 更新最佳匹配的模板数据
            best_template_name_matched = ext_tmpl_name # 更新最佳匹配的模板名称

    return best_template_data_matched, best_bbox_on_frame, best_overall_score, best_template_name_matched


# --- 主程序 ---
def main():
    global OUTPUT_VIDEO_PATH # 声明 OUTPUT_VIDEO_PATH 为全局变量，以便在函数内修改

    external_templates = load_external_templates() # 加载外部模板
    if not external_templates: # 如果没有加载到外部模板
        print(
            "Warning: No external templates found or loaded. Proceeding with manual ROI selection only for initialization.") # 打印警告

    cap_initial_select = cv2.VideoCapture(VIDEO_PATH) # 打开视频文件用于初始帧选择
    if not cap_initial_select.isOpened(): # 检查视频是否成功打开
        print(f"Error: Could not open video: {VIDEO_PATH}") # 打印错误信息
        return # 退出程序

    selected_frame_for_roi_raw = None # 用于存储用户选择的原始帧 (用于ROI选择)
    frame_idx_selection = 0 # 帧选择界面的当前帧索引
    final_selected_frame_idx = -1 # 用户最终选择的帧的索引
    selection_window_name = "Select Initial Frame (N: Next, S: Select, Q: Quit)" # 帧选择窗口标题

    while True: # 循环读取视频帧供用户选择
        ret, frame_capture = cap_initial_select.read() # 读取一帧
        if not ret: # 如果读取失败 (视频结束或错误)
            print("Error: Could not read frame for selection or video ended before selection.") # 打印错误信息
            cap_initial_select.release() # 释放视频捕获对象
            cv2.destroyAllWindows() # 关闭所有OpenCV窗口
            return # 退出程序

        current_frame_raw_selection = frame_capture.copy() # 复制当前捕获的帧
        # 缩放帧以便于显示
        frame_display_selection, _ = resize_frame_if_needed(current_frame_raw_selection, MAX_DISPLAY_DIM)

        # 在显示的帧上绘制提示文字
        cv2.putText(frame_display_selection, f"Frame: {frame_idx_selection}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_display_selection, "N: Next, S: Select, Q: Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(selection_window_name, frame_display_selection) # 显示帧

        key = cv2.waitKey(0) & 0xFF # 等待用户按键 (0表示无限等待)

        if key == ord('n') or key == ord('N'): # 如果按下 'n' 或 'N'
            frame_idx_selection += 1 # 移动到下一帧
            continue
        elif key == ord('s') or key == ord('S'): # 如果按下 's' 或 'S'
            selected_frame_for_roi_raw = current_frame_raw_selection # 存储选定的原始帧
            final_selected_frame_idx = frame_idx_selection # 存储选定的帧索引
            print(f"Selected frame index {final_selected_frame_idx} for ROI selection.") # 打印选择信息
            break # 跳出选择循环
        elif key == ord('q') or key == ord('Q') or key == 27:  # 如果按下 'q', 'Q' 或 ESC 键
            print("User quit during frame selection. Exiting.") # 打印用户退出信息
            cap_initial_select.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow(selection_window_name) # 关闭帧选择窗口
    cap_initial_select.release() # 释放视频捕获对象

    if selected_frame_for_roi_raw is None: # 如果没有选择任何帧
        print("Error: No frame was selected for ROI. Exiting.") # 打印错误信息
        return # 退出程序

    # 对选定的帧进行缩放以进行ROI选择和后续处理
    frame_init_scaled, global_scale_factor = resize_frame_if_needed(selected_frame_for_roi_raw, MAX_DISPLAY_DIM)
    if global_scale_factor != 1.0: # 如果帧被缩放了
        print(
            f"Initial frame will be processed at scale: {global_scale_factor:.2f}. New dims: {frame_init_scaled.shape[:2]}") # 打印缩放信息

    roi_selection_window_title = "Select Target ROI (Press Enter/Space to Confirm, C to Cancel)" # ROI选择窗口标题
    manual_roi = cv2.selectROI(roi_selection_window_title, frame_init_scaled, False) # 让用户在缩放后的帧上选择ROI
    cv2.destroyWindow(roi_selection_window_title) # 关闭ROI选择窗口

    if manual_roi == (0, 0, 0, 0) or manual_roi[2] <= 0 or manual_roi[3] <= 0: # 检查ROI是否有效
        print("No valid manual ROI selected. Exiting.") # 打印无效ROI信息
        return # 退出程序

    x_m_roi, y_m_roi, w_m_roi, h_m_roi = manual_roi # 解包手动选择的ROI坐标 (在缩放后的帧上)
    print(f"Manual ROI (on scaled frame): x={x_m_roi}, y={y_m_roi}, w={w_m_roi}, h={h_m_roi}") # 打印ROI信息

    frame_init_gray_full = cv2.cvtColor(frame_init_scaled, cv2.COLOR_BGR2GRAY) # 将缩放后的初始帧转换为灰度图
    # frame_init_gray_processed = cv2.equalizeHist(frame_init_gray_full) # 考虑是否需要直方图均衡化
    frame_init_gray_processed = frame_init_gray_full # 使用原始灰度图

    # 从处理过的灰度初始帧中提取手动选择的ROI作为初始模板
    init_tmpl_manual = frame_init_gray_processed[y_m_roi: y_m_roi + h_m_roi, x_m_roi: x_m_roi + w_m_roi].copy()
    if init_tmpl_manual.size == 0: # 检查模板是否为空
        print("Error: Manual initial template is empty. Please select a valid region. Exiting.") # 打印错误
        return
    print(f"Manual initial template size: {init_tmpl_manual.shape}") # 打印手动初始模板尺寸

    initial_bbox = manual_roi # 初始边界框即为手动选择的ROI (在缩放帧的坐标系下)
    init_tmpl = init_tmpl_manual # 初始模板

    templates = deque(maxlen=MAX_TEMPLATES) # 初始化一个双端队列用于存储自适应模板

    if external_templates: # 如果存在外部模板
        print("Matching external templates on the selected frame...") # 提示正在匹配外部模板
        # 在选定的初始帧上匹配外部模板
        ext_match_data, ext_match_bbox, ext_match_score, ext_match_name = \
            match_external_templates_on_frame(frame_init_gray_processed, external_templates)

        if ext_match_data is not None and ext_match_score >= NCC_BASE_THRESHOLD: # 如果找到强匹配
            print(f"Strong match found for external template '{ext_match_name}' "
                  f"on selected frame. Score: {ext_match_score:.2f}. Bbox: {ext_match_bbox}")
            templates.append(ext_match_data.copy()) # 将匹配到的外部模板添加到自适应模板池
            print(f"Added template '{ext_match_name}' to adaptive template pool.")
        else:
            print("No strong match found for any external template on the selected frame, or templates list is empty.")

    cap_tracking = cv2.VideoCapture(VIDEO_PATH) # 重新打开视频文件用于跟踪
    if not cap_tracking.isOpened(): # 检查视频是否成功打开
        print(f"Error: Could not re-open video for tracking: {VIDEO_PATH}")
        return

    # 跳过帧直到用户选择的起始帧的下一帧
    print(f"Skipping to frame {final_selected_frame_idx + 1} for tracking start...")
    for i in range(final_selected_frame_idx + 1): # final_selected_frame_idx 是选择的帧，跟踪从其后开始
        ret_skip, _ = cap_tracking.read() # 读取并丢弃帧
        if not ret_skip: # 如果在跳帧过程中视频结束
            print(f"Error: Video ended while skipping frames to tracking start point (tried to skip to frame {i+1}).")
            cap_tracking.release()
            return

    print("Reached tracking start frame.") # 到达跟踪起始帧

    frame_h_out, frame_w_out = frame_init_scaled.shape[:2] # 获取输出视频的帧高度和宽度 (基于缩放后的初始帧)
    fps_video = cap_tracking.get(cv2.CAP_PROP_FPS) # 获取视频的FPS
    if fps_video <= 0: fps_video = 30 # 如果FPS无效，则默认为30

    output_dir = os.path.dirname(OUTPUT_VIDEO_PATH) # 获取输出视频的目录
    if output_dir and not os.path.exists(output_dir): # 如果目录非空且不存在
        os.makedirs(output_dir); # 创建输出目录
        print(f"Created output directory: {output_dir}")

    video_writer = None # 初始化视频写入对象
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 定义 MP4 编码器
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_video, (frame_w_out, frame_h_out)) # 创建 VideoWriter
        if not video_writer.isOpened(): # 如果 MP4 写入器打开失败
            print(f"Error: VideoWriter (mp4v) failed for {OUTPUT_VIDEO_PATH}. Trying XVID (.avi)...") # 尝试 XVID AVI 格式
            avi_output_path = OUTPUT_VIDEO_PATH.rsplit('.', 1)[0] + ".avi" # 生成 AVI 文件名
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # 定义 XVID 编码器
            video_writer = cv2.VideoWriter(avi_output_path, fourcc, fps_video, (frame_w_out, frame_h_out))
            if video_writer.isOpened(): # 如果 AVI 写入器成功打开
                print(f"Switched to AVI output: {avi_output_path}")
                OUTPUT_VIDEO_PATH = avi_output_path # 更新输出路径
            else:
                print(f"Error: Fallback VideoWriter (XVID) also failed. Video will not be saved.")
                video_writer = None # 标记 VideoWriter 为 None
    except Exception as e: # 捕获初始化 VideoWriter 时的其他异常
        print(f"Exception initializing VideoWriter: {e}. Video will not be saved.")
        video_writer = None

    prev_gray_processed = frame_init_gray_processed.copy() # 将初始处理过的灰度帧作为前一帧

    last_time = time.time() # 记录当前时间，用于计算dt
    dt_initial = 1.0 / fps_video # 计算初始的dt
    kf = init_kalman(initial_bbox, dt_initial) # 初始化卡尔曼滤波器，使用初始bbox和dt

    traj = [] # 用于存储目标轨迹点的列表
    lost_counter = 0 # 目标丢失计数器
    is_re_detecting = False # 是否处于重新检测模式的标志
    last_known_bbox = initial_bbox # 上一个已知目标的边界框

    tracker_display_window_title = "Tracker Output (Press 'q' or ESC to quit)" # 跟踪输出窗口的标题
    total_frames_processed = 0 # 跟踪循环中处理的总帧数
    frames_target_found = 0 # 目标被成功跟踪的帧数
    times_target_lost_event = 0 # 目标丢失事件（进入重新检测模式）的次数

    current_tracking_frame_idx = 0 # 当前跟踪循环中的帧索引 (相对于跟踪开始的帧)

    while True: # 主跟踪循环
        ret, frame_loop_raw = cap_tracking.read() # 从视频中读取一帧
        if not ret: # 如果读取失败 (视频结束或错误)
            print("Video ended or failed to read next frame during tracking.")
            break # 跳出循环

        total_frames_processed += 1 # 总处理帧数加一

        # 如果初始帧被缩放过，则当前帧也需要按相同比例缩放
        if global_scale_factor != 1.0:
            frame_loop_scaled = cv2.resize(frame_loop_raw, None, fx=global_scale_factor, fy=global_scale_factor,
                                           interpolation=cv2.INTER_AREA)
        else:
            frame_loop_scaled = frame_loop_raw.copy() # 否则直接复制

        current_time = time.time() # 获取当前时间
        dt_actual = current_time - last_time # 计算实际的时间间隔 dt
        if dt_actual <= 0: dt_actual = 1e-3 # 防止 dt 为零或负
        last_time = current_time # 更新上一帧的时间

        curr_gray_full = cv2.cvtColor(frame_loop_scaled, cv2.COLOR_BGR2GRAY) # 将当前缩放后的帧转换为灰度图
        # curr_gray_processed = cv2.equalizeHist(curr_gray_full) # 考虑是否需要直方图均衡化
        curr_gray_processed = curr_gray_full # 使用原始灰度图

        # 估计全局仿射变换 (用于运动补偿)
        M, _ = estimate_global_affine(prev_gray_processed, curr_gray_processed, orb, bf)

        # 更新卡尔曼滤波器的状态转移矩阵中的dt
        kf.transitionMatrix[0, 4] = dt_actual
        kf.transitionMatrix[1, 5] = dt_actual
        prediction = kf.predict() # 卡尔曼滤波器进行预测
        # 从预测结果中获取预测的中心点 (pcx, pcy) 和宽高 (pw, ph)
        pcx, pcy = prediction[0, 0], prediction[1, 0]
        pw, ph = max(5, int(prediction[2, 0])), max(5, int(prediction[3, 0])) # 确保宽高至少为5

        search_center_x, search_center_y = pcx, pcy # 初始化搜索中心为卡尔曼预测的中心
        if M is not None and last_known_bbox is not None: # 如果全局运动估计成功且有上一个已知框
            # 使用全局运动模型来调整搜索中心
            last_cx = last_known_bbox[0] + last_known_bbox[2] / 2
            last_cy = last_known_bbox[1] + last_known_bbox[3] / 2
            transformed_center_x = M[0, 0] * last_cx + M[0, 1] * last_cy + M[0, 2]
            transformed_center_y = M[1, 0] * last_cx + M[1, 1] * last_cy + M[1, 2]
            search_center_x = transformed_center_x
            search_center_y = transformed_center_y

        frame_h_proc, frame_w_proc = curr_gray_processed.shape # 获取当前处理帧的尺寸
        if is_re_detecting: # 如果处于重新检测模式
            # 定义一个较大的搜索区域，基于上一个已知框和全局搜索因子
            search_w_roi = int(last_known_bbox[2] * GLOBAL_SEARCH_SCALE_FACTOR)
            search_h_roi = int(last_known_bbox[3] * GLOBAL_SEARCH_SCALE_FACTOR)
            scx_rd = last_known_bbox[0] + last_known_bbox[2] / 2 # 上一个已知框的中心
            scy_rd = last_known_bbox[1] + last_known_bbox[3] / 2
            sx = max(0, int(scx_rd - search_w_roi / 2)) # 搜索区域左上角 x
            sy = max(0, int(scy_rd - search_h_roi / 2)) # 搜索区域左上角 y
        else: # 如果处于正常跟踪模式
            # 定义搜索区域，基于卡尔曼预测的宽高和搜索窗口因子
            search_w_roi = int(pw * SEARCH_WINDOW_FACTOR)
            search_h_roi = int(ph * SEARCH_WINDOW_FACTOR)
            sx = max(0, int(search_center_x - search_w_roi / 2)) # 搜索区域左上角 x
            sy = max(0, int(search_center_y - search_h_roi / 2)) # 搜索区域左上角 y

        # 确保搜索区域不超出帧边界
        ex = min(frame_w_proc, sx + search_w_roi) # 搜索区域右下角 x
        ey = min(frame_h_proc, sy + search_h_roi) # 搜索区域右下角 y

        # 提取搜索区域图像，如果区域无效则创建一个空数组
        search_roi_img = curr_gray_processed[sy:ey, sx:ex] if ex > sx and ey > sy else np.array([[]],
                                                                                                dtype=curr_gray_processed.dtype)

        best_score = -1.0 # 初始化当前帧的最佳匹配分数
        best_bbox_abs = None # 初始化当前帧的最佳匹配边界框 (在完整处理帧上的绝对坐标)
        found_in_frame = False # 标记当前帧是否找到目标

        # 当前用于匹配的模板列表 (包含初始模板和自适应模板)
        current_templates_to_match = [init_tmpl] + list(templates)

        if search_roi_img.size > 10: # 确保搜索区域图像有效 (大小大于10个像素)
            pyramid_search_roi = build_pyramid(search_roi_img, PYRAMID_MAX_LEVEL) # 构建搜索区域的图像金字塔

            for tmpl_candidate in current_templates_to_match: # 遍历所有待匹配的模板
                if tmpl_candidate is None or tmpl_candidate.shape[0] == 0 or tmpl_candidate.shape[1] == 0: # 跳过无效模板
                    continue

                # 遍历搜索区域金字塔的每一层
                for level, roi_level_img in enumerate(pyramid_search_roi):
                    scale_py = 2 ** level # 当前金字塔层级的缩放比例

                    # 确保模板不大于当前金字塔层级的 ROI 图像
                    if tmpl_candidate.shape[0] > roi_level_img.shape[0] or \
                       tmpl_candidate.shape[1] > roi_level_img.shape[1]:
                        continue # 如果模板过大，则跳到下一层

                    score, loc = match_ncc(tmpl_candidate, roi_level_img) # 进行 NCC 匹配

                    if score > best_score: # 如果找到了更好的匹配
                        best_score = score # 更新最佳分数
                        # loc 是相对于 roi_level_img 的
                        roi_rel_x_at_level = loc[0] # 匹配位置在当前层级ROI上的x坐标
                        roi_rel_y_at_level = loc[1] # 匹配位置在当前层级ROI上的y坐标

                        # 将 loc 缩放回 search_roi_img 的坐标
                        roi_rel_x = int(roi_rel_x_at_level * scale_py)
                        roi_rel_y = int(roi_rel_y_at_level * scale_py)

                        # 在完整处理帧上的绝对坐标
                        abs_x = sx + roi_rel_x
                        abs_y = sy + roi_rel_y
                        match_w = int(tmpl_candidate.shape[1]) # 匹配宽度为模板自身的宽度
                        match_h = int(tmpl_candidate.shape[0]) # 匹配高度为模板自身的高度

                        # 确保 bbox 在完整帧边界内且具有正的维度
                        checked_abs_x = max(0, abs_x)
                        checked_abs_y = max(0, abs_y)
                        # 调整宽高以适应帧边界，同时保持左上角不变
                        checked_w = min(frame_w_proc - checked_abs_x, match_w)
                        checked_h = min(frame_h_proc - checked_abs_y, match_h)

                        if checked_w > 0 and checked_h > 0: # 如果调整后的宽高有效
                             best_bbox_abs = (checked_abs_x, checked_abs_y, checked_w, checked_h)


            current_threshold = NCC_BASE_THRESHOLD # 获取当前匹配阈值
            if is_re_detecting: current_threshold *= 0.9 # 如果在重新检测模式，稍微降低阈值

            if best_score >= current_threshold and best_bbox_abs is not None: # 如果最佳分数高于阈值且找到了有效bbox
                found_in_frame = True # 标记目标已找到
                frames_target_found += 1 # 成功跟踪帧数加一
                x_m, y_m, w_m, h_m = best_bbox_abs # 解包最佳匹配的bbox
                cx_m, cy_m = x_m + w_m / 2, y_m + h_m / 2 # 计算中心点
                meas = np.array([cx_m, cy_m, w_m, h_m], np.float32).reshape(4, 1) # 构建测量向量
                kf.correct(meas) # 使用测量值校正卡尔曼滤波器

                # 如果匹配分数足够高且不处于重新检测模式，则更新模板
                if best_score > TEMPLATE_UPDATE_CONF_THRESHOLD and not is_re_detecting:
                    # 确保新模板的区域在当前灰度图内且宽高大于0
                    if y_m + h_m <= curr_gray_processed.shape[0] and x_m + w_m <= curr_gray_processed.shape[1] and h_m > 0 and w_m > 0:
                        new_patch = curr_gray_processed[y_m: y_m + h_m, x_m: x_m + w_m] # 提取新的模板图像块
                        # 确保新图像块有效且初始模板尺寸有效
                        if new_patch.size > 0 and init_tmpl.shape[0] > 0 and init_tmpl.shape[1] > 0:
                            try:
                                # 将新图像块缩放到与初始模板相同的大小
                                resized_patch = cv2.resize(new_patch, (init_tmpl.shape[1], init_tmpl.shape[0]),
                                                           interpolation=cv2.INTER_AREA)
                                templates.append(resized_patch) # 将更新后的模板添加到队列中
                            except cv2.error: # 捕获可能的resize错误
                                pass # 静默忽略resize错误

                lost_counter = 0 # 重置丢失计数器
                if is_re_detecting: # 如果之前处于重新检测模式
                    print(
                        f"Info: Target re-acquired at frame index (tracking loop): {current_tracking_frame_idx}!") # 打印重新获取目标的信息
                is_re_detecting = False # 退出重新检测模式
                last_known_bbox = best_bbox_abs # 更新上一个已知bbox
            else: # 如果匹配分数低于阈值或未找到有效bbox
                found_in_frame = False # 标记目标未找到
                lost_counter += 1 # 丢失计数器加一
                if not is_re_detecting and lost_counter > MAX_LOST_FRAMES: # 如果不在重新检测模式且丢失帧数超过阈值
                    print(f"Info: Target lost (frame {current_tracking_frame_idx}). Re-detection mode...") # 进入重新检测模式
                    is_re_detecting = True
                    times_target_lost_event += 1 # 丢失事件次数加一
                    lost_counter = 0 # 重置丢失计数器以用于重新检测阶段
                elif is_re_detecting and lost_counter > RE_DETECTION_FRAMES: # 如果在重新检测模式且重新检测帧数超过阈值
                    print(f"Info: Re-detection failed (frame {current_tracking_frame_idx}). Target considered lost.") # 重新检测失败
                    is_re_detecting = False # 停止重新检测
                    lost_counter = MAX_LOST_FRAMES + 1 # 确保目标保持“丢失”状态直到明确找到
        else: # 如果搜索区域图像太小或无效
            found_in_frame = False # 标记目标未找到
            lost_counter += 1 # 丢失计数器加一
            if not is_re_detecting and lost_counter > MAX_LOST_FRAMES: # 处理同上的丢失逻辑
                is_re_detecting = True;
                lost_counter = 0; # 为重新检测阶段重置计数器
                times_target_lost_event += 1
            elif is_re_detecting and lost_counter > RE_DETECTION_FRAMES: # 处理同上的重新检测失败逻辑
                is_re_detecting = False;
                lost_counter = MAX_LOST_FRAMES + 1 # 保持丢失状态

        display_frame_output = frame_loop_scaled.copy() # 复制当前缩放帧用于显示
        box_color_current = BOX_COLOR_LOST # 初始化当前框颜色为丢失状态
        status_text = "Status: Lost" # 初始化状态文本
        x_draw, y_draw, w_draw, h_draw = 0, 0, 0, 0 # 初始化绘制框的坐标

        if found_in_frame and best_bbox_abs: # 如果在当前帧找到目标
            x_draw, y_draw, w_draw, h_draw = best_bbox_abs # 使用找到的bbox进行绘制
            center_x_traj, center_y_traj = int(x_draw + w_draw / 2), int(y_draw + h_draw / 2) # 计算中心点用于轨迹
            if w_draw > 0 and h_draw > 0: traj.append((center_x_traj, center_y_traj)) # 添加到轨迹列表
            box_color_current = BOX_COLOR_TRACKING # 设置框颜色为跟踪状态
            status_text = f"Status: Tracking (Score: {best_score:.2f})" # 更新状态文本
        elif is_re_detecting: # 如果处于重新检测模式
            x_draw, y_draw = int(pcx - pw / 2), int(pcy - ph / 2) # 使用卡尔曼预测的位置绘制
            w_draw, h_draw = int(pw), int(ph)
            box_color_current = BOX_COLOR_REDETECTION # 设置框颜色为重新检测状态
            status_text = f"Status: Re-detecting... (Attempt: {lost_counter+1}/{RE_DETECTION_FRAMES})" # 更新状态文本
        else: # 如果丢失且未处于重新检测模式 (在MAX_LOST_FRAMES之前或重新检测失败后)
            x_draw, y_draw = int(pcx - pw / 2), int(pcy - ph / 2) # 使用卡尔曼预测的位置绘制
            w_draw, h_draw = int(pw), int(ph)
            box_color_current = BOX_COLOR_LOST # 设置框颜色为丢失状态
            status_text = f"Status: Lost ({lost_counter} frames since detection)" # 更新状态文本

        x_draw, y_draw, w_draw, h_draw = map(int, [x_draw, y_draw, w_draw, h_draw]) # 将绘制坐标转换为整数
        if w_draw > 0 and h_draw > 0: # 确保绘制的宽高有效
            # 裁剪绘制框以确保其在显示帧的边界内
            x_c, y_c = max(0, x_draw), max(0, y_draw)
            w_c = min(w_draw, display_frame_output.shape[1] - x_c)
            h_c = min(h_draw, display_frame_output.shape[0] - y_c)
            if w_c > 0 and h_c > 0: # 确保裁剪后的宽高仍然有效
                cv2.rectangle(display_frame_output, (x_c, y_c), (x_c + w_c, y_c + h_c), box_color_current, 2) # 绘制矩形框

        max_traj_points_draw = 100 # 最大绘制的轨迹点数
        if len(traj) > 1: # 如果轨迹点数大于1
            visible_traj_pts = traj[max(0, len(traj) - max_traj_points_draw):] # 取最近的轨迹点
            if visible_traj_pts: # 确保可见轨迹点列表不为空
                pts_traj = np.array(visible_traj_pts, np.int32).reshape(-1, 1, 2) # 转换为 polylines 需要的格式
                cv2.polylines(display_frame_output, [pts_traj], isClosed=False, color=TRAJECTORY_COLOR, thickness=2) # 绘制轨迹线

        # 在显示帧上绘制状态文本
        cv2.putText(display_frame_output, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color_current, 2)
        # 计算并显示当前帧在原始视频中的实际帧号
        actual_frame_num_in_video = final_selected_frame_idx + 1 + current_tracking_frame_idx
        cv2.putText(display_frame_output, f"Frame (Video): {actual_frame_num_in_video}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # 使用白色文本显示通用信息
        cv2.imshow(tracker_display_window_title, display_frame_output) # 显示处理后的帧

        if video_writer and video_writer.isOpened(): # 如果视频写入对象有效且已打开
            video_writer.write(display_frame_output) # 将当前显示帧写入视频文件

        prev_gray_processed = curr_gray_processed.copy() # 更新前一帧的灰度图
        current_tracking_frame_idx += 1 # 跟踪循环的帧索引加一

        key = cv2.waitKey(30) & 0xFF # 等待 30毫秒，检测按键
        if key == ord('q') or key == 27: # 如果按下 'q' 或 ESC
            break # 退出跟踪循环

    cap_tracking.release() # 释放视频捕获对象
    if video_writer and video_writer.isOpened(): # 如果视频写入对象有效且已打开
        video_writer.release() # 释放视频写入对象
        print(f"Output video saved to: {OUTPUT_VIDEO_PATH}") # 打印输出视频保存路径
    elif video_writer is None and OUTPUT_VIDEO_PATH: # 检查 VideoWriter 是否从未成功打开但路径已设置
        print("Output video was not saved due to VideoWriter initialization issues.") # 打印未保存视频的警告


    cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口
    print("Tracker finished.") # 打印跟踪结束信息
    print("--- Tracking Statistics ---") # 打印跟踪统计信息标题
    print(f"Total frames processed in tracking loop: {total_frames_processed}") # 打印总处理帧数
    if total_frames_processed > 0: # 确保处理过帧才计算百分比
        print(
            f"Frames target actively tracked: {frames_target_found} ({(frames_target_found / total_frames_processed) * 100:.2f}%)") # 打印目标被主动跟踪的帧数和百分比
        print(f"Number of times target lost (entered re-detection): {times_target_lost_event}") # 打印目标丢失（进入重新检测）的次数
    print("Note: True miss/false alarm rates require ground truth data for comparison.") # 提示真实性能评估需要地面真实数据


if __name__ == '__main__': # 如果脚本作为主程序运行
    main() # 调用主函数
