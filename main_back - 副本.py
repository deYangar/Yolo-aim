import cv2
import argparse
import torch
import numpy as np
import os
import pynput
import time
import win32gui
import win32con
import sys
from PyQt5.QtCore import QCoreApplication
from configparser import ConfigParser
from aim_csgo.verify_args import verify_args
from aim_csgo.screen_inf import grab_screen_win32, get_parameters
from aim_csgo.cs_model import load_model
from aim_csgo.aim_lock_pi import lock

from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh

# ----------------------------------------------------#
# 1. 对yolov5的基础代码进行重构
# 2. 利用yolov5的数据输出进行辅助瞄准
# 3. 在基础代码上加上可视化界面
# ----------------------------------------------------#

config = ConfigParser()
config.read('config.ini', encoding='UTF-8')

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default=config.get('SECTION 1', "model-path"), help='模型地址')
parser.add_argument('--imgsz', type=int, default=config.getint('SECTION 1', "imgsz"), help='和你训练模型时imgsz一样')
parser.add_argument('--conf-thres', type=float, default=config.getfloat('SECTION 1', "conf-thres"), help='置信阈值')
parser.add_argument('--iou-thres', type=float, default=config.getfloat('SECTION 1', "iou-thres"), help='交并比阈值')
parser.add_argument('--use-cuda', type=bool, default=config.getboolean('SECTION 1', "use-cuda"), help='是否使用cuda')

parser.add_argument('--show-window', type=bool, default=config.getboolean('SECTION other', "show-window"),
                    help='是否显示实时检测窗口(新版里改进了效率。若为True，不要去点右上角的X！)')
parser.add_argument('--top-most', type=bool, default=config.getboolean('SECTION other', "top-most"),
                    help='是否保持实时检测窗口置顶')
parser.add_argument('--resize-window', type=float, default=config.getfloat('SECTION other', "resize-window"),
                    help='缩放实时检测窗口大小')
parser.add_argument('--thickness', type=int, default=config.getint('SECTION other', "thickness"),
                    help='画框粗细，必须大于1/resize-window')
parser.add_argument('--show-fps', type=bool, default=config.getboolean('SECTION other', "show-fps"), help='是否显示帧率')
parser.add_argument('--show-label', type=bool, default=config.getboolean('SECTION other', "show-label"), help='是否显示标签')

parser.add_argument('--region', type=tuple, default=eval(config.get('SECTION other', "region")),
                    help='检测范围；分别为横向和竖向，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')

parser.add_argument('--hold-lock', type=bool, default=config.getboolean('SECTION other', "hold-lock"),
                    help='lock模式；True为按住，False为切换')
parser.add_argument('--lock-sen', type=float, default=config.getfloat('SECTION other', "lock-sen"),
                    help='lock幅度系数；若在桌面试用请调成1，在游戏中(csgo)则为灵敏度')
parser.add_argument('--lock-smooth', type=float, default=config.getfloat('SECTION other', "lock-smooth"),
                    help='lock平滑系数；越大越平滑，最低1.0')
parser.add_argument('--lock-button', type=str, default=config.get('SECTION other', "lock-button"),
                    help='lock按键；只支持鼠标按键')
parser.add_argument('--head-first', type=bool, default=config.getboolean('SECTION other', "head-first"), help='是否优先瞄头')
parser.add_argument('--lock-tag', type=list, default=eval(config.get('SECTION other', "lock-tag")),
                    help='对应标签；缺一不可，自己按以下顺序对应标签，ct_head ct_body t_head t_body')
parser.add_argument('--lock-choice', type=list, default=eval(config.get('SECTION other', "lock-choice")),
                    help='目标选择；可自行决定锁定的目标，从自己的标签中选')

args = parser.parse_args()

# ------------------------------------------------#
# 判断输入的基本参数是否正确
# ------------------------------------------------#
verify_args(args)

# ------------------------------------------------#
# 先获取当前文件的完整路径然后去除文件名返回目录+'\\'
# ------------------------------------------------#
cur_dir = os.path.dirname(os.path.abspath(__file__)) + '\\'

args.model_path = cur_dir + args.model_path  # 模型的完整路径
args.lock_tag = [str(i) for i in args.lock_tag]  # 将标签由int转为str
args.lock_choice = [str(i) for i in args.lock_choice]  # 将锁定目标由int转为str

device = 'cuda' if args.use_cuda else 'cpu'  # 判断用cpu还是gpu预测
half = device != 'cpu'  # 当device为cuda时 half = cuda
imgsz = args.imgsz  # 预测模型的大小

conf_thres = args.conf_thres  # 置信度
iou_thres = args.iou_thres  # 交并比阈值

top_x, top_y, x, y = get_parameters()  # 获取屏幕的左上角和右下角的坐标
len_x, len_y = int(x * args.region[0]), int(y * args.region[1])  # 检测范围的宽和高
top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))  # 检测范围的左上角起始坐标
monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}  # 检测范围的具体参数

model = load_model(args)  # model由FP32转为FP16
stride = int(model.stride.max())  # model的最大stride
names = model.module.names if hasattr(model, 'module') else model.names  # 模型里面的标签名字

lock_mode = False  # 全局变量是否开启锁定
lock_button = eval('pynput.mouse.Button.' + args.lock_button)  # 开启锁定的键位
mouse = pynput.mouse.Controller()  # 鼠标的管理

# ------------------------------------------------#
# 是否显示实时检测窗口
# ------------------------------------------------#
if args.show_window:
    cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('csgo-detect', int(len_x * args.resize_window), int(len_y * args.resize_window))


# ------------------------------------------------#
# 开启非阻塞鼠标监控
# ------------------------------------------------#
def on_click(x, y, button, pressed):
    global lock_mode
    if button == lock_button:  # 如果按键为开启锁定的键位
        if args.hold_lock:  # 如果开启按住锁定
            if pressed:
                lock_mode = True
                print('locking...')
            else:
                lock_mode = False
                print('lock mode off')
        else:  # 开启切换锁定
            if pressed:
                lock_mode = not lock_mode
                print('lock mode', 'on' if lock_mode else 'off')


listener = pynput.mouse.Listener(on_click=on_click)
listener.start()

close = True


# ------------------------------------------------#
# 开启非阻塞键盘监控
# 当按“END”键时退出整个程序
# ------------------------------------------------#
def on_press(key):
    global close
    if key == pynput.keyboard.Key.end:
        # QCoreApplication.instance().quit()
        close = False

        print(1)


listener1 = pynput.keyboard.Listener(on_press=on_press)
listener1.start()

print('enjoy yourself!')


t0 = time.time()
cnt = 0
while close:
    if cnt % 20 == 0:
        top_x, top_y, x, y = get_parameters()  # 获取屏幕的左上角和右下角的坐标
        len_x, len_y = int(x * args.region[0]), int(y * args.region[1])  # 检测范围的宽和高
        top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(
            top_y + y // 2 * (1. - args.region[1]))  # 检测范围的左上角起始坐标
        monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}  # 检测范围的具体参数
        cnt = 1

    # ------------------------------------------------#
    # 调用win32进行截图
    # ------------------------------------------------#
    img0 = grab_screen_win32(region=(top_x, top_y, top_x + len_x, top_y + len_y))
    img0 = cv2.resize(img0, (len_x, len_y))

    img = letterbox(img0, imgsz, stride=stride)[0]  # 在满足多个步幅约束的情况下调整图像大小并填充图像

    # ------------------------------------------------#
    # 预测前对图片进行处理
    # ------------------------------------------------#
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.

    if len(img.shape) == 3:
        img = img[None]

    # ------------------------------------------------#
    # 对图片进行预测
    # ------------------------------------------------#
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)  # 对推断结果运行非最大抑制（NMS）

    aims = []
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            # 将框从 img_size 重新缩放为 im0 大小
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # ------------------------------------------------#
            # 写入结果
            # 当预测结果有东西才会进行
            # ------------------------------------------------#
            for *xyxy, conf, cls in reversed(det):
                # bbox:(tag, x_center, y_center, x_width, y_width)
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 标准化 xywh
                line = (cls, *xywh)  # 标签格式
                aim = ('%g ' * len(line)).rstrip() % line
                aim = aim.split(' ')
                aims.append(aim)

        # ------------------------------------------------#
        # 如果有目标就会进行锁定并显示画面
        # ------------------------------------------------#
        if len(aims):
            if lock_mode:
                lock(aims, mouse, top_x, top_y, len_x, len_y, args)  # 锁定目标

            if args.show_window:
                for i, det in enumerate(aims):
                    tag, x_center, y_center, width, height = det
                    x_center, width = len_x * float(x_center), len_x * float(width)
                    y_center, height = len_y * float(y_center), len_y * float(height)
                    top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))  # 对目标进行画框
                    cv2.rectangle(img0, top_left, bottom_right, (0, 255, 0), thickness=args.thickness)
                    if args.show_label:
                        cv2.putText(img0, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 0, 0), 4)  # 对标签进行显示

    # ------------------------------------------------#
    # 是否实时显示画面
    # ------------------------------------------------#
    if args.show_window:

        if args.show_fps:  # 显示fps值
            cv2.putText(img0, "FPS:{:.1f}".format(1. / (time.time() - t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 235), 4)
            # print(1. / (time.time() - t0))
            t0 = time.time()
        cv2.imshow('csgo-detect', img0)

        if args.top_most:  # 是否保持实时检测窗口置顶
            hwnd = win32gui.FindWindow(None, 'csgo-detect')
            CVRECT = cv2.getWindowImageRect('csgo-detect')
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        cv2.waitKey(1)
    # cnt += 1
