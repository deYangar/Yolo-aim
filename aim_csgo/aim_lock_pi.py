import time
import aim_csgo.ghub_mouse as ghub
from math import *

# 利用反正切函数处理鼠标现位置与目标位置的距离，距离越远需处理的次数越多。

flag = 0


def lock(aims, mouse, top_x, top_y, len_x, len_y, args):
    mouse_pos_x, mouse_pos_y = mouse.position
    aims_copy = aims.copy()
    aims_copy = [x for x in aims_copy if x[0] in args.lock_choice]
    k = 4.07 * (1 / args.lock_smooth)
    if len(aims_copy):
        dist_list = []
        tag_list = [x[0] for x in aims_copy]
        if args.head_first:
            if args.lock_tag[0] in tag_list or args.lock_tag[2] in tag_list:  # 有头
                aims_copy = [x for x in aims_copy if x[0] in [args.lock_tag[0], args.lock_tag[2]]]
        for det in aims_copy:
            _, x_c, y_c, _, _ = det
            dist = (len_x * float(x_c) + top_x - mouse_pos_x) ** 2 + (len_y * float(y_c) + top_y - mouse_pos_y) ** 2
            dist_list.append(dist)

        det = aims_copy[dist_list.index(min(dist_list))]
        tag, x_center, y_center, width, height = det
        x_center, width = len_x * float(x_center) + top_x, len_x * float(width)
        y_center, height = len_y * float(y_center) + top_y, len_y * float(height)
        rel_x = int(k / args.lock_sen * atan((mouse_pos_x - x_center) / 640) * 640)
        if tag in [args.lock_tag[0], args.lock_tag[2]]:
            rel_y = int(k / args.lock_sen * atan((mouse_pos_y - y_center) / 640) * 640)
            if flag:
                return
            ghub.mouse_xy(-rel_x, -rel_y)
            if args.auto_fire:
                if abs(rel_x) < 10 and abs(rel_y) < 10:
                    ghub.mouse_down(1)
                    ghub.mouse_up(1)
                    print(1)

            print(-rel_x, -rel_y)
        elif tag in [args.lock_tag[1], args.lock_tag[3]]:
            rel_y = int(k / args.lock_sen * atan((mouse_pos_y - y_center + 1 / 6 * height) / 640) * 640)
            if flag:
                return
            ghub.mouse_xy(-rel_x, -rel_y)
            if args.auto_fire:
                if abs(rel_x) < 6 and abs(rel_y) < 6:
                    ghub.mouse_down(1)
                    ghub.mouse_up(1)
                    print(2)
            print(-rel_x, -rel_y)
