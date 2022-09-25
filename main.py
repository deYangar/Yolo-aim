import sys
import untitled

from PyQt5.QtWidgets import QApplication, QMainWindow
from configparser import ConfigParser


config = ConfigParser()
config.read('config.ini', encoding='UTF-8')


def c_cuda():
    if ui.checkBox.isChecked():
        config.set('SECTION 1', 'use-cuda', 'True')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()

    else:
        config.set('SECTION 1', 'use-cuda', 'False')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()


def show_window():
    if ui.checkBox_2.isChecked():
        config.set('SECTION other', 'show-window', 'True')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()

    else:
        config.set('SECTION other', 'show-window', 'False')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()


def top_most():
    if ui.checkBox_3.isChecked():
        config.set('SECTION other', 'top-most', 'True')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()

    else:
        config.set('SECTION other', 'top-most', 'False')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()


def resize_window():
    resize_window_num = ui.lineEdit_2.text()
    config.set('SECTION other', 'resize-window', resize_window_num)
    f = open('config.ini', "w")
    config.write(f)  # 写进文件
    f.close()


def thickness():
    thickness_num = ui.lineEdit_3.text()
    config.set('SECTION other', 'thickness', thickness_num)
    f = open('config.ini', "w")
    config.write(f)  # 写进文件
    f.close()


def show_fps():
    if ui.checkBox_4.isChecked():
        config.set('SECTION other', 'show-fps', 'True')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()

    else:
        config.set('SECTION other', 'show-fps', 'False')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()


def show_label():
    if ui.checkBox_5.isChecked():
        config.set('SECTION other', 'show-label', 'True')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()

    else:
        config.set('SECTION other', 'show-label', 'False')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()


def region():
    region_num = ui.lineEdit_4.text()
    config.set('SECTION other', 'region', region_num)
    f = open('config.ini', "w")
    config.write(f)  # 写进文件
    f.close()


def hold_lock():
    if ui.checkBox_6.isChecked():
        config.set('SECTION other', 'hold-lock', 'True')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()

    else:
        config.set('SECTION other', 'hold-lock', 'False')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()


def lock_sen():
    lock_sen_num = ui.lineEdit_5.text()
    config.set('SECTION other', 'lock-sen', lock_sen_num)
    f = open('config.ini', "w")
    config.write(f)  # 写进文件
    f.close()


def lock_smooth():
    lock_smooth_num = ui.lineEdit_6.text()
    config.set('SECTION other', 'lock-smooth', lock_smooth_num)
    f = open('config.ini', "w")
    config.write(f)  # 写进文件
    f.close()


def lock_button():
    lock_button_num = ui.lineEdit_7.text()
    config.set('SECTION other', 'lock-button', lock_button_num)
    f = open('config.ini', "w")
    config.write(f)  # 写进文件
    f.close()


def head_first():
    if ui.checkBox_8.isChecked():
        config.set('SECTION other', 'head-first', 'True')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()

    else:
        config.set('SECTION other', 'head-first', 'False')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()

def auto_fire():
    if ui.checkBox_9.isChecked():
        config.set('SECTION other', 'auto-fire', 'True')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()

    else:
        config.set('SECTION other', 'auto-fire', 'False')
        f = open('config.ini', "w")
        config.write(f)  # 写进文件
        f.close()


def conf_thres():
    conf_thres_num = ui.lineEdit.text()
    config.set('SECTION 1', 'conf-thres', conf_thres_num)
    f = open('config.ini', "w")
    config.write(f)  # 写进文件
    f.close()


def start():
    from main_back import Predict
    Predict()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = untitled.Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.checkBox.clicked.connect(c_cuda)
    ui.checkBox_2.clicked.connect(show_window)
    ui.checkBox_3.clicked.connect(top_most)
    ui.checkBox_4.clicked.connect(show_fps)
    ui.checkBox_5.clicked.connect(show_label)
    ui.checkBox_6.clicked.connect(hold_lock)
    ui.checkBox_8.clicked.connect(head_first)
    ui.checkBox_9.clicked.connect(auto_fire)
    ui.lineEdit.editingFinished.connect(conf_thres)
    ui.lineEdit_2.editingFinished.connect(resize_window)
    ui.lineEdit_3.editingFinished.connect(thickness)
    ui.lineEdit_4.editingFinished.connect(region)
    ui.lineEdit_5.editingFinished.connect(lock_sen)
    ui.lineEdit_6.editingFinished.connect(lock_smooth)
    ui.lineEdit_7.editingFinished.connect(lock_button)
    ui.pushButton.clicked.connect(start)
    MainWindow.show()
    sys.exit(app.exec_())
