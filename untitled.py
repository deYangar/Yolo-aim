# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(350, 500)
        MainWindow.setMinimumSize(QtCore.QSize(350, 500))
        MainWindow.setMaximumSize(QtCore.QSize(350, 522))
        MainWindow.setMouseTracking(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("photo/title.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setWindowOpacity(0.9)
        MainWindow.setStyleSheet("QMainWindow{\n"
"border-image:url(photo/back.jpeg);\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(350, 500))
        self.centralwidget.setMaximumSize(QtCore.QSize(350, 500))
        self.centralwidget.setStyleSheet("QWidget{\n"
"border-image:url(photo/back.jpeg);\n"
"}")
        self.centralwidget.setObjectName("centralwidget")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(20, 130, 91, 21))
        font = QtGui.QFont()
        font.setFamily("华光粗圆_CNKI")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.checkBox.setFont(font)
        self.checkBox.setStyleSheet("QCheckBox{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(280, 130, 51, 20))
        self.lineEdit.setMinimumSize(QtCore.QSize(31, 20))
        font = QtGui.QFont()
        font.setFamily("华光粗圆_CNKI")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("QLineEdit{\n"
"border-image:url();\n"
"background-color:rgb(255, 255, 255,160);\n"
"border:none;\n"
"color:#003cff;\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.lineEdit.setClearButtonEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(160, 130, 91, 21))
        font = QtGui.QFont()
        font.setFamily("华光粗圆_CNKI")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.label.setObjectName("label")
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(20, 160, 91, 21))
        font = QtGui.QFont()
        font.setFamily("华光粗圆_CNKI")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setStyleSheet("QCheckBox{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.checkBox_2.setChecked(True)
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(20, 190, 91, 16))
        self.checkBox_3.setStyleSheet("QCheckBox{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.checkBox_3.setChecked(True)
        self.checkBox_3.setObjectName("checkBox_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(160, 160, 91, 21))
        font = QtGui.QFont()
        font.setFamily("华光粗圆_CNKI")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_2.setObjectName("label_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(280, 160, 51, 20))
        self.lineEdit_2.setStyleSheet("QLineEdit{\n"
"border-image:url();\n"
"background-color:rgb(255, 255, 255,160);\n"
"border:none;\n"
"color:#003cff;\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(160, 190, 71, 21))
        font = QtGui.QFont()
        font.setFamily("华光粗圆_CNKI")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_3.setObjectName("label_3")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(280, 190, 51, 20))
        self.lineEdit_3.setStyleSheet("QLineEdit{\n"
"border-image:url();\n"
"background-color:rgb(255, 255, 255,160);\n"
"border:none;\n"
"color:#003cff;\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(20, 220, 81, 16))
        self.checkBox_4.setStyleSheet("QCheckBox{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.checkBox_4.setChecked(True)
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(20, 250, 81, 16))
        self.checkBox_5.setStyleSheet("QCheckBox{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.checkBox_5.setChecked(True)
        self.checkBox_5.setObjectName("checkBox_5")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(160, 280, 101, 21))
        font = QtGui.QFont()
        font.setFamily("华光粗圆_CNKI")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_4.setObjectName("label_4")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(280, 280, 51, 20))
        self.lineEdit_4.setStyleSheet("QLineEdit{\n"
"border-image:url();\n"
"background-color:rgb(255, 255, 255,160);\n"
"border:none;\n"
"color:#003cff;\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.checkBox_6 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_6.setGeometry(QtCore.QRect(20, 280, 71, 21))
        self.checkBox_6.setStyleSheet("QCheckBox{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.checkBox_6.setChecked(True)
        self.checkBox_6.setObjectName("checkBox_6")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(160, 220, 91, 21))
        self.label_5.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_5.setObjectName("label_5")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(280, 220, 51, 20))
        self.lineEdit_5.setStyleSheet("QLineEdit{\n"
"border-image:url();\n"
"background-color:rgb(255, 255, 255,160);\n"
"border:none;\n"
"color:#003cff;\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(160, 250, 91, 21))
        self.label_6.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_6.setObjectName("label_6")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(280, 250, 51, 20))
        self.lineEdit_6.setStyleSheet("QLineEdit{\n"
"border-image:url();\n"
"background-color:rgb(255, 255, 255,160);\n"
"border:none;\n"
"color:#003cff;\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.checkBox_8 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_8.setGeometry(QtCore.QRect(20, 310, 81, 16))
        self.checkBox_8.setStyleSheet("QCheckBox{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.checkBox_8.setObjectName("checkBox_8")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(110, 370, 121, 51))
        self.pushButton.setStyleSheet("QPushButton{\n"
"background-color: #bee4f9;\n"
"font: 20pt \"华光粗圆_CNKI\";\n"
"border-image:url();\n"
"border:1px groove gray;\n"
"border-radius:20%;\n"
"padding:1px 4px;\n"
"border-style: outset;}\n"
"QPushButton:hover{\n"
"color: rgb(0, 85, 255);}\n"
"QPushButton:pressed{\n"
"background-color:#aaffff;\n"
"border-style: inset;}")
        self.pushButton.setObjectName("pushButton")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(130, 40, 161, 41))
        self.label_7.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(0, 44, 132);\n"
"font: 25pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(40, 20, 71, 61))
        self.label_8.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(0, 44, 132);\n"
"font: 60pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(10, 450, 261, 16))
        self.label_9.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(255, 0, 0);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(80, 450, 161, 61))
        self.label_10.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(255, 0, 0);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_10.setObjectName("label_10")
        self.checkBox_9 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_9.setGeometry(QtCore.QRect(20, 340, 81, 16))
        self.checkBox_9.setStyleSheet("QCheckBox{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.checkBox_9.setObjectName("checkBox_9")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(160, 310, 71, 16))
        self.label_11.setStyleSheet("QLabel{\n"
"border-image:url();\n"
"color:rgb(0, 85, 255);\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.label_11.setObjectName("label_11")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_7.setGeometry(QtCore.QRect(280, 310, 51, 20))
        self.lineEdit_7.setStyleSheet("QLineEdit{\n"
"border-image:url();\n"
"background-color:rgb(255, 255, 255,160);\n"
"border:none;\n"
"color:#003cff;\n"
"font: 10pt \"华光粗圆_CNKI\";\n"
"}")
        self.lineEdit_7.setObjectName("lineEdit_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI csgo自瞄"))
        self.checkBox.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">N卡可选</span></p><p><span style=\" font-weight:600; color:#ff0000;\">A卡与核显不打勾</span></p></body></html>"))
        self.checkBox.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.checkBox.setText(_translate("MainWindow", "选用CUDA"))
        self.lineEdit.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; color:#ff0000;\">请输入0到1之间的小数</span></p></body></html>"))
        self.lineEdit.setText(_translate("MainWindow", "0.6"))
        self.label.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">大于此值的对象，将会被识别并瞄准</span></p><p><span style=\" font-weight:600; color:#ff0000;\">不大于此值的对象不会显示</span></p><p><span style=\" font-weight:600; color:#ff0000;\">此值填0-1的小数</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "置信度(0-1)："))
        self.checkBox_2.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">小窗口用来观看检测效果</span></p></body></html>"))
        self.checkBox_2.setText(_translate("MainWindow", "显示小窗口"))
        self.checkBox_3.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">打勾：小窗口在各种软件的最顶上</span></p><p><span style=\" font-weight:600; color:#ff0000;\">不打勾：其他软件可以覆盖在小窗口上</span></p></body></html>"))
        self.checkBox_3.setText(_translate("MainWindow", "小窗口置顶"))
        self.label_2.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">缩放小窗口的大小，值为0-1的小数</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "缩放窗口(0-1)："))
        self.lineEdit_2.setText(_translate("MainWindow", "1"))
        self.label_3.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">识别出对象画框的粗细，必须大于1/缩放窗口值</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "画框粗细>1："))
        self.lineEdit_3.setText(_translate("MainWindow", "1"))
        self.checkBox_4.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">打勾：显示检测的fps（帧率）</span></p><p><span style=\" font-weight:600; color:#ff0000;\">不打勾：不显示</span></p></body></html>"))
        self.checkBox_4.setText(_translate("MainWindow", "显示帧率"))
        self.checkBox_5.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">打勾：显示检测对象的标签</span></p><p><span style=\" font-weight:600; color:#ff0000;\">不打勾：不显示</span></p></body></html>"))
        self.checkBox_5.setText(_translate("MainWindow", "显示标签"))
        self.label_4.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">检测窗口的大小，为0-1的小数一般设为(0.3,0.3)</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "检测屏幕的比例："))
        self.lineEdit_4.setText(_translate("MainWindow", "(0.3,0.3)"))
        self.checkBox_6.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">打勾：   按住按键瞄准</span></p><p><span style=\" font-weight:600; color:#ff0000;\">不打勾：切换按键瞄准</span></p></body></html>"))
        self.checkBox_6.setText(_translate("MainWindow", "Aim模式"))
        self.label_5.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">瞄准幅度，桌面为1.0，csgo中为游戏灵敏度</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "lock幅度系数："))
        self.lineEdit_5.setText(_translate("MainWindow", "1.0"))
        self.label_6.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">瞄准平滑数，使鼠标瞄准抖动幅度更小，一般建议设为3.0</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "lock平滑系数："))
        self.lineEdit_6.setText(_translate("MainWindow", "3.0"))
        self.checkBox_8.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">打勾：优先瞄准头部</span></p><p><span style=\" font-weight:600; color:#ff0000;\">不打勾：离那个近先瞄准那个</span></p></body></html>"))
        self.checkBox_8.setText(_translate("MainWindow", "优先锁头"))
        self.pushButton.setText(_translate("MainWindow", "START"))
        self.label_7.setText(_translate("MainWindow", "CSGO 自瞄"))
        self.label_8.setText(_translate("MainWindow", "AI"))
        self.label_9.setText(_translate("MainWindow", "注意：关闭自瞄软件为键盘上的END键！！！"))
        self.label_10.setText(_translate("MainWindow", "第一次打开可能会慢一点！"))
        self.checkBox_9.setText(_translate("MainWindow", "自动开枪"))
        self.label_11.setText(_translate("MainWindow", "lock按键："))
        self.lineEdit_7.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">left：鼠标左键</span></p><p><span style=\" font-weight:600; color:#ff0000;\">right：鼠标右键</span></p><p><span style=\" font-weight:600; color:#ff0000;\">middle：鼠标中键</span></p></body></html>"))
        self.lineEdit_7.setText(_translate("MainWindow", "left"))