from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets

from gui_frame.gui_pyqt import Ui_GUI_PyQt
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import os
import time
import matplotlib


class mywindow(QtWidgets.QWidget, Ui_GUI_PyQt):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)

        self.tabWidget.setCurrentIndex(0)
        os.system('tmux set -g mouse on')
        # tab train ----------------------------------------------------
        self.t_train_dep_img_size_box.setCurrentIndex(1)

        valid = QtGui.QIntValidator()
        valid.setBottom(1)
        self.t_train_n_epoc_line.setValidator(valid)  # TODO others add constraints

        self.t_train_data_path_button.clicked.connect(
            lambda: self.file_dialog2line_text(self.t_train_data_path_line, if_folder=True)
        )
        self.t_train_csv_path_button.clicked.connect(
            lambda: self.file_dialog2line_text(self.t_train_csv_path_line, if_folder=False)
        )

        self.t_tr_log_dir_line.setText(os.path.join(os.getcwd(), 'runs'))

        # self.t_tr_term_cont_layout = QtWidgets.QVBoxLayout(self.term_train_cont)
        # self.t_tr_term_cont_layout.setObjectName("t_tr_term_cont_layout")
        # self.t_tr_term_cont_layout.addItem()

        self.t_train_data_combobox.activated[str].connect(lambda text: self.update_t_tr_cmd())
        self.t_train_data_path_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_csv_path_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_n_epoc_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_d_step_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_batch_size_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_z_dim_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_ndf_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_ngf_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_dep_img_size_box.activated[str].connect(lambda text: self.update_t_tr_cmd())
        self.t_train_lr_g_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_lr_d_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_lr_beta1_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_lr_beta2_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_data_percent_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_ckpt_factor_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_emb_dim_line.textChanged.connect(self.update_t_tr_cmd)
        self.t_train_data_aug_box.toggled.connect(self.update_t_tr_cmd)
        self.t_train_cond_box.toggled.connect(self.update_t_tr_cmd)

        self.update_t_tr_cmd()

        self.t_tr_run_button.clicked.connect(self.t_tr_training)
        self.t_tr_stop_tr_button.clicked.connect(self.t_tr_stop_tr_act)

        # terminal --------
        os.system('tmux kill-session -t train_session')
        self.process = QtCore.QProcess(self)
        self.process.start("xterm", ["-fa", "Monospace", "-fs", "14",
                                     "-into",  f"{int(self.term_train_cont.winId())}",
                                     '-e', 'tmux', 'new', '-s', 'train_session',
                                     ])
        # tab continue training init
        self.init_t_ct()

        # tab evaluation ini
        self.t_ev_init()

        # tab web--------------------------------------------
        self.t_tfb_path_line.setText(os.path.join(os.getcwd(), 'runs'))
        self.t_tfb_open_path_button.accepted.connect(
            lambda: self.file_dialog2line_text(self.t_tfb_path_line, if_folder=True))

        os.system('tmux kill-session -t tfb_session')
        self.process = QtCore.QProcess(self)
        self.process.start("xterm", ["-fa", "Monospace", "-fs", "14",
                                     "-into",  f"{int(self.tfb_term_cont.winId())}",
                                     '-e', 'tmux', 'new', '-s', 'tfb_session',
                                     ])

        self.t_tfb_begin_button.clicked.connect(self.t_tfb_begin_button_on_clicked)
        self.t_tfb_stop_button.clicked.connect(self.t_tfb_stop_button_on_clicked)

        self.web = QWebEngineView(self.tfb_cont)
        self.web.setGeometry(QtCore.QRect(20, 20, 900, 500))
        # self.web.load(QUrl("http://localhost:6006/"))
        self.web.load(QUrl.fromLocalFile(os.path.join(os.getcwd(), 'gui_frame/default_tfb.html')))
        # self.web.load(QUrl("http://youtube.com"))
        # self.web.show()
        return

    def t_tfb_stop_button_on_clicked(self):
        os.system("tmux send-keys -t tfb_session C-c")
        self.web.load(QUrl.fromLocalFile(os.path.join(os.getcwd(), 'gui_frame/default_tfb.html')))
        return

    def init_t_ct(self):
        self.t_ct_log_dir_line.setText(os.path.join(os.getcwd(), 'runs'))
        self.t_ct_open_path_button.clicked.connect(
            lambda: self.file_dialog2line_text(self.t_ct_log_dir_line, if_folder=False, file_filter='(*.pth)')
        )
        self.t_ct_log_dir_line.textChanged.connect(self.t_ct_log_dir_on_text_changed)
        os.system('tmux kill-session -t continue_session')
        self.process = QtCore.QProcess(self)
        self.process.start("xterm", ["-fa", "Monospace", "-fs", "14",
                                     "-into",  f"{int(self.t_ct_term_cont.winId())}",
                                     '-e', 'tmux', 'new', '-s', 'continue_session',
                                     ])

        self.t_ct_tr_button.clicked.connect(
            lambda: os.system(f"tmux send-key -t continue_session "
                              f"\" python cwgan_gp.py --data "
                              f"{self.t_ct_data_combobox.currentText()}  "
                              f"--recover "
                              f"--checkpoint-file " 
                              f"{self.t_ct_log_dir_line.text()} \" "
                              f"Enter"
                              )
        )
        self.t_ct_stop_button.clicked.connect(
            lambda: os.system(f"tmux send-key -t continue_session C-c ")
        )
        return

    def t_ct_log_dir_on_text_changed(self):
        file_path = self.t_ct_log_dir_line.text()
        if os.path.isfile(file_path):
            import torch
            ckpt = torch.load(file_path)
            args = ckpt['arg']
            del ckpt
            self.t_ct_data_path_line.setText(str(args['root']))
            self.t_ct_csv_path_line.setText(str(args['csv_file']))
            self.t_ct_data_combobox.setCurrentText(str(args['data']))
            self.t_ct_n_epoc_line.setText(str(args['n_epoc']))
            self.t_ct_d_step_line.setText(str(args['d_step']))
            self.t_ct_batch_size_line.setText(str(args['d_step']))
            self.t_ct_z_dim_line.setText(str(args['z_dim']))
            self.t_ct_ndf_line.setText(str(args['ndf']))
            self.t_ct_ngf_line.setText(str(args['ngf']))
            self.t_ct_dep_img_size_box.setCurrentText(str(args['depth'])+'/'+str(args['img_size']))
            self.t_ct_lr_d_line.setText(str(args['lr_d']))
            self.t_ct_lr_g_line.setText(str(args['lr_g']))
            self.t_ct_lr_beta1_line.setText(str(args['lr_beta1']))
            self.t_ct_lr_beta2_line.setText(str(args['lr_beta2']))
            self.t_ct_data_percent_line.setText(str(args['data_percentage']))
            self.t_ct_ckpt_factor_line.setText(str(args['checkpoint_factor']))
            self.t_ct_emb_dim_line.setText(str(args['embedding_dim']))
            self.t_ct_data_aug_box.setChecked(args['data_aug'])
            self.t_ct_cond_box.setChecked(args['condition'])

            self.t_ct_ct_tr_cmd_line.setPlainText(
                f"python cwgan_gp.py --data {self.t_ct_data_combobox.currentText()}  "
                f"--recover "
                f"\r\n\t --checkpoint-file {self.t_ct_log_dir_line.text()}"
            )
        return

    def t_ev_init(self):
        self.t_ev_pth_path_button.clicked.connect(
            lambda: self.file_dialog2line_text(self.t_ev_pth_path_line, if_folder=False, file_filter="(*.pth)")
        )
        self.t_ev_pth_eval_button.clicked.connect(self.t_ev_pth_eval_button_on_click)

        return

    def t_ev_pth_eval_button_on_click(self):
        file_path = self.t_ev_pth_path_line.text()
        if os.path.exists(file_path):
            import evaluate

            cache_path = os.path.join(os.getcwd(), "cache")
            if os.path.isdir(cache_path) is False:
                os.makedirs(cache_path)

            try:
                evaluate.evaluate_model(
                    ckpt_file=file_path, device="cuda", show_fig=False, save_path=os.path.join(cache_path, "eval.png"))
            except:
                evaluate.evaluate_model(
                    ckpt_file=file_path, device="cuda", show_fig=False, save_path=os.path.join(cache_path, "eval.png"))

            image = QtGui.QPixmap()
            image.load(os.path.join(cache_path, "eval.png"))
            self.graphicsView.scene = QGraphicsScene()  # 创建一个图片元素的对象
            item = QGraphicsPixmapItem(image)  # 创建一个变量用于承载加载后的图片
            self.graphicsView.scene.addItem(item)  # 将加载后的图片传递给scene对象
            self.graphicsView.setScene(self.graphicsView.scene)
        else:
            reply = QMessageBox.information(self,
                                            "Error",
                                            "Invalid File path",
                                            QMessageBox.Ok)
        return

    def t_tfb_begin_button_on_clicked(self):
        os.system(
            f'tmux send-key -t tfb_session \"tensorboard --logdir {self.t_tfb_path_line.text()}\" Enter'
        )
        # time.sleep(2)
        time.sleep(4)
        self.web.load(QUrl("http://localhost:6006/"))
        return

    def file_dialog2line_text(self, line_text, if_folder, file_filter='(*.csv)'):
        # home_dir = self.t_train_data_path_line.text()
        home_dir = line_text.text()
        if os.path.exists(home_dir) is False:
            home_dir = os.getcwd()

        if if_folder:
            f_name = QFileDialog.getExistingDirectory(self, 'Open file', home_dir)
        else:
            f_name, _ = QFileDialog.getOpenFileName(self, 'Open file', home_dir, file_filter)
        # self.t_train_data_path_line.setText(f_name)
        if f_name == '':
            f_name = line_text.text()
        line_text.setText(f_name)
        return

    def closeEvent(self, event):
        import os
        os.system('tmux kill-session -t train_session')
        os.system('tmux kill-session -t tfb_session')
        os.system('tmux kill-session -t continue_session')
        return

    def update_t_tr_cmd(self):
        cmd_list = ['python', 'cwgan_gp.py']
        cmd_list.append('--data')
        cmd_list.append(self.t_train_data_combobox.currentText())
        cmd_list.append('--n-epoc')
        cmd_list.append(self.t_train_n_epoc_line.text())
        cmd_list.append('--d-step')
        cmd_list.append(self.t_train_d_step_line.text())
        cmd_list.append('--batch-size')
        cmd_list.append(self.t_train_batch_size_line.text())
        cmd_list.append('--z-dim')
        cmd_list.append(self.t_train_z_dim_line.text())
        cmd_list.append('--ndf')
        cmd_list.append(self.t_train_ndf_line.text())
        cmd_list.append('--ngf')
        cmd_list.append(self.t_train_ngf_line.text())
        depth, img_size = self.t_train_dep_img_size_box.currentText().split('/')
        cmd_list.append('--depth')
        cmd_list.append(depth)
        cmd_list.append('--img-size')
        cmd_list.append(img_size)
        cmd_list.append('--lr-g')
        cmd_list.append(self.t_train_lr_g_line.text())
        cmd_list.append('--lr-d')
        cmd_list.append(self.t_train_lr_d_line.text())
        cmd_list.append('--lr-beta1')
        cmd_list.append(self.t_train_lr_beta1_line.text())
        cmd_list.append('--lr-beta2')
        cmd_list.append(self.t_train_lr_beta2_line.text())
        cmd_list.append('--data-percentage')
        cmd_list.append(self.t_train_data_percent_line.text())
        cmd_list.append('--checkpoint-factor')
        cmd_list.append(self.t_train_ckpt_factor_line.text())
        cmd_list.append('--embedding-dim')
        cmd_list.append(self.t_train_emb_dim_line.text())
        if self.t_train_data_aug_box.isChecked():
            cmd_list.append('--data-aug')
        if self.t_train_cond_box.isChecked():
            cmd_list.append('--condition')
        cmd_list.append('--root')
        cmd_list.append(self.t_train_data_path_line.text())
        cmd_list.append('--csv-file')
        cmd_list.append(self.t_train_csv_path_line.text())

        cmd = ''
        cnt = 1
        for _item in cmd_list:
            cmd = cmd + _item + '  '
            if len(cmd)/(110*cnt) > 1.:
                cmd = cmd + '\n\t'
                cnt += 1
        # cmd = cmd.replace('\n', ' ')
        # cmd = cmd.replace('\t', ' ')
        self.t_tr_cmd_lineedit.setPlainText(cmd)
        return

    def t_tr_training(self):
        cmd = self.t_tr_cmd_lineedit.toPlainText()
        cmd = cmd.replace('\n', ' ')
        cmd = cmd.replace('\t', ' ')
        os.system(f'tmux send-key -t train_session "{cmd}" Enter')
        return

    def t_tr_stop_tr_act(self):
        os.system("tmux send-keys -t train_session C-c")
        return


if __name__ == "__main__":
    import sys
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())