"""
실행파일
필요 파일 : models.py, gui_model.py, test.jpg, /warning_imgs
"""
import sys
import gui_model
from PyQt5.QtWidgets import QApplication


def run_program():  # 프로그램을 실행합니다.
    if __name__ == "__main__":
        app = QApplication(sys.argv)
        w = gui_model.GUI(0, 9600, 'COM3')
        sys.exit(app.exec_())


run_program()
