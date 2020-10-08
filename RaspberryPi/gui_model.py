"""
class GUI : GUI 프로그램를 위해 PyQt5 상속받아 정의
class Mymodel : 각 모델들의 객체를 생성하고 관리하는 MyModel Class 정의
"""
import time
import serial
import models
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

MASK_RATE_LIMIT = 0.5
UPPER_AGE_LIMIT = 50
LOWER_AGE_LIMIT = 25


class MyModel:  # 딥러닝 모델들을 관리하는 클래스
    def __init__(self):  # 딥러닝 모델객체 생성
        self.face_detector = models.FaceDetector()
        self.mask_detector = models.MaskDetector()
        self.age_estimator = models.AgeEstimator()
        self.limit_rate = MASK_RATE_LIMIT
        img = cv2.imread('test.jpg')
        self.predict(img)

    def predict_mask(self, img):  # 마스크인식 모델 메소드
        self.face_detector.predict(img)
        face_num = self.face_detector.get_face_number()
        if face_num == 0:
            return face_num, -1, -1, -1, -1

        x1, y1, x2, y2 = self.face_detector.set_xy_large_face()
        mask_face_img = img[y1:y2, x1:x2]
        mask, no_mask = self.mask_detector.predict(mask_face_img)

        if mask >= self.limit_rate:
            label = 'Mask %d%%' % (mask*100)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
        else:
            label = 'Mask %d%%' % (no_mask*100)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

        return 1, mask, no_mask, -1, -1

    def predict_age(self, img):  # 나이인식모델  메소드
        self.face_detector.predict(img)
        face_num = self.face_detector.get_face_number()
        if face_num == 0:
            return face_num, -1, -1, -1, -1
        x1, y1, x2, y2 = self.face_detector.set_xy_large_face()
        self.face_detector.convert_xy_square_coordinate()
        xw1, yw1, xw2, yw2 = self.face_detector.convert_xy_margin_coordinate(0.4)
        yw1, yw2 = self.face_detector.change_y_coordinate(-10)
        age_face_img = img[yw1:yw2, xw1:xw2]
        age_face_img = self.age_estimator.convert_predict_img(age_face_img)

        gender, age = self.age_estimator.predict(age_face_img)
        label = "No Mask {},{}".format(int(age), "M" if gender < 0.5 else "F")

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

        return 1, -1, -1, age, gender

    def predict(self, img):  # 사용 X
        self.face_detector.predict(img)
        face_num = self.face_detector.get_face_number()
        if face_num == 0:
            return face_num, -1, -1, -1, -1
        x1, y1, x2, y2 = self.face_detector.set_xy_large_face()
        self.face_detector.convert_xy_square_coordinate()
        xw1, yw1, xw2, yw2 = self.face_detector.convert_xy_margin_coordinate(0.4)
        yw1, yw2 = self.face_detector.change_y_coordinate(-10)
        mask_face_img = img[y1:y2, x1:x2]
        age_face_img = img[yw1:yw2, xw1:xw2]
        age_face_img = self.age_estimator.convert_predict_img(age_face_img)

        mask, nomask = self.mask_detector.predict(mask_face_img)

        if mask > self.limit_rate:
            label = 'Mask %d%%' % (mask*100)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
            return 1, mask, nomask, -1, -1
        else:
            gender, age = self.age_estimator.predict(age_face_img)
            label = "No Mask {},{}".format(int(age), "M" if gender < 0.5 else "F")
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (0, 255, 0), 2)
            # cv2.putText(img, text=label, org=(xw1, yw1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
            #             color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
            return 1, -1, -1, age, gender


class MySerial(serial.Serial):  # 시리얼통신객체
    def __init__(self, baudrate, port):
        serial.Serial.__init__(self, baudrate=baudrate, port=port)


class SerialThread(QThread):  # 시리얼통신을 위한 Q쓰레드 클래스
    # 쓰레드의 커스텀 이벤트
    # 데이터 전달 시 형을 명시해야 함
    threadEvent = pyqtSignal(str)

    def __init__(self, baudrate, port, parent=None):
        super().__init__()
        self.main = parent
        self.isRun = False
        self.my_serial = MySerial(baudrate=baudrate, port=port)

    def get_serial_data(self):  # 시리얼 데이터 지속 갱신
        if self.my_serial.readable():
            res = self.my_serial.readline()
            return res
        else:
            return None

    def write(self, data):  # 시리얼 데이터 write
        self.my_serial.write(data)

    def run(self):  # 쓰레드 실행
        while self.isRun:
            res = self.get_serial_data()
            if res is not None:
                res = res.decode()
                if 'pink' in res:
                    self.threadEvent.emit('old')
                elif 'white' in res:
                    self.threadEvent.emit('ordinary')
                elif 'blue' in res:
                    self.threadEvent.emit('youth')


class GUI(QWidget):  # 영상띄워주는클래스
    def __init__(self, video, baudrate, port):
        super().__init__()
        self.setWindowTitle('Smart Gate System')
        self.setGeometry(150, 150, 650, 540)
        self.init_ui()
        self.video = video
        self.baudrate = baudrate
        self.port = port
        self.mode = 'mask'
        self.warning_img = cv2.imread('warning_imgs/wait.jpg')
        self.start_detect, self.open_door, self.connected = False, False, False
        self.start_time = time.time()
        self.model = MyModel()
        self.face_num, self.mask_rate, self.nomask_rate, self.age, self.gender = -1, -1, -1, -1, -1
        self.res = None

    def init_ui(self):  # UI 생성
        self.ui_layout = QVBoxLayout()
        self.frame_layout = QHBoxLayout()
        self.btn_layout = QHBoxLayout()

        self.frame = QLabel(self)
        self.frame2 = QLabel(self)

        self.lbl = QLabel("Status Label", self)

        self.btn = QPushButton('Start')
        self.btn2 = QPushButton('Stop')
        self.btn3 = QPushButton('Detect')
        self.btn4 = QPushButton('Serial')
        self.btn5 = QPushButton('Mask_mode')
        self.btn.clicked.connect(self.start)
        self.btn2.clicked.connect(self.stop)
        self.btn3.clicked.connect(self.detect_btn)
        self.btn4.clicked.connect(self.serial_btn)
        self.btn5.clicked.connect(self.mode_btn)

        self.frame_layout.addWidget(self.frame)
        self.frame_layout.addWidget(self.frame2)

        self.btn_layout.addWidget(self.btn)
        self.btn_layout.addWidget(self.btn2)
        self.btn_layout.addWidget(self.btn3)
        self.btn_layout.addWidget(self.btn4)
        self.btn_layout.addWidget(self.btn5)

        self.ui_layout.addLayout(self.frame_layout)
        self.ui_layout.addLayout(self.btn_layout)
        self.ui_layout.addWidget(self.lbl)

        self.setLayout(self.ui_layout)

        self.show()

    @pyqtSlot()
    def threadStart(self):  # 시리얼 통신 쓰레드 시작
        if not self.serial_thread.isRun:
            print('메인 : 쓰레드 시작')
            self.serial_thread.isRun = True
            self.serial_thread.start()

    @pyqtSlot()
    def threadStop(self):  # 시리얼 통신 쓰레드 정지
        if self.serial_thread.isRun:
            print('메인 : 쓰레드 정지')
            self.serial_thread.isRun = False

    # 쓰레드 이벤트 핸들러
    # 장식자에 파라미터 자료형을 명시
    @pyqtSlot(str)
    def threadEventHandler(self, res):  # 쓰레드 이벤트 발생할때마다 실행
        print('받은 데이터 :', res)
        self.lbl.setText(res)
        self.res = res
        self.start_detect = True
        self.start_time = time.time()

    def start(self):  # start 버튼 이벤트함수
        self.cpt = cv2.VideoCapture(self.video)
        self.fps = self.cpt.get(5)
        _, self.img = self.cpt.read()

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(1000 / self.fps)

    def stop(self):  # stop 버튼 이벤트함수
        self.frame.setPixmap(QPixmap.fromImage(QImage()))
        self.frame2.setPixmap(QPixmap.fromImage(QImage()))
        self.timer.stop()
        self.cpt.release()

    def detect_btn(self):  # detect 버튼 이벤트함수
        self.start_detect = True
        self.start_time = time.time()

    def serial_btn(self):  # serial 버튼 이벤트함수
        try:
            # 쓰레드 인스턴스 생성
            self.serial_thread = SerialThread(self.baudrate, self.port)
            # 쓰레드 이벤트 연결
            self.serial_thread.threadEvent.connect(self.threadEventHandler)
            self.threadStart()
            self.connected = True
        except:
            print("시리얼통신 연결실패")

    def mode_btn(self):  # 나이인식 / 마스크인식 모드 스위치버튼
        if self.mode == 'mask':
            self.mode = 'age'
            self.btn5.setText("Age_mode")
        else:
            self.mode = 'mask'
            self.btn5.setText('Mask_mode')

    def is_timeover(self, set_time):  # 설정한 시간 지나면 True, detect 함수에서 start_time 초기화
        if (time.time() - self.start_time) > set_time:
            return True
        else:
            return False

    def draw_detection_area(self, img):  # detection area 표시
        h, w = img.shape[:2]
        ratio = 0.2
        x1, y1 = int(w * ratio), int(h * ratio)
        x2, y2 = int(w * (1 - ratio)), int(h * (1 - ratio))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, 'Face Detection Area', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

    def cv2_to_qpixmap(self, img):  # 이미지를 label 에 띄울수있게 이미지 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)

        return pix

    def serial_write(self, data):  # 시리얼 데이터 write
        if self.connected:
            self.serial_thread.write(data)
            print('send ', data)

    def next_frame(self):  # 설정한 timer 마다 실행되서 label 에 이미지 띄움
        self.timer.timeout.disconnect()

        _, img = self.cpt.read()
        img = cv2.resize(img, dsize=(0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
        
        # 카메라에 따라 좌우/상하반전 실행
        #img = cv2.flip(img ,0)
        #img = cv2.flip(img, 1)
        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_copy = img.copy()
        self.draw_detection_area(img_copy)

        pix = self.cv2_to_qpixmap(img_copy)
        self.frame.setPixmap(pix)

        age_mismatch = False
        if self.start_detect:
            if self.mode == 'mask':
                self.face_num, self.mask_rate, self.nomask_rate, self.age, self.gender = self.model.predict_mask(img)
            elif self.mode == 'age':
                self.face_num, self.mask_rate, self.nomask_rate, self.age, self.gender = self.model.predict_age(img)
                print(self.res, self.age)
                if (self.res == 'old') and (self.age < UPPER_AGE_LIMIT):
                    age_mismatch = True
                elif (self.res == 'youth') and (self.age > LOWER_AGE_LIMIT):
                    age_mismatch = True
                else:
                    age_mismatch = False

            print(self.face_num, self.mask_rate, self.nomask_rate, self.age, self.gender)
            detected_pix = self.cv2_to_qpixmap(img)
            self.frame2.setPixmap(detected_pix)
            txt = "Face_number:{}, Mask_rate:{:.3f}, No_mask_rate:{:.3f}, Age:{}".format(self.face_num, self.mask_rate,
                                                                                self.nomask_rate, int(self.age))
            self.lbl.setText(txt)

            if self.mode == 'mask':
                if self.face_num == 0:
                    self.warning_img = cv2.imread('warning_imgs/no_face.jpg')
                    self.serial_write(b'n')
                elif self.mask_rate >= 0.5:
                    self.warning_img = cv2.imread('warning_imgs/pass.jpg')
                    self.serial_write(b'y')
                elif self.mask_rate < 0.5:
                    self.warning_img = cv2.imread('warning_imgs/mask.jpg')
                    self.serial_write(b'n')
            elif self.mode == 'age':
                if self.face_num == 0:
                    self.warning_img = cv2.imread('warning_imgs/no_face.jpg')
                    self.serial_write(b'n')
                elif age_mismatch:
                    self.warning_img = cv2.imread('warning_imgs/age_mismatch.jpg')
                    self.serial_write(b'n')
                elif not age_mismatch:
                    self.warning_img = cv2.imread('warning_imgs/pass.jpg')
                    self.serial_write(b'y')

        time_over = self.is_timeover(5)
        if time_over:
            self.warning_img = cv2.resize(self.warning_img, dsize=(img.shape[1], img.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
            warning_pix = self.cv2_to_qpixmap(self.warning_img)
            self.frame2.setPixmap(warning_pix)

        self.timer.timeout.connect(self.next_frame)
        self.start_detect = False




