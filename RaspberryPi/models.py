"""
얼굴탐지 마스크탐지 모델객체코드
얼굴탐지 마스크탐지 Reference : https://github.com/kairess/mask-detection
나이추정 Reference : https://github.com/yu4u/age-gender-estimation
얼굴탐지모델은 caffe모델 opencv로 로드
마스크탐지 및 나이추정 모델 tensorflow.keras로 로드
"""
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from wide_resnet import WideResNet


class FaceDetector:  # 얼굴탐지객체
    def __init__(self):  # 모델로드, 클래스변수설정
        self.facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
        self.dets, self.img = (None, None)
        self.face_num = 0
        self.x1, self.y1, self.x2, self.y2 = (0, 0, 0 ,0)
        self.height, self.width = (0, 0)

    def predict(self, img):  # 모델입력을 위한 이미지전처리 후 추정
        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        self.facenet.setInput(blob)
        self.dets = self.facenet.forward()
        self.height, self.width = img.shape[:2]

        return self.dets

    def get_face_number(self):  # 얼굴의 개수 반환
        num = 0
        for i in range(self.dets.shape[2]):
            confidence = self.dets[0, 0, i, 2]
            if confidence > 0.5:
                num += 1

        return num

    def set_xy_large_face(self):  # 얼굴 중 가장 넓이가 큰 좌표값 반환
        dic = {}
        for i in range(self.dets.shape[2]):
            confidence = self.dets[0, 0, i, 2]
            if confidence > 0.5:
                x1, y1, x2, y2 = self.get_xy_coordinate(i)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                dic[str(i)] = width * height

        index, size = sorted(dic.items())[0]
        self.x1, self.y1, self.x2, self.y2 = self.get_xy_coordinate(int(index))

        return self.x1, self.y1, self.x2, self.y2

    def get_xy_coordinate(self, n):  # XY 좌표값들을 추정결과 dets 에서 구함
        self.x1 = int(self.dets[0, 0, n, 3] * self.width)
        self.y1 = int(self.dets[0, 0, n, 4] * self.height)
        self.x2 = int(self.dets[0, 0, n, 5] * self.width)
        self.y2 = int(self.dets[0, 0, n, 6] * self.height)

        return self.x1, self.y1, self.x2, self.y2

    def convert_xy_square_coordinate(self):  # XY 좌표값을 정사각형 좌표로 바꿈(나이추정모델에서 정사각형좌표필요)
        width = abs(self.x2 - self.x1)
        height = abs(self.y2 - self.y1)
        diff = abs(height - width)

        if width > height:
            self.xs1 = int(self.x1 + diff / 2)
            self.xs2 = int(self.x2 - diff / 2)
            self.ys1 = self.y1
            self.ys2 = self.y2
        else:
            self.ys1 = int(self.y1 + diff / 2)
            self.ys2 = int(self.y2 - diff / 2)
            self.xs1 = self.x1
            self.xs2 = self.x2

        self.s_width = int(abs(self.xs2 - self.xs1))
        self.s_height = int(abs(self.ys2 - self.ys1))

        return int(self.xs1), int(self.ys1), int(self.xs2), int(self.ys2), int(self.s_width), int(self.s_height)

    def convert_xy_margin_coordinate(self, margin=0.4):  # 좌표값들을 늘림, 얼굴 전체가 좌표값안에 있어야 되기떄문
        self.xw1 = max(int(self.xs1 - margin * self.s_width), 0)
        self.yw1 = max(int(self.ys1 - margin * self.s_height), 0)
        self.xw2 = min(int(self.xs2 + margin * self.s_width), self.width - 1)
        self.yw2 = min(int(self.ys2 + margin * self.s_height), self.height - 1)

        return self.xw1, self.yw1, self.xw2, self.yw2

    def change_y_coordinate(self, n):  # y좌표를 n 만큼 이동시킨다
        self.yw1 = max(self.yw1+n, 0)
        self.yw2 = min(self.yw2+n, self.height - 1)

        return self.yw1, self.yw2


class MaskDetector:  # 마스크탐지 객체
    def __init__(self):  # 모델로드
        self.model = load_model('models/mask_detector.model')

    def predict(self, img):  # 이미지 전처리 후 모델로 추정, mask 착용확률리턴
        face_input = cv2.resize(img, dsize=(224, 224))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)
        mask, nomask = self.model.predict(face_input).squeeze()

        return mask, nomask

    def predicts(self, img, face_detector):  # 모든 얼굴을 돌면서 이미지에 사각형그리기
        mask, nomask = -1, -1
        result_img = img.copy()
        dets = face_detector.dets

        for i in range(dets.shape[2]):
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue

            self.x1, self.y1, self.x2, self.y2 = face_detector.get_xy_coordinate(i)

            face_img = img[self.y1:self.y2, self.x1:self.x2]

            mask, nomask = self.predict(face_img)

            if mask > nomask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)

            self.draw_label(img, label, color)

        return result_img, mask, nomask

    def draw_label(self, img, label, color):  # 라벨 그리기
        cv2.rectangle(img, pt1=(self.x1, self.y1), pt2=(self.x2, self.y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(img, text=label, org=(self.x1, self.y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)

        return img


class AgeEstimator:  # 나이인식객체
    def __init__(self, img_size=64, depth=16, k=8):
        self.age_model = WideResNet(img_size, depth=depth, k=k)()
        self.age_model.load_weights('models/weights.28-3.73.hdf5')
        self.img_size = img_size

    def predicts(self, img, face_detector):
        result_img = img.copy()
        dets = face_detector.dets

        for i in range(dets.shape[2]):
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue

            x1, y1, x2, y2 = face_detector.get_xy_coordinate(i)

            # cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face_detector.convert_xy_square_coordinate()
            xw1, yw1, xw2, yw2 = face_detector.convert_xy_margin_coordinate(margin=0.4)
            yw1, yw2 = face_detector.change_y_coordinate(-10)

            age_face = img[yw1:yw2, xw1:xw2]
            age_face = self.convert_predict_img(age_face)
            predicted_gender, predicted_age = self.predict(age_face)
            self.draw_label(result_img, xw1, yw1, xw2, yw2)

        return result_img, self.predicted_age, self.predicted_gender

    def predict(self, img):  # 모델 실행 후 성별과 나이값 반환
        result = self.age_model.predict(img)
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_age = result[1].dot(ages).flatten()
        predicted_gender = result[0]
        self.predicted_age = predicted_age[0]
        self.predicted_gender = predicted_gender[0][0]

        return self.predicted_gender, self.predicted_age

    def convert_predict_img(self, img):  # 이미지 전처리
        age_face = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        age_face = np.expand_dims(age_face, axis=0)

        return age_face

    def draw_label(self, img, xw1, yw1, xw2, yw2):
        label = "{}, {}".format(int(self.predicted_age), "M" if self.predicted_gender < 0.5 else "F")
        cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (0, 255, 0), 2)
        cv2.putText(img, text=label, org=(xw1, yw1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                    color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

