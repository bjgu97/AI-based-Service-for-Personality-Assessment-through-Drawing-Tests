#from detectron2.utils.logger import setup_logger
#from detectron2.data.datasets import register_coco_instances
#from detectron2.engine import DefaultPredictor, DefaultTrainer
#from detectron2.evaluation import COCOEvaluator, inference_on_dataset
#from detectron2.utils.visualizer import Visualizer
#from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
#from detectron2.config import get_cfg
#from detectron2.data.detection_utils import build_transform_gen
#from detectron2.model_zoo import get_config_file
import os
import random
import cv2
import numpy as np
import glob
# Modules for classification
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize

count = 0


class Detector(object):
    def test_tree(img_dir):
        model = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

        cfg = get_cfg()
        cfg.merge_from_file(get_config_file(model))
        #cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
        cfg.SOLVER.IMS_PER_BATCH = 2
        # Learning Rate
        cfg.SOLVER.BASE_LR = 0.00025
        # Max Iteration
        cfg.SOLVER.MAX_ITER = 1000
        # Batch Size
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

        cfg.DATALOADER.NUM_WORKERS = 2

        # initialize from model zoo
        # 3 classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        print("HELLO WORLD END")
        
        #self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "./output2/model_final.pth")
        #self.cfg.MODEL.WEIGHTS = os.path.join("./output2/model_final.pth")
        cfg.MODEL.WEIGHTS = os.path.join("./output2/model_final.pth")
        cfg.DATASETS.TEST = (img_dir,)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

        predictor = DefaultPredictor(cfg)
        test_metadata = MetadataCatalog.get(img_dir)

        print(img_dir)
        for imageName in glob.glob(img_dir):
            print("test tree!!!!!!!!")
            print(imageName)
            im = cv2.imread(imageName)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=test_metadata,
                           scale=0.3
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow('result', out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Density attribute
        # function to count white percentage.
        img1 = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)

        count = 0

        # reshaping image
        scale_percent = 50
        width = int(img1.shape[1] * scale_percent / 100)
        height = int(img1.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)

        print('Resized Dimensions : ', img.shape)

        total = img.shape[0] * img.shape[1]
        # print(total)

        # using borders of image (20% each side)
        x1 = img.shape[0] / 5
        x2 = img.shape[0] - x1
        y1 = img.shape[1] / 5
        y2 = img.shape[1] - y1

        # axis x
        total = x1 * img.shape[1]
        count = getWhitePercent(img, total, 0, x1, 0, img.shape[1])
        count = getWhitePercent(img, total, x2, img.shape[0], 0, img.shape[1])

        # axis y
        total = y1 * img.shape[0]
        count = getWhitePercent(img, total, 0, img.shape[0], 0, y1)
        count = getWhitePercent(img, total, 0, img.shape[0], y2, img.shape[1])

        if count >= 3:
            print('image is dense')
            attr1 = "공간 밀도"
            attr2 = "꽉찬"
            adj1 = "근면하고"
            adj2 = "본능에 끌리지 않고"
            adj3 = "갇힌 감정 상태"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence00 = adj_sentence(attr1, attr2, adj, adj3)

        else:
            print('image is empty')
            attr1 = "공간 밀도"
            attr2 = "비어있는"
            adj1 = "민감하고"
            adj2 = "겸손하고"
            adj3 = "정신적으로 어떻게 행동해야 할지 모르는 상태"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence00 = adj_sentence(attr1, attr2, adj1, adj3)

        print(sentence00)
        final_sentence = sentence00 + "\r\n"


        # CROPPING DETECTED BOXES
        boxes = {}
        array = []
        # label name
        label_array = []

        for label in outputs["instances"].to("cpu").pred_classes:
            if label.item() == 0:
                label = "branches"
            elif label.item() == 1:
                label = "roots"
            elif label.item() == 2:
                label = "trunk"

            label_array.append(label)

        # coordinate
        for coordinates in outputs["instances"].to("cpu").pred_boxes:
            coordinates_array = []
            for k in coordinates:
                coordinates_array.append(int(k))
            array.append(coordinates_array)

        # label name + coordinates
        for i in range(len(label_array)):
            boxes[label_array[i]] = array[i]

        print("STEP1 BOXES: ",
              boxes)  # {'branches': [623, 826, 2086, 1853], 'trunk': [1075, 1796, 1459, 2269], 'roots': [733, 2228, 1549, 2439]}

        ################### EDITED: crop image ###################
        img_array = []
        for k, v in boxes.items():
            crop_img = im[v[1]:v[3], v[0]:v[2], :]
            # plt.imshow(crop_img, interpolation='nearest')
            # plt.show()

            cv2.imwrite(os.path.join("static/cropped_images", k + '.jpg'), crop_img)  # don't need this line
            img_array.append(crop_img)

        #print("IMG_ARRAY", img_array)
        #print("IMG_ARRAY LEN", len(img_array))  # 3(if root exists), 2(if root does not exist)
        if len(img_array) == 3:
            attr1 = "뿌리"
            attr2 = "뿌리가 보이는 상태인"
            adj1 = "원시성이 있고"
            adj2 = "전통과의 결부가 보여지고"
            adj3 = "정확성"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence0 = adj_sentence(attr1, attr2, adj, adj3)

            final_sentence = final_sentence + sentence0 + "\r\n"

        #else:
         #   sentence0 = ""

        #print(sentence0)

        for i in range(len(img_array)):
            plt.imshow(img_array[i])
            plt.show()

        # TESTING
        # TREES
        # for crown and branches
        img = img_array[0]

        bottle_resized = resize(img, (200, 200))
        bottle_resized = np.expand_dims(bottle_resized, axis=0)

        ######## crown shape ##########
        new_crown_shape_model = keras.models.load_model("crown_shape_model.h5")
        pred = new_crown_shape_model.predict(bottle_resized)

        if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
            attr1 = "관"
            attr2 = "아케이드 모양인"

            adj1 = "감수성이 있고"
            adj2 = "예의 바르고"
            adj3 = "의무감"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence1 = adj_sentence(attr1, attr2, adj, adj3)
            final_sentence = final_sentence + sentence1 + "\r\n"

        elif pred[0, 1] > pred[0, 2]:
            attr1 = "관"
            attr2 = "공 모양인"

            adj1 = "에너지가 부족하고"
            adj2 = "구성 감각이 결여되어있고"
            adj3 = "텅빈 마음"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence1 = adj_sentence(attr1, attr2, adj, adj3)
            final_sentence = final_sentence + sentence1 + "\r\n"
        #else:
        #    sentence1 = ""

        #print(sentence1)
        """
        if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
            attr1 = "관"
            attr2 = "아케이드 모양인"
            adj1 = "감수성이 있고"
            adj2 = "예의 바르고"
            adj3 = "의무감"
            sentence1 = adj_sentence(attr1, attr2, adj1, adj2, adj3)
        elif pred[0, 1] > pred[0, 2]:
            attr1 = "관"
            attr2 = "공 모양인"
            adj1 = "에너지가 부족하고"
            adj2 = "구성 감각이 결여되어있고"
            adj3 = "텅빈 마음"
            sentence1 = adj_sentence(attr1, attr2, adj1, adj2, adj3)
        elif pred[0, 2] > pred[0, 1]:
            pass
        """

        ######## crown shade ##########
        new_crown_shade_model = keras.models.load_model("crown_shade_model.h5")
        pred = new_crown_shade_model.predict(bottle_resized)
        # result = new_crown_shade_model.predict(img_array[0])

        if pred[0, 0] > pred[0, 1]:
            attr1 = "관"
            attr2 = "그림자 진"

            adj1 = "분위기에 좌우되고"
            adj2 = "정확성의 결여되어있고"
            adj3 = "부드러움"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence2 = adj_sentence(attr1, attr2, adj, adj3)

            final_sentence = final_sentence + sentence2 + "\r\n"


        #else:
        #    sentence2 = ""

        #print(sentence2)
        """
        if pred[0, 0] > pred[0, 1]:
            attr1 = "관"
            attr2 = "그림자 진"
            adj1 = "분위기에 좌우되고"
            adj2 = "정확성의 결여"
            adj3 = "부드러운"
            sentence2 = adj_sentence(adj1)
        """

        ######## crown fruit ##########
        new_fruit_model = keras.models.load_model("fruit_model.h5")
        pred = new_fruit_model.predict(bottle_resized)
        #print("Fruit: ")
        #print(pred)  # test with "treepic.jpg": [[0.08151636 0.0342663  0.8842173 ]]
        #print(np.around(pred))

        if pred[0, 0] > pred[0, 1]:
            attr1 = "관"
            attr2 = "과일이 매달려 있는"
            adj1 = "발달이 지체되어있고"
            adj2 = "자기 표현 능력이 결여되어있고"
            adj3 = "독립심의 결여"

            sentence3 = adj_sentence(attr1, attr2, adj, adj3)

            final_sentence = final_sentence + sentence3 + "\r\n"
        #else:
        #    sentence3 = ""

        #print(sentence3)
        """
        if pred[0, 0] > pred[0, 1]:
            attr = "Fix fruit: "
            adj1 = "지체된 발달"
            adj2 = "독립심이 없는"
            adj3 = "자기 표현 능력의 결여"
            sentence3 = adj_sentence(adj1)
        """

        ######## branch cut ##########
        new_cutbranch_model = keras.models.load_model("cut_branch_model.h5")
        pred = new_cutbranch_model.predict(bottle_resized)
        #print("Cut branch: ")
        #print(pred)  # test with "treepic.jpg": [[0.08151636 0.0342663  0.8842173 ]]
        #print(np.around(pred))  # [[0, 0, 1]]
        if pred[0, 0] > pred[0, 1]:
            attr1 = "가지"
            attr2 = "잘려있는"

            adj1 = "살려는 의지가 있고"
            adj2 = "억제되어있고"
            adj3 = "저항력"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence4 = adj_sentence(attr1, attr2, adj, adj3)

            final_sentence = final_sentence + sentence4 + "\r\n"
        #else:
        #    sentence4 = ""

        #print(sentence4)
        """
        if pred[0, 0] > pred[0, 1]:
            attr = "Cut branches: "
            adj1 = "저항력"
            adj2 = "억제된"
            adj3 = "살려는 의지"
            sentence4 = adj_sentence(adj1)
        """

        ################### EDITED: resize ###################
        img2 = img_array[1]
        bottle_resized2 = resize(img2, (200, 200))
        bottle_resized2 = np.expand_dims(bottle_resized2, axis=0)


        ######## trunk shape ##########
        new_trunk_shape_model = keras.models.load_model("trunk_shape_model.h5")
        pred = new_trunk_shape_model.predict(bottle_resized2)
        #print("Trunk shape: ")
        #print(pred)  # test with "treepic.jpg": [[0.08151636 0.0342663  0.8842173 ]]
        #print(np.around(pred))  # [[0, 0, 1]]
        if pred[0, 0] > pred[0, 1]:
            # attr = "Trunk base: "
            attr1 = "나무기둥"
            attr2 = "양쪽으로 넓은 모양인"

            adj1 = "봉쇄적 사고가 있고"
            adj2 = "이해가 느리고"
            adj3 = "학습곤란"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence5 = adj_sentence(attr1, attr2, adj, adj3)


        else:
            attr1 = "나무기둥"
            attr2 = "직선적인 모양인"

            adj1 = "규범적이고"
            adj2 = "고집이 세고"
            adj3 = "냉정함"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence5 = adj_sentence(attr1, attr2, adj, adj3)

        if sentence5:
            final_sentence = final_sentence + sentence5 + "\r\n"

        #print(sentence5)
        """
        if pred[0, 0] > pred[0, 1]:
            attr = "Trunk base: "
            adj1 = "봉쇄적 사고"
            adj2 = "이해가 느린"
            adj3 = "학습곤란이 있는"
            sentence5 = adj_sentence(adj1)
        else:
            attr = "Trunk straight: "
            adj1 = "냉정한"
            adj2 = "규범적인"
            adj3 = "정확한"
            sentence5 = adj_sentence(adj1)
        """


        ######## trunk wave  ##########
        new_trunk_wave_model = keras.models.load_model("trunk_wave_model.h5")
        pred = new_trunk_wave_model.predict(bottle_resized2)
        #print("Trunk wave: ")
        #print(pred)  # test with "treepic.jpg": [[0.08151636 0.0342663  0.8842173 ]]
        #print(np.around(pred))  # [[0, 0, 1]]
        if pred[0, 0] > pred[0, 1]:
            attr1 = "나무기둥"
            attr2 = "구불거리는 모양인"

            adj1 = "생동감이 있는"
            adj2 = "적응력이 큰"
            adj3 = "생기"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence6 = adj_sentence(attr1, attr2, adj, adj3)

            final_sentence = final_sentence + sentence6 + "\r\n"
        #else:
        #    sentence6 = ""


        #print(sentence6)
        """
        if pred[0, 0] > pred[0, 1]:
            attr = "Trunk wave: "
            adj1 = "생동감이 있는"
            adj2 = "적응력이 큰"
            adj3 = "생기가 있는"
            sentence6 = adj_sentence(adj1)
        """

        ######## trunk lines  ##########
        new_trunk_lines_model = keras.models.load_model("trunk_lines_model.h5")
        pred = new_trunk_lines_model.predict(bottle_resized2)
        #print("Trunk lines: ")
        #print(pred)  # test with "treepic.jpg": [[0.08151636 0.0342663  0.8842173 ]]
        #print(np.around(pred))  # [[0, 0, 1]]
        if pred[0, 0] > pred[0, 1]:
            attr1 = "나무기둥"
            attr2 = "흩어진 선으로 이루어진"

            adj1 = "예민하고"
            adj2 = "감정이입을 강하게 하는 경향이 있고"
            adj3 = "민감성"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence7 = adj_sentence(attr1, attr2, adj, adj3)

            final_sentence = final_sentence + sentence7 + "\r\n"
        #else:
        #    sentence7 = ""


        #print(sentence7)
        """
        if pred[0, 0] > pred[0, 1]:
            attr = "Trunk lines: "
            adj1 = "강한 감정이입"
            adj2 = "성격의 상실이 있는"
            adj3 = "영역을 분명하게 느끼지 못하는(나/너, 자아/대상)"
            sentence7 = adj_sentence(adj1)
        """

        ######## trunk shade  #########
        new_trunk_shade_model = keras.models.load_model("trunk_shade_model.h5")
        pred = new_trunk_shade_model.predict(bottle_resized2)
        #print("Trunk shade: ")
        #print(pred)  # test with "treepic.jpg": [[0.08151636 0.0342663  0.8842173 ]]
        #print(np.around(pred))  # [[0, 0, 1]]
        if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2] and pred[0, 0] > pred[0, 3]:
            # attr = "Trunk full shade: "
            attr1 = "나무기둥"
            attr2 = "전체에 명암이 있는"

            adj1 = "수동적이고"
            adj2 = "강박적이고"
            adj3 = "불안정감"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence8 = adj_sentence(attr1, attr2, adj, adj3)
            final_sentence = final_sentence + sentence8 + "\r\n"

        elif pred[0, 1] > pred[0, 2] and pred[0, 1] > pred[0, 0] and pred[0, 1] > pred[0, 3]:
            # attr = "Trunk right shade: "
            attr1 = "나무기둥"
            attr2 = "오른쪽에 그림자가 있는"

            adj1 = "접촉할 능력이 있고"
            adj2 = "접촉할 능력이 있고"
            adj3 = "적응력"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence8 = adj_sentence(attr1, attr2, adj, adj3)
            final_sentence = final_sentence + sentence8 + "\r\n"

        elif pred[0, 2] > pred[0, 0] and pred[0,2] > pred[0,1] and pred[0, 2] > pred[0, 3]:
            # attr = "Trunk left shade: "
            attr1 = "나무기둥"
            attr2 = "왼쪽에 그림자가 있는"

            adj1 = "외향적이고"
            adj2 = "억제하는 경향이 있고"
            adj3 = "민감성"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence8 = adj_sentence(attr1, attr2, adj, adj3)
            final_sentence = final_sentence + sentence8 + "\r\n"

        #print(sentence8)
        """
        if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2] and pred[0, 0] > pred[0, 3]:
            attr = "Trunk full shade: "
            adj1 = "우울증"
            adj2 = "불안정감"
            adj3 = "수동적"
            sentence8 = adj_sentence(adj1)
        elif pred[0, 1] > pred[0, 2] and pred[0, 1] > pred[0, 3]:
            attr = "Trunk right shade: "
            adj1 = "기꺼이 적응"
            adj2 = "접촉할 능력이 있음"
            adj3 = ""
            sentence8 = adj_sentence(adj1)
        elif pred[0, 2] > pred[0, 3]:
            attr = "Trunk left shade: "
            adj1 = "외향적 경향"
            adj2 = "상하기 쉬운"
            adj3 = "민감"
            sentence8 = adj_sentence(adj1)
        """

        ######## trunk tilt  #########
        new_trunk_tilt_model = keras.models.load_model("trunk_tilt_model.h5")
        pred = new_trunk_tilt_model.predict(bottle_resized2)
        #print("Trunk tilt: ")
        #print(pred)  # test with "treepic.jpg": [[0.08151636 0.0342663  0.8842173 ]]
        #print(np.around(pred))  # [[0, 0, 1]]
        if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
            # attr = "Trunk right tilt: "
            attr1 = "나무기둥"
            attr2 = "오른쪽으로 기울어진"

            adj1 = "집중을 잘하고"
            adj2 = "유혹에 빠지기 쉽고"
            adj3 = "민감성"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence9 = adj_sentence(attr1, attr2, adj, adj3)
            final_sentence = final_sentence + sentence9 + "\r\n"

        elif pred[0, 1] > pred[0, 2]:
            # attr = "Trunk left tilt: "
            attr1 = "나무기둥"
            attr2 = "왼쪽으로 기울어진"

            adj1 = "도전적이고"
            adj2 = "감정을 억누르는 경향이 있고"
            adj3 = "방어적 태도"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence9 = adj_sentence(attr1, attr2, adj, adj3)
            final_sentence = final_sentence + sentence9 + "\n"



        #else:
        #    sentence9 = ""

        #print(sentence9)
        """
        if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
            attr = "Trunk right tilt: "
            adj1 = "도전적인"
            adj2 = "방어적 태도"
            adj3 = "감정을 억누름"
            sentence9 = adj_sentence(adj1)
        elif pred[0, 1] > pred[0, 2]:
            attr = "Trunk left tilt: "
            adj1 = "민감"
            adj2 = "유혹에 빠지기 쉬운"
            adj3 = "쉽게 영향 받음"
            sentence9 = adj_sentence(adj1)
        """

        ######## trunk pattern  #########
        new_trunk_pattern_model = keras.models.load_model("trunk_pattern_model.h5")
        pred = new_trunk_pattern_model.predict(bottle_resized2)
        #print("Trunk pattern: ")
        #print(pred)  # test with "treepic.jpg": [[0.08151636 0.0342663  0.8842173 ]]
        #print(np.around(pred))  # [[0, 0, 1]]
        if pred[0, 1] > pred[0, 0] and pred[0, 1] > pred[0, 2]:
            # attr = "Trunk round pattern: "
            attr1 = "나무기둥"
            attr2 = "둥근 나무 껍질 무늬가 있는"

            adj1 = "접촉을 위한 준비 능력이 있고"
            adj2 = "접촉을 위한 준비 능력이 있고"
            adj3 = "자발적 적응 능력"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence10 = adj_sentence(attr1, attr2, adj, adj3)
            final_sentence = final_sentence + sentence10 + "\r\n"

        elif pred[0, 2] > pred[0, 1]:
            # attr = "Trunk scratch pattern: "
            attr1 = "나무기둥"
            attr2 = "긁힌 모양의 무늬가 있는"

            adj1 = "냉정하고"
            adj2 = "규범적이고"
            adj3 = "센 고집"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence10 = adj_sentence(attr1, attr2, adj, adj3)
            final_sentence = final_sentence + sentence10 + "\n"


        #else:
        #    sentence10 = ""

       # print(sentence10)
        """
        if pred[0, 1] > pred[0, 0] and pred[0, 1] > pred[0, 2]:
            attr = "Trunk round pattern: "
            adj1 = "접촉을 위한 준비 능력"
            adj2 = "자발적 적응"
            adj3 = ""
            sentence10 = adj_sentence(adj1)
        elif pred[0, 2] > pred[0, 1]:
            attr = "Trunk scratch pattern: "
            adj1 = "민감"
            adj2 = "고집스러운"
            adj3 = "관찰력이 뛰어난"
            sentence10 = adj_sentence(adj1)
        """


        ######## low branch #########
        new_lowbranch_model = keras.models.load_model("low_branch_model.h5")
        pred = new_lowbranch_model.predict(bottle_resized2)
        #print("Low branch: ")
        #print(pred)  # test with "treepic.jpg": [[0.08151636 0.0342663  0.8842173 ]]
        #print(np.around(pred))  # [[0, 0, 1]]
        if pred[0, 0] > pred[0, 1]:
            # attr = "Low branch: "
            attr1 = "나무기둥"
            attr2 = "가지가 있는"

            adj1 = "신뢰성이 없고"
            adj2 = "행동이 어린 아이 같고"
            adj3 = "부분적 발달 억제"

            adj_list = [adj1, adj2]
            adj = random.choice(adj_list)

            sentence11 = adj_sentence(attr1, attr2, adj, adj3)

            final_sentence = final_sentence + sentence11 + "\r\n"
       # else:
        #    sentence11 = ""


       # print(sentence11)

        #final_sentence = sentence1 +"\n"+ sentence2 + "\n"+sentence3 + "\n"+sentence4 + "\n"+sentence5 + "\n"+sentence6 + "\n"+sentence7 + "\n"+sentence8 +"\n"+\
        #                 sentence9 + "\n"+sentence10 + "\n"+sentence11
        return final_sentence
        """
        if pred[0, 0] > pred[0, 1]:
            attr = "Low branch: "
            adj1 = "신뢰성이 없는"
            adj2 = "행동이 어린 아이 같은"
            adj3 = "부분적 발달 억제"
            sentence11 = adj_sentence(adj1)
        """



        # return outputs
        """
        predictor = DefaultPredictor(self.cfg)
        im = cv2.imread(img_dir)
        outputs = predictor(im)

        # with open(self.curr_dir+'/data.txt', 'w') as fp:
        # 	json.dump(outputs['instances'], fp)
        # 	# json.dump(cfg.dump(), fp)

        # get metadata

        test_metadata = DatasetCatalog.get("train_dataset")

        # visualise
        v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.4)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow('result', v.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        return v
        """

    def test_cat(self, img_dir):

        #self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.WEIGHTS = os.path.join("./output3/model_final.pth")
        self.cfg.DATASETS.TEST = (img_dir,)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

        predictor = DefaultPredictor(self.cfg)
        test_metadata = MetadataCatalog.get(img_dir)

        for imageName in glob.glob(img_dir):
            im = cv2.imread(imageName)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=test_metadata,
                           scale=0.3
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow('result', out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # CAT TEST

        left = 0
        right = 0

        bottle_resized = resize(im, (200, 200))
        bottle_resized = np.expand_dims(bottle_resized, axis=0)

        new_concept_model = keras.models.load_model("concept_model.h5")
        pred = new_concept_model.predict(bottle_resized)
        if pred[0, 0] > pred[0, 1]:
            print("RIGHT HEMISPHERE")
            right+=1
        else:
            print("LEFT HEMISPHERE")
            left+=1

        new_movement_model = keras.models.load_model("movement_model.h5")
        pred = new_movement_model.predict(bottle_resized)
        if pred[0, 0] > pred[0, 1]:
            print("LEFT HEMISPHERE")
            left += 1
        else:
            print("RIGHT HEMISPHERE")
            right += 1

        c = image_colorfulness(im)
        if c > 50:
            print("COLORFUL CAT - RIGHT HEMISPHERE")
            right += 1
        else:
            print("COLORLESS CAT - LEFT HEMISPHERE")
            left += 1

        print("RIGHT: ", right)
        print("LEFT: ", left)

        # Cat pie result
        plt.rcParams['figure.figsize'] = [12, 8]

        group_names = ['right brain', 'left brain']
        group_sizes = [right, left]
        group_colors = ['red', 'blue']

        plt.pie(group_sizes,
                labels=group_names,
                colors=group_colors,
                autopct='%1.2f%%',  # second decimal place
                shadow=True,
                startangle=90,
                textprops={'fontsize': 14})  # text font size

        plt.axis('equal')  # equal length of X and Y axis
        plt.title('Your brain is?', fontsize=20)
        plt.savefig('./static/images/new_plot.jpg')

        plt.show()

        result = ""
        if right > left:
            result = "당신은 우뇌입니다"
        elif left > right:
            result = "당신은 좌뇌입니다"
        return result
# sentence forming
# def adj_sentence(attr, adj1, adj2, adj3) HOW ABOUT DOING IT LIKE THIS FOR EACH ATTRIBUTE?
def adj_sentence(attr1, attr2, adj, adj3):
    sentences = ["%s이 %s인 것으로 보아 당신은 %s이고 %s이 있는 것으로 보여집니다.", "%s이(가) %s로 나타남으로써 당신은 %s이고 %s 경향이 보여집니다.",\
                 "당신의 그림의 %s이(가) %s인 것으로 보아 당신은 %s이고 %s이 있어보입니다.",
                 "당신이 그린 그림을 보니 %s이(가) %s인 것을 보아 %s이고 %s를 내제하고 있는 것이 보여집니다."]
    sentence = random.choice(sentences)

    # conj = ["그리고", "또한", "또, ", "더불어, ", ""]
    # end = ["마지막으로", "최종적으로"]

    # sentence1 = random.choice(sentences)
    # conj1 = random.choice(conj)
    # end1 = random.choice(end)

    # sentence = sentence1 % adj
    sentence = sentence % (attr1, attr2, adj, adj3)
    return sentence


def getWhitePercent(img, total, x0, x1, y0, y1):
    global count
    white = 0
    other = 0

    for x in range(int(x0), int(x1)):
        for y in range(int(y0), int(y1)):
            #print("img[x][y]: ", img[x][y])
            if (img[x][y] >= 245).all():
                white += 1
            else:
                other += 1

    percentage = white * 100 / total
    # print(percentage)

    if percentage < 85:
        count += 1
    else:
        count += 0

    return count

# Cat colorfulness
def image_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)  # Red channel -  Blue channel
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)