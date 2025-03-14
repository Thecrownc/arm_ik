import numpy as np
import json
import glob
import codecs
import os

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class tococo(object):
    def __init__(self, jsonfile, save_path, a):
        self.images = []
        self.categories = [
            {
                "supercategory": "yaban",
                "id": 1,
                "name": "yaban_1",
                "keypoints":
                    [
                        "1", "2",
                        "3", "4"
                    ],
                "skeleton": [
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 1]
                ]
            }
        ]
        self.annotations = []
        self.jsonfile = os.listdir(jsonfile)
        self.save_path = save_path  # 保存json的路径
        self.class_id = a  # class  我们的类别只有一个 person
        self.coco = {}
        self.path = jsonfile

    def labelme_to_coco(self):
        for num, json_file in enumerate(self.jsonfile):
            json_file = os.path.join(self.path, json_file)
            data = codecs.open(json_file, 'r')
            data = json.load(data)
            self.images.append(self.get_images(json_file[-17:-4] + 'jpg', data["imageHeight"], data["imageWidth"]))
            shapes = data["shapes"]
            annotation = {}  # 一个annotation代表一张图片中的所有samples
            num_keypoints = 0
            keypoints = [0] * 3 * 4  #这里是我们标注的关节点个数  如有改动，需要修改
            flag = 0
            for shape in shapes:
                if shape['shape_type'] == 'rectangle' or shape["label"] == 'bbox':
                    bbox = []
                    temp = shape["points"]
                    try:
                        x_min = min(temp[0][0], temp[1][0])
                    except IndexError as e:
                        print('class: {}, image: {}'.format(self.class_id, int(json_file[-17:-5])))

                    x_max = max(temp[0][0], temp[1][0])
                    y_min = min(temp[0][1], temp[1][1])
                    y_max = max(temp[0][1], temp[1][1])
                    bbox.append(x_min)
                    bbox.append(y_min)
                    w = x_max - x_min + 1
                    h = y_max - y_min + 1
                    bbox.append(w)
                    bbox.append(h)
                    annotation['bbox'] = bbox
                    flag = flag + 1
                else:
                    idx = int(shape['label'])

                    try:
                        keypoints[(idx - 1) * 3 + 0] = shape['points'][0][0]
                        keypoints[(idx - 1) * 3 + 1] = shape['points'][0][1]
                        keypoints[(idx - 1) * 3 + 2] = 2
                        num_keypoints = num_keypoints + 1
                    except IndexError as e:
                        print('class: {}, image: {}'.format(self.class_id, int(json_file[-17:-5])))

            if flag == 0:
                print('{}\\{} does not contain bbox\n'.format(self.class_id, json_file))
            annotation['segmentation'] = [[]]
            annotation['num_keypoints'] = num_keypoints
            annotation['iscrowd'] = 0
            annotation['keypoints'] = keypoints
            annotation['image_id'] = int(json_file[-17:-5])  # 对应的图片ID
            if 'bbox' not in annotation:
                annotation['bbox'] = [0, 0, data['imageWidth'], data['imageHeight']]
                annotation['area'] = 0
            else:
                annotation['area'] = int(bbox[2] * bbox[3])
            annotation['category_id'] = 1
            annotation['id'] = int(json_file[-17:-5])  # 对象id
            self.annotations.append(annotation)
            self.image_id = int(json_file[-17:-5])

        self.coco["images"] = self.images
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations



    def get_images(self, filename, height, width):
        image = {}
        image["height"] = height
        image['width'] = width
        image["id"] = int(filename[-16:-4])
        image["file_name"] = filename
        return image

    def get_categories(self, name, class_id):
        category = {}
        category["supercategory"] = "person"
        category['id'] = class_id
        category['name'] = name
        return category

    def save_json(self):
        self.labelme_to_coco()
        coco_data = self.coco
        # 保存json文件
        json.dump(coco_data, open(self.save_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示
        return self.image_id


json_path = r'/home/embodyrobotics/data/yolodata/xml'   #保存json的文件夹路径
c = tococo(json_path, save_path=r'/home/embodyrobotics/data/yolodata/xml/val.json', a=1)  #我们将我们的左右json文件合成为一个json文件，这是最后json文件的名称
image_id = c.save_json()

