import cv2
import numpy

input_size = 640
stride = 32
palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                       [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                       [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                       [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                      dtype=numpy.uint8)
kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

def draw(frame,kps):
    image = frame.copy()
    shape = image.shape[:2]
    r = min(1.0, input_size / shape[0], input_size / shape[1])
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = input_size - pad[0]
    h = input_size - pad[1]
    w = numpy.mod(w, stride)
    h = numpy.mod(h, stride)
    w /= 2
    h /= 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image,
                               top, bottom,
                               left, right,
                               cv2.BORDER_CONSTANT)  # add border
    i = 0
    for x_coord,y_coord in kps:
        i=i+1
        color_k = [int(x) for x in kpt_color[i]]
        cv2.circle(frame,
                   (int(x_coord), int(y_coord)),
                   5, color_k, -1, lineType=cv2.LINE_AA)
    output_path = './results/test.jpg'
    cv2.imwrite(output_path, frame)