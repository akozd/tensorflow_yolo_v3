import os.path
import argparse
import time
import tensorflow as tf
import numpy as np
from models.yolo_v3 import yolo_v3
from utils.utils import process_image, get_anchors, get_classes, convert_box_coordinates, non_max_suppression, draw_boxes

def _main():

    # parse command line arguments
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument(
        '--path_to_input_image', type=str, required=True,
        help='The path to the input image on which object detection will be performed on.\n\
        This argument is required.')
    parser.add_argument(
        '--path_to_trained_model', default='model_weights/coco_pretrained_weights.ckpt',type=str,
        help="The path to the location of pretrained model weights, which will be loaded into\n\
        the model and then used for object detection. The default pretrained weights path is\n\
        'model_weights/coco_pretrained_weights.ckpt', which contains weights trained on\n\
        the coco dataset.")
    parser.add_argument(
        '--save_as', type=str, default=None,
        help='The filename for the image on which object detection was performed. If no filename\n\
        is provided, the image will be saved as "[original_name] + _yolo_v3.jpg".')
    parser.add_argument(
        '--tensorboard_save_path', default='tensorboard/tensorboard_detect/',
        help="")
    parser.add_argument(
        '--class_path', default='utils/coco_classes.txt', type=str,
        help='The path that points towards where the class names for the dataset are stored.\n\
        The default path is "utils/coco_classes.txt".')
    parser.add_argument(
        '--anchors_path', default='utils/anchors.txt', type=str,
        help='The path that points towards where the anchor values for the model are stored.\n\
        The default path is "utils/anchors.txt", which contains anchors trained on the coco dataset.')
    parser.add_argument(
        '--input_height', default=416, type=int,
        help='The input height of the yolov3 model. The height must be a multiple of 32.\n\
        The default height is 416.')
    parser.add_argument(
        '--input_width', default=416, type=int,
        help='The input width of the yolov3 model. The width must be a mutliple of 32.\n\
        The default width is 416.')
    args = vars(parser.parse_args())

    h = args['input_height']
    w = args['input_width']
    anchors = get_anchors(args['anchors_path'])
    classes = get_classes(args['class_path'])
    save_as = args['save_as']
    if save_as is None:
        filename_w_ext = os.path.basename(args['path_to_input_image'])
        filename, file_extension = os.path.splitext(filename_w_ext)
        save_as = filename + '_yolo_v3' + file_extension

    image, original_im = process_image(args['path_to_input_image'], h, w)

    tf.reset_default_graph()

    # build graph
    with tf.variable_scope('x_input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 3])
    
    yolo_outputs = yolo_v3(inputs=X, num_classes=len(classes), anchors=anchors, h=h, w=w, training=False) # output

    with tf.variable_scope('obj_detections'):
        raw_outputs = tf.concat(yolo_outputs, axis=1)

    # pass image through model
    with tf.Session() as sess:

        writer = tf.summary.FileWriter(args['tensorboard_save_path'],sess.graph)
        writer.close()

        saver = tf.train.Saver()
        print('restoring model weights...')
        saver.restore(sess, save_path=args['path_to_trained_model'])
        print('feeding image found at filepath: ', args['path_to_input_image'])
        start = time.time()
        ro = sess.run(raw_outputs, feed_dict={X:[np.array(image, dtype=np.float32)]})
        end = time.time()
        total_time = end-start
        print("total inference time was: "+ str(round(total_time,2)) + " seconds (that's " + str(round(60.0/total_time,2)) + " fps!)")
    
    # convert box coordinates, apply nms, and draw boxes
    boxes = convert_box_coordinates(ro)
    filtered_boxes = non_max_suppression(boxes, confidence_threshold=0.5,iou_threshold=0.4)
    draw_boxes(save_as,args['class_path'],filtered_boxes,original_im, image)
    
    print('image with detections saved as: ', save_as)

if __name__ == '__main__':
    _main()