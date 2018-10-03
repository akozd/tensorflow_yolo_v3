import numpy as np
import cv2

def draw_boxes(output_filename, classes_filename, inputs, original_image, resized_image):
    """
    Draws identified boxes along with class probabilities on the original image
    and then saves the image with the output file name.

    Parameters
    ----------
    output_filename : string
        The name that the image with detected objects should be saved as.
    classes_filename : string
        A binary file that contains the name of the classes.
    inputs : dictionary

    original_image : ndarray
        An array of shape:
        [image height, image width, 3]
        The original image simply loaded into a numpy array with a RGB color profile. 
    resized_image : ndarray
        An array of shape:
        [input_height, input_wdith, 3]
        The array is divided by 255.0 in order to turn the pixel values into numbers between zero 
        and one. Since cv2 load images in BGR, the array is also converted to a RGB color profile.
    """
   
    names = {}
    with open(classes_filename) as f:
        class_names = f.readlines()
        for id, name in enumerate(class_names):
            names[id] = name

    height_ratio = original_image.shape[0] / resized_image.shape[0]
    width_ratio = original_image.shape[1] / resized_image.shape[1]
    ratio = (width_ratio, height_ratio)

    for object_class, box_coords_and_prob in inputs.items():
        for box_coord, object_prob in box_coords_and_prob:

            box_coord = box_coord.reshape(2,2) * ratio
            box_coord = box_coord.reshape(-1)

            x0y0 = (int(box_coord[0]),int(box_coord[1]))
            x1y1 = (int(box_coord[2]), int(box_coord[3]))

            textx0y0 = (x0y0[0],x0y0[1]-4)

            cv2.rectangle(original_image, x0y0, x1y1, (255,255,255), 3)
            text_label = str(names[object_class])[:-1] + ", " + str(round(object_prob*100,2)) + "%"
            cv2.putText(original_image, text_label, textx0y0, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

    cv2.imwrite(output_filename, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    """
    Applies non-max suppression to predicted boxes.

    Parameters
    ----------
    predictions_with_boxes : ndarray
        An array of shape:
        [1, num_large_obj_detectors + num_med_obj_detectors + num_small_obj_detectors, num_classes + 5]
        where num_x_obj_detectors = num_anchors_per_layer * yolo_layer_grid_w * yolo_layer_grid_h.
    confidence_threshold : float
        A number between zero and one which indicates the minimum object confidence prediction necessary
        for a particular box's data to not be thrown out. For example, the confidence threshold might be
        set to 0.7 and a detector might predict a box with confidence of 0.8. This detector's box data will
        therefore be put in the 'result' dictionary since it is above the confidence threshold.
    iou_threshold : float
        The threshold for deciding if two boxes overlap.

    Returns
    -------
    result : dictionary
        A dictionary of structure: 
        {unique_class_index : [(box_1_data, box_1_prob),(box_2_data, box_2_prob)], etc...}
        where unique_class_index is the index that corresponds with the class's name, 
        box_x_data is a ndarray of size [4] that contains the box information associated 
        with the class index, and box_x_prob is a float that gives the probability of the box
        being in fact the identified class.
    """

    def iou(box1, box2):
        """
        Calculates the intersection over union (IOU) of two bounding boxes, which is the 
        ratio of the area where the two boxes overlap to the joint area of the boxes as a whole.
        Two perfectly overlapping boxes will have an IOU of 1, while two boxes that don't 
        overlap one another at all will have an IOU of 0.

        Parameters
        ----------
        box1 : ndarray
            Array of shape [x_min, y_min, x_max, y_max].
        box2 : ndarray
            Array of shape [x_min, y_min, x_max, y_max].
      
        Returns
        -------
        iou : float
            The IOU result of the two boxes.
        """

        b1_x0, b1_y0, b1_x1, b1_y1 = box1
        b2_x0, b2_y0, b2_x1, b2_y1 = box2

        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        iou = int_area / (b1_area + b2_area - int_area + 1e-05)

        return iou
    
    conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])
        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)
    
        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                ious = np.array([iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result

def convert_box_coordinates(detections):
    """
    Converts coordinates in the form of center_x, center_y, width, height to 
    min_x, min_y, max_x, max_y. The coordinate values are already scaled up to 
    the input dimension shapes.

    Parameters
    ----------
    detections : ndarray
        An array of shape:
        [1, num_large_obj_detectors + num_med_obj_detectors + num_small_obj_detectors, num_classes + 5]
        where num_x_obj_detectors = num_anchors_per_layer * yolo_layer_grid_w * yolo_layer_grid_h. 

    Returns
    -------
    detections : ndarray
        The original detections array with converted coordinates.
    """

    split = np.array_split(detections, [1, 2, 3, 4, 85], axis=2)
    center_x = split[0]
    center_y = split[1]
    width = split[2]
    height = split[3]
    attrs = split[4]
    
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = np.concatenate([x0, y0, x1, y1], axis=-1)
    detections = np.concatenate([boxes, attrs], axis=-1)
    
    return detections

def process_image(image_path, input_height, input_width):
    """
    Takes any image and transforms it into the format needed for object detection with yolov3.

    Parameters
    ----------
    image_path : string
        Path that points to where the image on which object detection should be performed is stored.
    input_height : int
        The height of the input that will be fed into the yolov3 model.
    input_width : int
        The width of the input that will be fed into the yolov3 model.
    
    Returns
    -------
    resized_image : ndarray
        An array of shape:
        [input_height, input_wdith, 3]
        The array is divided by 255.0 in order to turn the pixel values into numbers between zero 
        and one. Since cv2 load images in BGR, the array is also converted to a RGB color profile.
    image : ndarray
        An array of shape:
        [image height, image width, 3]
        The original image simply loaded into a numpy array with a RGB color profile.
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image,(input_width,input_height))
    resized_image = resized_image / 255.0

    return resized_image, image

def rand(a=0, b=1):
    '''Returns a random number between a and b.'''
    return np.random.rand()*(b-a) + a

def get_classes(classes_path):
    """
    Reads the class names from a file and returns them in a list.

    Parameters
    ----------
    classes_path : string
        Path that points to where the class name information is stored.

    Returns
    -------
    class_names : list
        A list of format:
        ['class_1', 'class_2', 'class_3', ...]
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    """
    Reads the anchors from a file and returns them in a list.

    Parameters
    ----------
    anchors_path : string
        Path that points to where the anchor information is stored.

    Returns
    -------
    anchors : list
        A list of format:
        [[anchor1_width, anchor1_height], [anchor2_width, anchor2_height], [anchor3_width, anchor3_height], ...]
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = list(zip(anchors[::2], anchors[1::2]))

    return anchors

def prepare_data(annotations_path, training_validation_split=0.9, batch_size=32, overfit_model=False):
    """
    Takes the raw data from the text file and splits it up into a training set
    and a validation set based on the train/val split hyperparameter. If the model
    is in overfit mode, the only data prepared will be the first image recorded in 
    the text file.

    Parameters
    ----------
    annotations_path : string
        The path which points to the text file with image and box data.
        The file structures data in the following way:

        image_data/images/img1.jpg 50,100,150,200,0 30,50,200,120,3
        image_data/images/img2.jpg 20,100,123,157,70 30,77,200,120,21 44,50,60,60,2

        Or to put it in more abstract terms:
        path/to/first/training/image min_x,min_y,max_x,max_y,class_num, etc..
        (up to max_num_boxes_per_image of boxes per image)
    training_validation_split : float
        The percentage of the data that should be kept for training 
        versus validation. A value of '1.0' would indicate that all
        of the data will be used to train the model. 
    batch_size : int
        The batch size to be used per training step. If the model is training
        to overfit, the batch_size parameter will be overwritten and a batch size
        of 1 will be prescribed. In overfit mode, the model continuously only trains
        on one image over and over again. The model will choose to overfit on the 
        first image listed in the text file containing the training data image paths
        and box information.
    overfit_model : bool
        Whether or not to train the model only on one image and to purposefully 
        overfit it. This can be useful to see whether or not the loss function and
        hyperparameters have been set up correctly.

    Returns
    -------
    training_data : list
        A sublist of the entire training data set to be used for training.
    validation_data : list
        A sublist of the entire training data set to be used for validation.
    batch_size : int
        The size of the batch to be used with every training step.
    """
    
    with open(annotations_path) as f:
        lines = f.readlines()
    if overfit_model:
        training_data = lines[0:1]
        validation_data = None
        train_batch_size = 1
    else:
        np.random.shuffle(lines)
        training_data = lines[:int(len(lines) * training_validation_split)]
        validation_data = lines[len(training_data):]
        train_batch_size = min(len(training_data), batch_size)
        val_batch_size = min(len(validation_data), batch_size)
        batch_size = min(train_batch_size, val_batch_size)
        training_data = training_data[:batch_size]
        validation_data = validation_data[:batch_size]
    
    return training_data, validation_data, batch_size

def augment_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    """
    Takes the iamge and box data and applies random transformations if the 'random' parameter is set
    to true. Otherwise, the image and box data will only be reshaped to fit the input tensor size. 

    Parameters
    ----------
    annotation_line : list
        a list in the format of ['path/to/img1.jpg', '50,50,150,200,0', '10,70,200,120,3']
    input_shape : tuple
        The height and the width of the yolov3 input shape.
    random : bool
        Whether or not random distortions should be applied to the image and box data versus
        just general resizing to fit the input shape.
    max_boxes : int
        The max number of possible boxes marking objects in the image.
    jitter : float
        Random amount to shift the image.
    hue : float
        Affect the hue of an image.
    sat : float
        Affect the saturation of an image.
    val : float
        Affects the brightness of the image.
    proc_img : bool
        When set to true, the new image will be shifted around a bit and divided
        by 255.0.
    
    Returns
    -------
    image_data : ndarray
        Array of shape [w, h, 3] containing the image data. Inputs are divided by 255.0.
    box_data : ndarray
        Array of shape [max_boxes, 5] 
        where the '5' represents the min x coordinate, min y coordinate, max x coordinate, 
        max y coordinate, and the box's class number. The box coordinates are fully scaled numbers
        relative to the original image size. If there are are not enough boxes to fit up the
        box_data tensor (for example: the image only contains 5 boxes but the max number of boxes
        per image is 20), then the empty entries are simply filled with zeros.
    """
    
    line = annotation_line.split()
    image = cv2.imread(line[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw, _ = image.shape
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
            new_image = np.ones((w, h, 3), dtype=np.uint8)*128
            new_image[dy:dy+nh, dx:dx+nw, :] = image
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = np.ones((w, h, 3), dtype=np.uint8)*128
    
    idx, idy, inw, inh = dx, dy, nw, nh
    if dx < 0:
        image = image[:,abs(dx):,:]
        idx = 0
    if w < nw + dx:
        image = image[:,:w,:] 
        inw = w
    if dy < 0:
        image = image[abs(dy):,:,:]
        idy = 0
    if h < nh + dy:
        image = image[:h,:,:]
        inh = h

    new_image[idy:idy+inh, idx:idx+inw, :] = image
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = cv2.flip(image, flipCode=1)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)/255.0
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_HSV2RGB) 

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data

def create_y_true(box_data, anchors, num_classes, h=416, w=416):
        """   
        A wrapper function for creating the full y_true and y_true_box_data numpy arrays used for 
        training the yolov3 model. 

        Parameters
        ----------
        box_data : ndarray
            A numpy array of shape:
            [batch_size, max_num_true_boxes_per_image, 5]
            where the '5' represents the min x coordinate, min y coordinate, max x coordinate, 
            max y coordinate, and the box's class number. The box coordinates are fully scaled numbers
            relative to the original image size.
        anchors : list
            A list of anchors with format:
            [[anchor1_width, anchor1_height], [anchor2_width, anchor2_height], [anchor3_width, anchor3_height], ...]
            The anchors are necessary for calculating an IOU with the box data to determine into which layer
            a particular box's data should be placed into. Large boxes will have high IOU's with large anchors, and
            therefore they will all be grouped into the same layer that detects large objects. On the other hand,
            small boxes will have high IOU's with small anchors, and therefore will be grouped into the layer
            responsible for detecting small objects. 
        num_classes : int
            The number of classes in the training data.
        h : int
            Height of the input.
        w : int
            Width of the input
        
        Returns
        -------
        y_true : ndarray
            The complete y_true array of shape:
            [batch_size, num_large_obj_detectors + num_medium_obj_detectors + num_small_obj_detectors, 5 + num_classes]
            where the '5' represents the center_x, center_y, width, height coordinates all as percentages of the
            original image size. 
            num_x_obj_detectors = num_anchors_per_layer * x_grid_height * x_grid_width
            The y_true numpy array is shaped like this for easy loading into the feed dictionary.

        y_true_box_data : ndarray
            the complete y_true_box_data array of shape:
            [batch_size, max_num_true_boxes_per_image * num_layers, 4]
            The y_true_box_data numpy array is shaped like this for easy loading into the feed dictionary.
        """ 

        def load_data(box_data, best_anchors, anchors_mask, grid_size, num_classes):
            """   
            Takes the box_data and maps it into the y_true numpy array for a particular
            grid size. The yolov3 model has three grid sizes for large, medium, and small.
            object detection. The mapping function used is a fully vectorized implementation 
            that does not use any loops whatsoever.

            Parameters
            ----------
            box_data : ndarray
                A numpy array of shape:
                [batch_size, max_num_true_boxes_per_image, 5]
                where the '5' represents the center x coordinate, center y coordinate, the width,
                the height, and the box's class number. The box coordinates are percentages of 
                the original image size. 
            best_anchors : ndarray
                index of best anchor
                A numpy array of shape:
                [batch_size, max_num_true_boxes_per_image]
                At every column index, each individual box stores the index of the anchor with which
                it has the highest IOU (intersection over union) value. 
            anchors_mask : list
                identifies which anchors should be used with this layer. If the best_anchors numpy
                array contains anchor indices that are not part of this layer (as determined by the
                anchors mask) they will be ignored.
            grid_size : tuple
                The size of this layer's grid. Will coincide with the grid sizes of yolov3's 
                yolo layers.
            num_classes : int
                The number of classes in the training data.

            Returns
            -------
            y_true : ndarray
                A numpy array of shape:
                [batch_size, grid_h * grid_w * num_anchors_per_layer, num_classes + 5]
                This array is the y_true for a particular grid size.
            box_data : ndarray
                A numpy array of shape:
                [batch_size, max_num_true_boxes_per_image, 4]
                The data for boxes whos highest IOU values coincide with anchors not belonging to 
                this particular layer have been set to zero. Only box data that belongs to the layer
                remains in the array.
            """
            
            num_anchors = len(anchors_mask)
            box_data_shape = box_data.shape

            # remove all anchors that aren't part of this layer
            best_anchors_mask = np.isin(best_anchors, anchors_mask, invert=True)
            best_anchors = best_anchors*1
            best_anchors -= min(anchors_mask)
            best_anchors[best_anchors_mask] = 0

            # set all of the box data that isn't part of this layer to zero
            box_data_mask = np.ones_like(best_anchors)
            box_data_mask[best_anchors_mask] = 0
            box_data_mask = np.expand_dims(box_data_mask, -1)
            box_data = box_data*box_data_mask
            
            i = np.floor(box_data[:,:,1]*grid_size[0]).astype('int32')
            j = np.floor(box_data[:,:,0]*grid_size[1]).astype('int32')
            
            # reshape all of these arrays for vectorized ops
            box_data = box_data.reshape([-1,box_data.shape[-1]])
            best_anchors = best_anchors.reshape([-1,1])
            i = i.reshape([-1,1])
            j = j.reshape([-1,1])

            # create one-hot class encodings
            classes = box_data[:,-1].reshape([-1]).astype(np.int)
            one_hot_array = np.zeros([box_data.shape[0],num_classes])
            one_hot_array[np.arange(box_data.shape[0]),classes] = 1

            box_data_mask = box_data[:,2]>0
            box_data[box_data_mask,4] = 1
            box_data = np.concatenate([box_data,one_hot_array],axis=-1)
        
            y_true = np.zeros([box_data_shape[0] * int(grid_size[0]) * int(grid_size[1]) * num_anchors, 5+num_classes])
            
            image_offset = np.repeat(np.linspace(0, y_true.shape[0], box_data_shape[0], endpoint=False, dtype=np.int), box_data.shape[0] / box_data_shape[0]).reshape([-1,1])
            grid_offset = num_anchors * (grid_size[0] * i + j)

            indexing_array = np.array(image_offset + grid_offset + best_anchors,dtype=np.int32)
            indexing_array = indexing_array[box_data_mask,:]
            indexing_array = indexing_array.reshape([-1])

            y_true[indexing_array,:] = box_data[box_data_mask]
            y_true = y_true.reshape([box_data_shape[0], int(grid_size[0]) * int(grid_size[1]) * num_anchors, num_classes+5])
            box_data = box_data.reshape([box_data_shape[0],box_data_shape[1],-1])

            return y_true, box_data[...,0:4]

        # convert from (min_x, min_y, max_x, max_y) to (center_x, center_y, width, height)
        anchors = np.array(anchors)
        boxes_xy = (box_data[:,:,0:2] + box_data[:,:,2:4]) // 2
        boxes_hw = box_data[:,:,2:4] - box_data[:,:,0:2]
        # change box coordinates to be percentages of the image size
        box_data[:, :, 0] = boxes_xy[...,0]/w
        box_data[:, :, 1] = boxes_xy[...,1]/h
        box_data[:, :, 2] = boxes_hw[...,0]/w
        box_data[:, :, 3] = boxes_hw[...,1]/h

        hw = np.expand_dims(boxes_hw, -2)
        anchors_broad = np.expand_dims(anchors, 0)
        anchor_maxes = anchors_broad / 2.
        anchor_mins = -anchor_maxes 
        box_maxes = hw / 2.
        box_mins = -box_maxes
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_hw = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]
        box_area = hw[..., 0] * hw[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchors = np.argmax(iou, axis=-1)
        large_obj_detectors, large_obj_boxes = load_data(box_data, best_anchors=best_anchors, anchors_mask=[6,7,8], grid_size=(h/32,w/32), num_classes=num_classes)
        medium_obj_detectors, medium_obj_boxes = load_data(box_data, best_anchors=best_anchors, anchors_mask=[3,4,5], grid_size=(h/16,w/16), num_classes=num_classes)
        small_obj_detectors, small_obj_boxes = load_data(box_data, best_anchors=best_anchors, anchors_mask=[0,1,2], grid_size=(h/8,w/8), num_classes=num_classes)
        
        y_true = np.concatenate([large_obj_detectors, medium_obj_detectors, small_obj_detectors], axis=1)
        y_true_box_data = np.concatenate([large_obj_boxes, medium_obj_boxes, small_obj_boxes], axis=1)
        
        return y_true, y_true_box_data

def get_training_batch(annotation_lines, anchors, num_classes, batch_size=32, h=416, w=416, random=False):
    """
    Takes the annotion lines, reads them, and from their information constructs the necessary 
    numpy arrays that store the data to train the yolov3 model.

    Parameters
    ----------
    annotation_lines : list
        A list of format
        ['image_data/images/img1.jpg 22,748,2184,2150,2 1590,2140,1832,2414,32 2414,858,2750,2002,0', ...
        The box data is of format min_x, min_y, max_x, max_y, class_number and is relative to the 
        original image size.
    anchors : list
        A list of format:
        [[anchor1_width, anchor1_height], [anchor2_width, anchor2_height], [anchor3_width, anchor3_height], ...]
    batch_size : int
        The amount of images to use in each batch per training step.
    h : int
        Height of the input shape into the yolov3 model.
    w : int
        Width of the input shape into the yolov3 model.
    
    Returns
    -------
    image_data : ndarray
        An array of shape [batch_size, h, w, 3]
        The image pixel data has been divided by 255.0 so that all values are between
        zero and one.
    y_true : ndarray
        An array containing the ground truth box coordinate and class information used for training 
        and calculating the loss of the yolo_v3 model. 
        A sample y_true array would be of shape:
        [batch_size, num_large_obj_detectors + num_med_obj_detectors + num_small_obj_detectors, num_classes + 5]
        where num_x_obj_detectors = num_anchors_per_layer * yolo_layer_grid_w * yolo_layer_grid_h.
    y_true_box_data : ndarray
        An array containing ground truth box data.
        A sample y_true_boxes array would be of the shape:
        [batch_size, num_anchors_per_layer * max_num_true_boxes_per_image, 5]
    """

    anchors = np.array(anchors,dtype=np.float32)
    image_data = []
    box_data = []
    
    for b in range(batch_size):
        if b==0:
            np.random.shuffle(annotation_lines)
        
        image, box = augment_data(annotation_lines[b], (h, w), random=random, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True)
        image_data.append(image)
        box_data.append(box)

    image_data = np.array(image_data)
    box_data = np.array(box_data)

    y_true, y_true_box_data = create_y_true(box_data, anchors, num_classes, h, w)

    return image_data, y_true, y_true_box_data