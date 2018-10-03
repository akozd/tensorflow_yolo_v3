import tensorflow as tf

def yolo_v3_loss(yolo_outputs, y_true, y_true_boxes, ignore_threshold, anchors, num_classes, h, w, batch_size):
    """
    A wrapper function that returns the loss associated with a forward pass of the yolo_v3 model.
    The main purpose of this function is to extract data from yolo_outputs, y_true, and y_true_boxes,
    which can then be fed sequentially into the loss_per_scale function, calculating the loss associated
    with each yolo layer scale.

    Parameters
    ----------
    yolo_outputs : tuple
        A tuple containing the results of a forward pass of a training yolo_v3 model. 
        The contents of the tuple will be:
        [large_object_box_detections, medium_object_box_detections, small_object_box_detections, 
        large_object_raw_detections, medium_object_raw_detections, small_object_raw_detections] 
        The elements of the list will be tensors of shape:
        [batch_size, yolo_layer_grid_h * yolo_layer_grid_w * num_anchors_per_layer, num_classes + 5]
    y_true : tensor
        A tensor containing the ground truth box coordinate and class information used for training 
        and calculating the loss of the yolo_v3 model. 
        A sample y_true tensor would be of shape:
        [batch_size, num_large_obj_detectors + num_med_obj_detectors + num_small_obj_detectors, num_classes + 5]
        where num_x_obj_detectors = num_anchors_per_layer * yolo_layer_grid_w * yolo_layer_grid_h.
        The heights and widths of the yolo layers will vary depending on if the layer is meant to detect
        large, medium, or small objects. The large, medium, and small y_true data is extracted from this tensor
        and then eventually reshaped to be in the more standard form of:
        [batch_size, yolo_layer_grid_h, yolo_layer_grid_w, num_anchors_per_layer, 5 + num_classes]
        y_true needs to be in its initial shape so that its values can easily be passed into a 
        placeholder variable in the feed dictionary. 
    y_true_boxes : tensor
        A tensor containing ground truth box data, which will eventually be used to perform the IOU 
        calculations with the yolo layers' predicted boxes. This tensor greatly reduced the computational
        cost of the IOU computation compared with using the y_true tensor instead.
        A sample y_true_boxes tensor would be of the shape:
        [batch_size, num_anchors_per_layer * max_num_true_boxes_per_image, 4]
        y_true_boxes needs to be in its initial shape so that its values can easily be passed into a 
        placeholder variable in the feed dictionary. The relevant data per yolo layer is extracted and then
        properly fed into the loss_per_scale function.
    ignore_threshold : float
        A number between zero and one.
    anchors : list
        A list of anchors with format:
        [[anchor1_width, anchor1_height], [anchor2_width, anchor2_height], [anchor3_width, anchor3_height], ...]
        The anchors are determined by running a k-means algorithm on the box coordinate training data, and are 
        sorted from smallest to largest. The largest width and height anchors will be used with the sparsest 
        yolo layer grid (responsible for detecting large objects), while the smallest anchors will be used 
        with the finest yolo layer grid (responsible for detecting small objects). 
    num_classes : int
        The number of classes found in the training dataset.
    h : int
        The height of the input image.
    w : int
        The width of the input image.
    batch_size : int
        The number of images per training batch. Used to help with reshaping tensors.

    Returns
    -------
    loss : tensor
        The total loss associated with one forward pass through the yolov3 model.
    """

    def loss_per_scale(name, yolo_layer_outputs, conv_layer_outputs, y_true, y_true_boxes, ignore_thresh, anchors, num_classes=3, h=100, w=100, batch_size=32):
        """
        Calculates and returns the loss associated with a particular layer scale.

        Parameters
        ----------
        name : string
            The name to be used to group the operations of the loss_per_scale function in Tensorboard.
        yolo_layer_outputs : tensor
            The outputs of a yolo layer, which are the fully scaled predicted boxes in the form of 
            'center_x, center_y, width, height'. If an input image is of the shape (416, 416), then
            a sample predicted box may have coordinates of (100,80,40,53).
            A sample yolo layer output will be a tensor of the shape:
            [batch_size, yolo_layer_grid_h, yolo_layer_grid_w, num_anchors_per_layer, 5 + num_classes] 
            The '5' represents the x_coord, y_coord, width_value, height_value, and obj_confidence_score.
            The yolo layer output is needed to calculate the IOU between the predicted boxes and the 
            true boxes.
        conv_layer_outputs : tensor
            The outputs of a convolutional layer, right before they are fed into a yolo layer.
            The convolutional layer will be a tensor of shape:
            [batch_size, yolo_layer_grid_h, yolo_layer_grid_w, num_anchors_per_layer * (5 + num_classes)]
            The convolutional layer outputs are raw predictions which have not been passed
            through the logarithmic nor exponential functions necessary to predict 
            fully scaled bounding boxes. The outputs of the convolutional layer are needed to calculate 
            the coordinate loss, the object confidence loss, and the class loss of each detector.
        y_true : tensor
            The ground truth tensor which contains the theoretical ideal output of a 
            corresponding yolo layer. 
            A sample y_true tensor will be of shape:
            [batch_size, yolo_layer_grid_h * yolo_layer_grid_w * num_anchors_per_layer, 5 + num_classes]
            which will then be reshaped into the shape of:
            [batch_size, yolo_layer_grid_h, yolo_layer_grid_w, num_anchors_per_layer, 5 + num_classes]
            The '5' represents the x_coord, y_coord, width_value, height_value, and obj_confidence_score.
            The coordinates of the boxes in y_true are stored in terms of 'center_x', 'center_y',
            'width', and 'height', and their values are percentages of the original input image size.
            In the case of y_true, the value at the obj_confidence_score index will always be 1 at the
            location of objects. y_true is needed to calculate the coordinate loss, the 
            object confidence loss, and the class loss. 
        y_true_boxes : tensor
            The ground truth boxes per image.
            A sample y_true_boxes tensor would be of shape:
            [batch_size, max_num_true_boxes_per_image, 4] 
            The y_true_boxes are needed to compute the IOU between the predicted boxes from the yolo output
            layer ground truth boxes. 
            y_true_boxes is used instead of y_true because of the significantly smaller computational cost.
            For each box predicted by a detector in the yolo layer, the IOU only has to be calculated between
            the single predicted box and the y_true_boxes. This means that only max_num_true_boxes_per_image 
            IOU calculations need to be made per predicted box. On the other hand if y_true would be used, 
            it would mean that yolo_layer_grid_h * yolo_layer_grid_w * num_anchors_per_layer IOU calculations
            would need to be made per predicted box, which is extremely expensive. 
        ignore_thresh : float
            The threshold which determines how high the IOU between a predicted box and a ground
            truth box needs to be in order for the predicted box to be ignored in the object confidence loss.
            If for example the threshold is set to 0.5, then only predicted boxes that score an IOU greater
            than 0.5 with their ground truth boxes will be ignored.
        anchors : list
            A sublist of the anchors list, of length num_anchors/num_layers. 
            The formatting of the sublist is as follows: 
            [[anchor1_width, anchor1_height], [anchor2_width, anchor2_height], [anchor3_width, anchor3_height]]
            The anchors work across all of the layer's detectors, acting as 'guides' for the 
            bounding box predictions. The anchors are needed to transform the y_true tensor in a way to make
            it comparable with the conv_layer_outputs.  
        num_classes : int
            The number of classes found in the training dataset.
        h : int
            The height of the input image.
        w : int
            The width of the input image.
        batch_size : int
            The number of images per training batch. Used to help with reshaping tensors.

        Returns
        -------
        loss : tensor
            The loss associated with the particular scale of a layer.
        """

        def iou(name, prct_yolo_outputs, y_true_boxes, shape, batch_size):
            """
            Calculates the IOU (Intersection over Union) between the predicted boxes (prct_yolo_outputs)
            and the true boxes (y_true_boxes). Every predicted box at each detector location will
            have an IOU calculated with all of the true boxes per image. A detector that predicts a box with a high 
            IOU is doing a good job with its prediction, so we don't want to penalize this detector. A mask is 
            created for the detectors whos max IOU value is above a certain threshold, which is then applied onto 
            the term in the loss function that penalizes detectors for wrongly detecting boxes. This mask prevents 
            the loss from increasing from detectors which have high IOUs, regardless of whether or not they should be
            predicting an object.

            Parameters
            ----------
            name : string
                The name that will be used for the IOU function in the TensorBoard graph
            prct_yolo_outputs : tensor
                The outputs of a yolo layer, which are the fully scaled predicted boxes
                in the form of 'center_x, center_y, width, height', divided by the original
                input width and height to turn them into percentages.
                A sample yolo layer output will be a tensor of the shape:
                [batch_size, yolo_layer_grid_h, yolo_layer_grid_w, num_anchors_per_layer, 4] 
                The '4' represents the x_coord, y_coord, width_value, and height_value.
            y_true_boxes : tensor
                The true boxes coordinates for each image, stored in the form of 
                'center_x, center_y, width, height', as percentages of the original input.
                y_true_boxes is a tensor of the shape:
                [batch_size, max_num_true_boxes_per_image, 4]
                The '4' represents the x_coord, y_coord, width_value, and height_value.
                y_true_boxes is generated by the 'create_y_true' function.
            shape : list
                The shape of the yolo layer outputs:
                [batch_size, yolo_layer_grid_h, yolo_layer_grid_w, num_anchors_per_layer, (5+num_classes)]
                The '5' represents the x_coord, y_coord, width_value, height_value, and class_number.
                Used to help with reshaping tensors.
            batch_size: int
                The number of images per training batch.
                Used to help with reshaping tensors.

            Returns
            -------
            max_ious : tensor
                A tensor containing the max IOU for each detector. The tensor will have the shape:
                [batch_size, num_detectors_per_layer]
                num_detectors_per_layer = yolo_layer_grid_h x yolo_layer_grid_w x num_anchors_per_layer
                The values at each index are the highest IOU score between each detector's predicted 
                box and the true boxes of a particular image.
            """

            with tf.variable_scope(name):

                prct_yolo_outputs = tf.reshape(prct_yolo_outputs,shape=(-1,shape[1]*shape[2]*shape[3],4))
                prct_yolo_outputs = tf.expand_dims(prct_yolo_outputs, -2)
                yolo_outputs_xy_mins = prct_yolo_outputs[:,:,:,0:2] - prct_yolo_outputs[:,:,:,2:4]/2.0
                yolo_outputs_xy_maxes = prct_yolo_outputs[:,:,:,0:2] + prct_yolo_outputs[:,:,:,2:4]/2.0

                y_true_boxes = y_true_boxes[:,:,0:4]
                y_true_boxes = tf.expand_dims(tf.reshape(y_true_boxes, [batch_size,-1,4]),1)
                y_true_xy_mins = y_true_boxes[...,0:2] - y_true_boxes[:,:,:,2:4]/2.0
                y_true_xy_maxes = y_true_boxes[...,0:2] + y_true_boxes[:,:,:,2:4]/2.0

                intersecting_mins = tf.maximum(yolo_outputs_xy_mins, y_true_xy_mins)
                intersecting_maxes = tf.minimum(yolo_outputs_xy_maxes, y_true_xy_maxes)

                intersect_hw = tf.maximum(intersecting_maxes - intersecting_mins, 0.0)
                intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]

                yolo_outputs_area = prct_yolo_outputs[..., 2] * prct_yolo_outputs[..., 3]
                y_true_area = y_true_boxes[...,2] * y_true_boxes[...,3]
                
                iou = intersect_area / (yolo_outputs_area + y_true_area - intersect_area)
                max_ious = tf.reduce_max(iou,axis=-1)

            return max_ious 

        with tf.variable_scope(name):

            num_anchors = len(anchors)
            shape = yolo_layer_outputs.get_shape().as_list()
        
            with tf.variable_scope('trnsfrm_yolo_layer'):
                wh = tf.cast(tf.constant([w,h,w,h]),tf.float32)
                percentage_yolo_outputs = yolo_layer_outputs[...,0:4] / wh
            
            with tf.variable_scope('trnsfrm_conv_layer'):
                conv_layer_outputs = tf.reshape(conv_layer_outputs, [-1,shape[1],shape[2],shape[3],shape[4]])

            with tf.variable_scope('trnsfrm_y_true'):
                y_true = tf.reshape(y_true, [-1,shape[1], shape[2], shape[3], shape[4]])
                percent_x, percent_y, percent_w, percent_h, obj_mask, classes = tf.split(y_true, [1, 1, 1, 1, 1, num_classes], axis=-1)
            
            with tf.variable_scope('pred_coords_loss'):

                with tf.variable_scope('cnvrt_y_true_coords'):
                    
                    with tf.variable_scope('cnvrt_xy'):

                        clustroid_x = tf.tile(tf.reshape(tf.range(shape[2], dtype=tf.float32), [1, -1, 1, 1]), [shape[2], 1, 1, 1])
                        clustroid_y = tf.tile(tf.reshape(tf.range(shape[1], dtype=tf.float32), [-1, 1, 1, 1]), [1, shape[1], 1, 1])
                        converted_x_true = percent_x * shape[2] - clustroid_x
                        converted_y_true = percent_y * shape[1] - clustroid_y
                    
                    with tf.variable_scope('cnvrt_wh'):

                        anchors = tf.constant(anchors,dtype=tf.float32)
                        anchors_w = tf.reshape(anchors[:,0], [1,1,1,num_anchors,1])
                        anchors_h = tf.reshape(anchors[:,1], [1,1,1,num_anchors,1])

                        converted_w_true = tf.log((percent_w/anchors_w)*w)
                        converted_h_true = tf.log((percent_h/anchors_h)*h)

                with tf.variable_scope('concat_cnvrtd_y_true'):
                    converted_y_true = tf.concat([converted_x_true,converted_y_true,converted_w_true,converted_h_true], axis=-1)
                
                with tf.variable_scope('replace_inf_with_zeros'):
                    converted_y_true = tf.where(tf.is_inf(converted_y_true),tf.zeros_like(converted_y_true),converted_y_true)
                
                with tf.variable_scope('box_loss_scale'):
                    box_loss_scale = 2 - y_true[...,2:3]*y_true[...,3:4]

                with tf.variable_scope('xy_coord_loss'):
                    xy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=converted_y_true[...,0:2], logits=conv_layer_outputs[...,0:2]) * obj_mask * box_loss_scale
                    xy_loss = tf.reduce_sum(xy_loss)
                
                with tf.variable_scope('wh_coord_loss'):
                    wh_loss = tf.square(converted_y_true[...,2:4]-conv_layer_outputs[...,2:4]) * 0.5 * obj_mask * box_loss_scale
                    wh_loss = tf.reduce_sum(wh_loss)
                
                with tf.variable_scope('compile_coord_loss'):
                    coord_loss = xy_loss + wh_loss

            with tf.variable_scope('pred_obj_loss'):
                
                with tf.variable_scope('create_ignore_mask'):
                    box_iou = iou('iou_yolo_bxs_y_true_bxs', percentage_yolo_outputs, y_true_boxes, shape, batch_size)
                    ignore_mask = tf.cast(tf.less(box_iou,ignore_thresh*tf.ones_like(box_iou)),tf.float32)
                    ignore_mask = tf.reshape(ignore_mask,[-1,shape[1],shape[2],num_anchors])
                    ignore_mask = tf.expand_dims(ignore_mask, -1)

                with tf.variable_scope('no_obj_loss'):
                    no_obj_loss = (1-obj_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=conv_layer_outputs[...,4:5]) * ignore_mask
                    no_obj_loss = tf.reduce_sum(no_obj_loss)

                with tf.variable_scope('obj_loss'):
                    obj_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=conv_layer_outputs[...,4:5])
                    obj_loss = tf.reduce_sum(obj_loss)
                
                with tf.variable_scope('compile_pred_obj_loss'):
                    confidence_loss = obj_loss + no_obj_loss

            with tf.variable_scope('pred_class_loss'):

                class_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[...,5:], logits=conv_layer_outputs[...,5:])
                class_loss = tf.reduce_sum(class_loss)

            with tf.variable_scope('compile_losses'):

                loss = coord_loss + confidence_loss + class_loss

        return loss

    num_anchors_per_detector = len(anchors)//3
    max_num_boxes_per_image = y_true_boxes.shape[1]//3

    num_large_detectors = int((h/32)*(w/32)*num_anchors_per_detector)
    num_medium_detectors = int((h/16)*(w/16)*num_anchors_per_detector)

    # extracting data from y_true and y_true_boxes

    with tf.variable_scope('y_true_0'):
        large_obj_y_true = y_true[:,:num_large_detectors,:]
    with tf.variable_scope('y_true_1'):
        medium_obj_y_true = y_true[:,num_large_detectors:num_large_detectors+num_medium_detectors,:]
    with tf.variable_scope('y_true_2'):    
        small_obj_y_true = y_true[:,num_large_detectors+num_medium_detectors:,:]

    with tf.variable_scope('box_data_0'):
        large_obj_y_true_boxes = y_true_boxes[:,:max_num_boxes_per_image,:]
    with tf.variable_scope('box_data_1'):  
        medium_obj_y_true_boxes = y_true_boxes[:,max_num_boxes_per_image:max_num_boxes_per_image*2,:]
    with tf.variable_scope('box_data_2'):
        small_obj_y_true_boxes = y_true_boxes[:,max_num_boxes_per_image*2:,:]

    # extracting data from yolo_outputs

    yolo_layer_outputs = yolo_outputs[:3]
    conv_layer_outputs = yolo_outputs[3:]

    large_obj_box_detections = yolo_layer_outputs[0]
    medium_obj_box_detections = yolo_layer_outputs[1]
    small_obj_box_detections  = yolo_layer_outputs[2]
    
    large_obj_raw_detections = conv_layer_outputs[0]
    medium_obj_raw_detections = conv_layer_outputs[1]
    small_obj_raw_detections = conv_layer_outputs[2]    
        
    # passing data through loss function

    large_obj_loss = loss_per_scale(
        'loss_0',
        large_obj_box_detections,
        large_obj_raw_detections,
        large_obj_y_true,
        large_obj_y_true_boxes,
        ignore_threshold, anchors[num_anchors_per_detector*2:], num_classes, h, w, batch_size)
    medium_obj_loss = loss_per_scale(
        'loss_1',
        medium_obj_box_detections,
        medium_obj_raw_detections,
        medium_obj_y_true,
        medium_obj_y_true_boxes,
        ignore_threshold, anchors[num_anchors_per_detector:num_anchors_per_detector*2], num_classes, h, w, batch_size)
    small_obj_loss = loss_per_scale(
        'loss_2',
        small_obj_box_detections,
        small_obj_raw_detections,
        small_obj_y_true,
        small_obj_y_true_boxes,
        ignore_threshold, anchors[:num_anchors_per_detector], num_classes, h, w, batch_size)

    with tf.variable_scope('total_loss'):
        loss = (large_obj_loss + medium_obj_loss + small_obj_loss)/batch_size

    return loss