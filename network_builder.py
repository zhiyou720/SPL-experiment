import tensorflow as tf
from network_architectures import VGGClassifier, FCCLayerClassifier


class ClassifierNetworkGraph:
    def __init__(self, input_x, target_placeholder, dropout_rate,
                 batch_size=100, n_classes=100, is_training=True, augment_rotate_flag=True,
                 tensorboard_use=False, use_batch_normalization=False, strided_dim_reduction=True,
                 network_name='VGG_classifier',SPL='None',pace_param=1.1):

        """
        Initializes a Classifier Network Graph that can build models, train, compute losses and save summary statistics
        and images
        :param input_x: A placeholder that will feed the input images, usually of size [batch_size, height, width,
        channels]
        :param target_placeholder: A target placeholder of size [batch_size,]. The classes should be in index form
               i.e. not one hot encoding, that will be done automatically by tf
        :param dropout_rate: A placeholder of size [None] that holds a single float that defines the amount of dropout
               to apply to the network. i.e. for 0.1 drop 0.1 of neurons
        :param batch_size: The batch size
        :param num_channels: Number of channels
        :param n_classes: Number of classes we will be classifying
        :param is_training: A placeholder that will indicate whether we are training or not
        :param augment_rotate_flag: A placeholder indicating whether to apply rotations augmentations to our input data
        :param tensorboard_use: Whether to use tensorboard in this experiment
        :param use_batch_normalization: Whether to use batch normalization between layers
        :param strided_dim_reduction: Whether to use strided dim reduction instead of max pooling
        :param SPL: the type of SPL (None, SPL, SPBL, SPCL_LinSW, SPCL_LogSW, SPCL_MW, SPLD)
        """
        self.batch_size = batch_size
        if network_name == "VGG_classifier":
            self.c = VGGClassifier(self.batch_size, name="classifier_neural_network",
                                   batch_norm_use=use_batch_normalization, num_classes=n_classes,
                                   layer_stage_sizes=[64, 128, 256], strided_dim_reduction=strided_dim_reduction)
        elif network_name == "FCCClassifier":
            self.c = FCCLayerClassifier(self.batch_size, name="classifier_neural_network",
                                   batch_norm_use=use_batch_normalization,  num_classes=n_classes,
                                   layer_stage_sizes=[64, 128, 256], strided_dim_reduction=strided_dim_reduction)

        self.input_x = input_x
        self.dropout_rate = dropout_rate
        self.targets = target_placeholder

        self.training_phase = is_training
        self.n_classes = n_classes
        self.iterations_trained = 0

        self.augment_rotate = augment_rotate_flag
        self.is_tensorboard = tensorboard_use
        self.strided_dim_reduction = strided_dim_reduction
        self.use_batch_normalization = use_batch_normalization
        self.SPL = SPL
        if self.SPL != "None":
            self.pace_param = tf.constant(pace_param)
            if self.SPL == "SPL" or self.SPL == "SPCL_LinSW":
                self.lp = tf.Variable(3.5)          #  initialize learning_pace lp >0
            elif self.SPL == "SPCL_LogSW":
                self.lp = tf.Variable(0.1)        # 0<=lp<=1
            elif self.SPL == "SPCL_MW":
                self.lp_1 =  tf.Variable(2.2)           # lp_1 > lp_2 > 0
                self.lp_2 =  tf.Variable(1.1)

    def loss(self):
        """build models, calculates losses, saves summary statistcs and images.
        Returns:
            dict of losses.
        """
        with tf.name_scope("losses"):
            image_inputs = self.data_augment_batch(self.input_x)  # conditionally apply augmentaions
            true_outputs = self.targets
            # produce predictions and get layer features to save for visual inspection
            preds, layer_features = self.c(image_input=image_inputs, training=self.training_phase,
                                           dropout_rate=self.dropout_rate)
            # compute loss and accuracy
            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(true_outputs, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            crossentropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_outputs, logits=preds))
            if self.SPL == "None":
                losses = crossentropy_loss
            elif self.SPL == "SPL":
                v = tf.cast(tf.less(crossentropy_loss, self.lp,name=None),dtype=tf.float32)
                losses = tf.subtract(tf.reduce_sum(v * crossentropy_loss), self.lp * tf.reduce_sum(v),name=None)
                self.lp *= self.pace_param
            elif self.SPL == 'SPCL_LinSW':
                mask = tf.cast(tf.less(crossentropy_loss, self.lp,name=None),dtype=tf.float32)
                v = mask * (tf.subtract(1.0,crossentropy_loss/self.lp))
                losses = tf.add(tf.reduce_sum(v * crossentropy_loss) , self.lp * tf.subtract( tf.reduce_sum(v*v)/2.0,tf.reduce_sum(v)))
                self.lp *= self.pace_param
            elif self.SPL == 'SPCL_LogSW':
                mask = tf.cast(tf.less(crossentropy_loss, self.lp,name=None),dtype=tf.float32)
                self.lp = tf.cond(tf.greater_equal(self.lp,1),true_fn=lambda:self.lp/self.pace_param,false_fn=lambda:self.lp)
                v = mask * tf.log(tf.subtract(tf.add(crossentropy_loss,1.0),self.lp)) / tf.log(tf.subtract(1.0,self.lp))
                sigma = tf.subtract(1.0,self.lp)
                losses = tf.add(tf.reduce_sum(v * crossentropy_loss),tf.subtract(tf.reduce_sum(sigma*v), tf.pow(sigma,v) / tf.log(sigma))) 
                self.lp *= self.pace_param
            elif self.SPL == "SPCL_MW":
                sigma = (self.lp_1 * self.lp_2) / (tf.subtract(self.lp_1, self.lp_2))
                mask_1 = tf.cast(tf.less_equal(crossentropy_loss, (self.lp_1 * sigma / tf.add(sigma, self.lp_1))),dtype=tf.float32)
                mask_2 = tf.cast(tf.greater(crossentropy_loss, (self.lp_1 * sigma / tf.add(sigma, self.lp_1))),dtype=tf.float32) * tf.cast(tf.less(crossentropy_loss, self.lp_1),tf.float32)
                mask_2 = mask_2 * tf.subtract(self.lp_1, crossentropy_loss) * sigma / (self.lp_1 * crossentropy_loss)
                v =tf.add(mask_1,mask_2)
                losses = tf.subtract(tf.reduce_sum(v * crossentropy_loss), sigma * tf.reduce_sum(tf.log(tf.add(v,sigma/self.lp_1))))
                self.lp_1 *= self.pace_param
                self.lp_2 *= self.pace_param
                
            # add loss and accuracy to collections
            tf.add_to_collection('losses',losses)
            tf.add_to_collection('crossentropy_losses', crossentropy_loss)
            tf.add_to_collection('accuracy', accuracy)

            # save summaries for the losses, accuracy and image summaries for input images, augmented images
            # and the layer features
            if len(self.input_x.get_shape().as_list()) == 4:
                self.save_features(name="VGG_features", features=layer_features)
            tf.summary.image('image', [tf.concat(tf.unstack(self.input_x, axis=0), axis=0)])
            tf.summary.image('augmented_image', [tf.concat(tf.unstack(image_inputs, axis=0), axis=0)])
            tf.summary.scalar('crossentropy_losses', crossentropy_loss)
            tf.summary.scalar('accuracy', accuracy)

        return {"crossentropy_losses": tf.add_n(tf.get_collection('crossentropy_losses'),
                                                name='total_classification_loss'),
                "accuracy": tf.add_n(tf.get_collection('accuracy'), name='total_accuracy'),
                "losses": tf.add_n(tf.get_collection('losses'),name='total_losses')}

    def save_features(self, name, features, num_rows_in_grid=4):
        """
        Saves layer features in a grid to be used in tensorboard
        :param name: Features name
        :param features: A list of feature tensors
        """
        for i in range(len(features)):
            shape_in = features[i].get_shape().as_list()
            channels = shape_in[3]
            y_channels = num_rows_in_grid
            x_channels = int(channels / y_channels)

            activations_features = tf.reshape(features[i], shape=(shape_in[0], shape_in[1], shape_in[2],
                                                                  y_channels, x_channels))

            activations_features = tf.unstack(activations_features, axis=4)
            activations_features = tf.concat(activations_features, axis=2)
            activations_features = tf.unstack(activations_features, axis=3)
            activations_features = tf.concat(activations_features, axis=1)
            activations_features = tf.expand_dims(activations_features, axis=3)
            tf.summary.image('{}_{}'.format(name, i), activations_features)

    def rotate_image(self, image):
        """
        Rotates a single image
        :param image: An image to rotate
        :return: A rotated or a non rotated image depending on the result of the flip
        """
        no_rotation_flip = tf.unstack(
            tf.random_uniform([1], minval=1, maxval=100, dtype=tf.int32, seed=None,
                              name=None))  # get a random number between 1 and 100
        flip_boolean = tf.less_equal(no_rotation_flip[0], 50)
        # if that number is less than or equal to 50 then set to true
        random_variable = tf.unstack(tf.random_uniform([1], minval=1, maxval=3, dtype=tf.int32, seed=None, name=None))
        # get a random variable between 1 and 3 for how many degrees the rotation will be i.e. k=1 means 1*90,
        # k=2 2*90 etc.
        image = tf.cond(flip_boolean, lambda: tf.image.rot90(image, k=random_variable[0]),
                        lambda: image)  # if flip_boolean is true the rotate if not then do not rotate
        return image

    def rotate_batch(self, batch_images):
        """
        Rotate a batch of images
        :param batch_images: A batch of images
        :return: A rotated batch of images (some images will not be rotated if their rotation flip ends up False)
        """
        shapes = map(int, list(batch_images.get_shape()))
        if len(list(batch_images.get_shape())) < 4:
            return batch_images
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked = tf.unstack(batch_images)
            new_images = []
            for image in batch_images_unpacked:
                new_images.append(self.rotate_image(image))
            new_images = tf.stack(new_images)
            new_images = tf.reshape(new_images, (batch_size, x, y, c))
            return new_images

    def data_augment_batch(self, batch_images):
        """
        Augments data with a variety of augmentations, in the current state only does rotations.
        :param batch_images: A batch of images to augment
        :return: Augmented data
        """
        batch_images = tf.cond(self.augment_rotate, lambda: self.rotate_batch(batch_images), lambda: batch_images)
        return batch_images

    def train(self, losses, learning_rate=1e-3, beta1=0.9):
        """
        Args:
            losses dict.
        Returns:
            train op.
        """
        c_opt = tf.train.AdamOptimizer(beta1=beta1, learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            c_error_opt_op = c_opt.minimize(losses["losses"], var_list=self.c.variables,
                                            colocate_gradients_with_ops=True)

        return c_error_opt_op

    def init_train(self):
        """
        Builds graph ops and returns them
        :return: Summary, losses and training ops
        """
        losses_ops = self.loss()
        c_error_opt_op = self.train(losses_ops)
        summary_op = tf.summary.merge_all()
        return summary_op, losses_ops, c_error_opt_op
