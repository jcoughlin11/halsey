"""
Title: rnn1.py
Purpose: Contains the rnn1 network architecture.
Notes:
"""


#============================================
#              build_rnn1_net
#============================================
def build_rnn1_net(inputShape, nActions):
        """
        Architecture for Recurrent Deep Q-Learning, from Hausknecht et
        al. 2017.

        See: https://www.tensorflow.org/beta/guide/keras/rnn

        and: https://tinyurl.com/y5g4f9hn (input and output shape for
            LSTMs)

        and: https://tinyurl.com/y28llddb (CNN LSTM,
            https://arxiv.org/abs/1411.4389)

        and: https://tinyurl.com/y28tkxak (how to use TimeDistributed)

        The input shape to an LSTM needs to be:
        (batchSize, nTimeSteps, nFeatures). A CNN, though doesn't care
        about time, just space, so it's shape is:
        (batchSize, nFeaturesDim1, ...)

        The way that the convolutional layers are applied to every
        time-step is to use the TimeDistributed layer. We pass the
        normal spatial shape to the CNN layer and then pass nTimeSteps
        to the wrapping TimeDistributed layer. All of this is then
        passed normally to the LSTM layer(s).

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Initialize empty model
        model = tf.keras.models.Sequential()
        # First convolutional layer. Here, inputShape should be
        # (nTimeSteps, nFeaturesDim1, ..., nFeaturesDimN). The batch
        # size is handled when calling fit(). So, the true shape of the
        # input passed to fit is (batchSize, nTimeSteps,
        # nFeaturesDim1,...)
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[8,8],
                    strides=[4,4],
                    activation='relu',
                    name='conv1'
                ),
                input_shape=inputShape
            )
        )
        # Second convolutional layer
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[4,4],
                    strides=[2,2],
                    activation='relu',
                    name='conv2'
                )
            )
        )
        # Third convolutional layer
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu',
                    name='conv3'
                )
            )
        )
        # Flatten layer
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
        # LSTM
        model.add(
            tf.keras.layers.LSTM(
                units=512,
                name='lstm1'
            )
        )
        # FC layer
        model.add(
            tf.keras.layers.Dense(
                units=nActions,
                activation='linear',
                name='output'
            )
        )
        return model
