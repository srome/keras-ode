from keras.layers import Conv2D
import tensorflow as tf
import keras.backend as K
from keras import regularizers
from keras.layers import Concatenate, Lambda, InputSpec
from keras.engine.topology import Layer

class ChannelZeroPadding(Layer):
    """ Multiple the number of channels in tensor by the "pads" variable """

    def __init__(self, padding=4, **kwargs):
        super(ChannelZeroPadding, self).__init__(**kwargs)
        self.padding=padding

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.built=True

    def call(self, x):
        return K.concatenate([x] + [K.zeros_like(x) for k in range(self.padding-1)], axis=-1)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1]*self.padding
        return tuple(output_shape)

class HamiltonianConv2D(Conv2D):
    
    def __init__(self, unroll_length=3, h=.01, **kwargs):
        
        # divide filters by 2 due to channel splitting
        super(HamiltonianConv2D, self).__init__(**kwargs)
        self.unroll_length = unroll_length
        self.h=h
    
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim/2, self.filters/2) # channel splitting

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters/2,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias2 = self.add_weight(shape=(self.filters/2,),
                            initializer=self.bias_initializer,
                            name='bias2',
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True  

    def call(self, inputs):
        h=self.h
        tmp, otmp = tf.split(inputs, num_or_size_splits=2, axis=-1)
                
        for k in range(self.unroll_length):
            y_f_tmp = K.conv2d(
            otmp,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
                    
            if self.use_bias:
                    y_f_tmp = K.bias_add(
                        y_f_tmp,
                        self.bias,
                        data_format=self.data_format)

            x_f_tmp = -K.conv2d_transpose(
                x=tmp,
                kernel=self.kernel,
                strides=self.strides,
                output_shape = K.shape(tmp),
                padding=self.padding,
                data_format=self.data_format)
            
        
            if self.use_bias:
                    x_f_tmp = K.bias_add(
                        x_f_tmp,
                        self.bias2,
                        data_format=self.data_format)
                    
                    
            if self.activation is not None:
                tmp = tmp + h*self.activation(y_f_tmp)     
                otmp = otmp + h*self.activation(x_f_tmp)  

        out = K.concatenate([tmp,otmp], axis=-1)
        return out

