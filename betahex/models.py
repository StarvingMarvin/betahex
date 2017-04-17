import functools
import tensorflow as tf
from tensorflow.python.layers.base import _to_list, _add_elements_to_collection
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops


def conv_layer(x, filters, size, activation, name=None):
    # conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    # conv_with_b = tf.nn.bias_add(conv, b)
    # conv_out = tf.nn.relu(conv_with_b, name=name)
    conv = tf.layers.conv2d(
        x, filters, size, activation=activation, padding='same',
        name=name, kernel_initializer=tf.random_normal_initializer(),
        bias_initializer=tf.random_normal_initializer(),
        kernel_regularizer=None
    )
    # return my_apply(conv, x)
    return conv


def my_apply(layer, inputs, **kwargs):
    """Wraps `call`, applying pre- and post-processing steps.
    
    Arguments:
      inputs: input tensor(s).
      **kwargs: additional keyword arguments to be passed to `self.call`.
    
    Returns:
      Output tensor(s).
    """
    # Define a custom getter to override tf.get_variable when creating layer
    # variables. The current custom getter is nested by the variable scope.
    def variable_getter(getter, name, shape, dtype=None, initializer=None,
                        regularizer=None, trainable=True, **kwargs):
      return layer._add_variable(
          name, shape, initializer=initializer, regularizer=regularizer,
          dtype=layer.dtype, trainable=trainable,
          variable_getter=functools.partial(getter, **kwargs))

    # Build (if necessary) and call the layer, inside a variable scope.
    with vs.variable_scope(layer._scope,
                           reuse=True if layer._built else layer._reuse,
                           custom_getter=variable_getter) as scope:
      with ops.name_scope(scope.original_name_scope):
        if not layer.built:
          input_list = _to_list(inputs)
          input_shapes = [x.get_shape() for x in input_list]
          if len(input_shapes) == 1:
              layer.build(input_shapes[0])
          else:
              layer.build(input_shapes)
              layer._built = True
        outputs = layer.call(inputs, **kwargs)

        # Apply activity regularization.
        # Note that it should be applied every time the layer creates a new
        # output, since it is output-specific.
        if hasattr(layer, 'activity_regularizer') and layer.activity_regularizer:
          output_list = _to_list(outputs)
          for output in output_list:
            with ops.name_scope('ActivityRegularizer'):
              activity_regularization = layer.activity_regularizer(output)
              layer._losses.append(activity_regularization)
            _add_elements_to_collection(
                activity_regularization, ops.GraphKeys.REGULARIZATION_LOSSES)

    # Update global default collections.
    _add_elements_to_collection(layer.updates, ops.GraphKeys.UPDATE_OPS)
    return outputs


def common_model(features, layer_dim_5=None, layer_dims_3=None):

    layer_dims_3 = layer_dims_3 or [96] * 4
    layer_dim_5 = layer_dim_5 or 128

    def model(input):
        tensors = [input[feat] for feat in features.feature_names]
        mangled = tf.cast(tf.concat(tensors, 3), tf.float32)
        prev = conv_layer(mangled, layer_dim_5, 5, tf.nn.elu)

        for dim in layer_dims_3:
            prev = conv_layer(prev, dim, 3, tf.nn.elu)

        return prev

    return model


def make_policy(features, layer_dim_5=None, layer_dims_3=None):

    common_f = common_model(features, layer_dim_5, layer_dims_3)

    def model(input):
        common = common_f(input)
        activation = conv_layer(common, 1, 1, None)

        logits = tf.reshape(
            activation,
            [-1, features.shape[0] * features.shape[1]],
            name="logits"
        )

        return logits

    return model

