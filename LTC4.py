import tensorflow as tf
import numpy as np
from enum import Enum
from tensorflow.compat.v1.nn import rnn_cell as rnn_cell

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

# The main LTCCell class that inherits from tf.keras.layers.AbstractRNNCell
class LTCCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, num_units, input_mapping=MappingType.Affine, solver=ODESolver.SemiImplicit, ode_solver_unfolds=6, activation=tf.nn.tanh, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units
        self._ode_solver_unfolds = ode_solver_unfolds
        self._solver = solver
        self._input_mapping = input_mapping
        self._activation = activation

    # Returns the state size of the cell
    @property
    def state_size(self):
        return self._num_units

    # Returns the output size of the cell
    @property
    def output_size(self):
        return self._num_units

    # Builds the weights for the cell
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self._num_units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self._num_units, self._num_units), initializer='glorot_uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self._num_units,), initializer='zeros', name='bias')
        self.built = True

    # Implements the forward pass of the cell
    def call(self, inputs, states):
        prev_output = states[0]
        net_input = tf.matmul(inputs, self.kernel)
        net_input += tf.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias
        output = self._activation(net_input)  # Use the activation function

        return output, [output]

    # Activation function for the cell (to be implemented)
    def activation(self, net_input):
        pass

    # Returns the configuration of the cell
    def get_config(self):
        config = super().get_config()
        config.update({"_num_units": self._num_units})
        return config

# Define necessary classes from ctrnn_model.py
# (CTRNN, NODE, and CTGRU class definitions, which are not directly related to LTCCell)
class CTRNN(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, global_feedback=False, activation=tf.nn.tanh, cell_clip=None, **kwargs):
        self.units = units
        self.global_feedback = global_feedback
        self.activation = activation
        self.cell_clip = cell_clip
        super(CTRNN, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        net_input = tf.matmul(inputs, self.kernel)
        net_input += tf.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias
        output = self.activation(net_input)

        if self.cell_clip is not None:
            output = tf.clip_by_value(output, -self.cell_clip, self.cell_clip)

        return output, [output]

class NODE(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, cell_clip=None, **kwargs):
        self.units = units
        self.cell_clip = cell_clip
        super(NODE, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        net_input = tf.matmul(inputs, self.kernel)
        net_input += tf.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias
        output = tf.nn.tanh(net_input)

        if self.cell_clip is not None:
            output = tf.clip_by_value(output, -self.cell_clip, self.cell_clip)

        return output, [output]


class CTGRU(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, cell_clip=None, **kwargs):
        self.units = units
        self.cell_clip = cell_clip
        super(CTGRU, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], 2 * self.units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, 2 * self.units), initializer='glorot_uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(2 * self.units,), initializer='zeros', name='bias')
        self.kernel_c = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel_c')
        self.recurrent_kernel_c = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='recurrent_kernel_c')
        self.bias_c = self.add_weight(shape=(self.units,), initializer='zeros', name='bias_c')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        zr = tf.matmul(inputs, self.kernel)
        zr += tf.matmul(prev_output, self.recurrent_kernel)
        zr += self.bias
        z, r = tf.split(zr, 2, axis=-1)

        z = tf.sigmoid(z)
        r = tf.sigmoid(r)

        c = tf.matmul(inputs, self.kernel_c)
        c += r * tf.matmul(prev_output, self.recurrent_kernel_c)
        c += self.bias_c
        c = tf.nn.tanh(c)

        output = (1 - z) * prev_output + z * c

        if self.cell_clip is not None:
            output = tf.clip_by_value(output, -self.cell_clip, self.cell_clip)

        return output, [output]

