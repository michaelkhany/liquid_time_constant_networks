import tensorflow as tf
import numpy as np
from enum import Enum

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

# The main LTCCell class that inherits from tf.keras.layers.Layer
# LTC4_0231105
class LTCCell(tf.keras.layers.Layer):
    def __init__(self, num_units, input_mapping=MappingType.Affine, solver=ODESolver.SemiImplicit, ode_solver_unfolds=6, activation=tf.nn.tanh, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units
        self._ode_solver_unfolds = ode_solver_unfolds
        self._solver = solver
        self._input_mapping = input_mapping
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self._num_units), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self._num_units, self._num_units), initializer='glorot_uniform', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self._num_units,), initializer='zeros', name='bias')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        net_input = tf.matmul(inputs, self.kernel)
        net_input += tf.matmul(prev_output, self.recurrent_kernel)
        net_input += self.bias
        
        # # Logging intermediate values
        # tf.print("Inputs:", inputs)
        # tf.print("Previous State:", prev_output)
        # tf.print("Net Input:", net_input)
        
        # Use the selected ODE solver to calculate the output
        if self._solver == ODESolver.SemiImplicit:
            output = self._semi_implicit_solver(prev_output, net_input)
        elif self._solver == ODESolver.Explicit:
            output = self._explicit_solver(prev_output, net_input)
        elif self._solver == ODESolver.RungeKutta:
            output = self._runge_kutta_solver(prev_output, net_input)
        else:
            raise ValueError("Unsupported ODE Solver type.")
        
        # # Logging output
        # tf.print("Output:", output)
        
        # # Calculate and log some metrics
        # tf.print("Output Mean:", tf.reduce_mean(output))
        # tf.print("Output Std Dev:", tf.math.reduce_std(output))

        return output, [output]

    def _semi_implicit_solver(self, prev_output, net_input):
        output = prev_output + self._ode_solver_unfolds * (self._activation(net_input) - prev_output)
        return output

    def _explicit_solver(self, prev_output, net_input):
        output = prev_output + self._ode_solver_unfolds * self._activation(net_input)
        return output

    def _runge_kutta_solver(self, prev_output, net_input):
        dt = 1.0 / self._ode_solver_unfolds
        k1 = self._activation(net_input)
        k2 = self._activation(net_input + 0.5 * dt * k1)
        k3 = self._activation(net_input + 0.5 * dt * k2)
        k4 = self._activation(net_input + dt * k3)
        output = prev_output + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"_num_units": self._num_units, "_solver": self._solver})
        return config

# Defined necessary classes from ctrnn_model.py
# (CTRNN, NODE, and CTGRU class definitions, which are not directly related to LTCCell)
class CTRNN(tf.keras.layers.Layer):
    def __init__(self, units, global_feedback=False, activation=tf.nn.tanh, cell_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.global_feedback = global_feedback
        self.activation = activation
        self.cell_clip = cell_clip

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

class NODE(tf.keras.layers.Layer):
    def __init__(self, units, cell_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.cell_clip = cell_clip

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

class CTGRU(tf.keras.layers.Layer):
    def __init__(self, units, cell_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.cell_clip = cell_clip

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


# # Example usage with LTCCell
# num_units = 10
# ltc_cell = LTCCell(num_units)

# rnn_layer = tf.keras.layers.RNN(ltc_cell)

# # Create some dummy input data
# batch_size = 5
# time_steps = 3
# input_dim = 4
# inputs = tf.random.normal([batch_size, time_steps, input_dim])

# # Apply the RNN layer
# output = rnn_layer(inputs)