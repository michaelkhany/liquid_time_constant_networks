import tensorflow as tf
import numpy as np
from LTC4 import LTCCell
from LTC4 import ODESolver

# Define the test function
def test_ltc_ode_solvers():
    num_units = 10
    batch_size = 5
    time_steps = 3
    input_dim = 4

    # Create some dummy input data
    inputs = tf.random.normal([batch_size, time_steps, input_dim])

    # Test with each ODE solver
    for solver in [ODESolver.SemiImplicit, ODESolver.Explicit, ODESolver.RungeKutta]:
        print(f"\nTesting with ODE Solver: {solver.name}")
        ltc_cell = LTCCell(num_units=num_units, solver=solver)
        rnn_layer = tf.keras.layers.RNN(ltc_cell, return_sequences=True)
        
        # Apply the RNN layer
        output = rnn_layer(inputs)
        print(f"Output (shape {output.shape}):\n{output.numpy()}")

        # Calculate and print some additional metrics
        output_mean = tf.reduce_mean(output).numpy()
        output_std = tf.math.reduce_std(output).numpy()
        print(f"Output Mean: {output_mean}")
        print(f"Output Std Dev: {output_std}")

# Run the test function
test_ltc_ode_solvers()
