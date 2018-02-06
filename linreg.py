# Creator: GENE PATRICK S. RIBLE
# Date: February 2, 2018

"""
NOTE:  Depending on the polynomial coefficients, the program may take several minutes (especially if the coefficients are of
the order 10^2 or lower) up to 1 hour or maybe more to complete. Please be patient and wait until it converges.
----------------------------
Deep Learning Assignment 1

This is a gradient descent implementation for linear regression of arbitrary polynomial function
in python3 and numpy . The program is activated by:

python3 linreg.py 4.4 3.3 2.2 -1.1

where the arguments are the coefficients of the polynomial (i.e. 4.4x^3 + 3.3x^2 + 2.2x - 1.1). To prove its robustness against
noise, uniform distribution [-1,1] noise is added to the output y. Up to 3rd deg polynomial is expected, which means that
the maximum number of arguments when running the program is four.
"""

import sys
import numpy as np

rand_size = 1000
rand_max = 11
learn_rate = 1e-6

max_step = 3
min_step = 0.1

accuracy = 0.01

ave_abs_err = accuracy + 1  # initialize to any value
prev_ave_abs_err = ave_abs_err  # initialize to any value

theta = rand_max * np.random.random(4)
prev_theta = 2 * theta + 1  # initialize such that not equal to theta
prev_prev_theta = 1000 * prev_theta  # initialize such that neither equal to theta nor prev_theta
prev_theta3 = prev_theta[3]  # initialize to any value
prev_theta2 = prev_theta[2]  # initialize to any value
prev_theta1 = prev_theta[1]  # initialize to any value
prev_theta0 = prev_theta[0]  # initialize to any value
abs_diff_theta = np.array([1, 1, 1, 1])  # initialize to any value > 1e-10 in the first loop

if(len(sys.argv) == 5):
    d = round(float(sys.argv[1]), 1)
    c = round(float(sys.argv[2]), 1)
    b = round(float(sys.argv[3]), 1)
    a = round(float(sys.argv[4]), 1)
if(len(sys.argv) == 4):
    d = float(0)
    c = round(float(sys.argv[1]), 1)
    b = round(float(sys.argv[2]), 1)
    a = round(float(sys.argv[3]), 1)
if(len(sys.argv) == 3):
    d = float(0)
    c = float(0)
    b = round(float(sys.argv[1]), 1)
    a = round(float(sys.argv[2]), 1)
if(len(sys.argv) == 2):
    d = float(0)
    c = float(0)
    b = float(0)
    a = round(float(sys.argv[1]), 1)
if(len(sys.argv) <= 0 or len(sys.argv) > 5):
    sys.exit()

x = rand_max * np.random.random(rand_size)
y = a * x**0 + b * x**1 + c * x**2 + d * x**3
# INSERT NOISE
n = np.random.random(rand_size) - np.random.random(rand_size)
y = y + n

break_log = []

if(len(sys.argv) >= 5):
    while ave_abs_err > accuracy:
        func_err = theta[0] * x**0 + theta[1] * x**1 + theta[2] * x**2 + theta[3] * x**3 - y
        abs_func_err = np.abs(func_err)

        prev_ave_abs_err = ave_abs_err
        ave_abs_err = np.average(abs_func_err)

        grad_func_loss = np.array([np.average(func_err * x**0), np.average(func_err * x**1), np.average(func_err * x**2), np.average(func_err * x**3)])

        norm_grad_func_loss = np.linalg.norm(grad_func_loss)

        unit_grad_func_loss = grad_func_loss / (norm_grad_func_loss + np.finfo(float).eps)

        step = learn_rate * norm_grad_func_loss
        abs_diff_abs_err = np.abs(prev_ave_abs_err - ave_abs_err)

        if(abs_diff_abs_err <= 0.1):
            if(ave_abs_err < 0.8):
                min_step = 1e-7
            elif(ave_abs_err < 1):
                min_step = 1e-5
            elif(ave_abs_err < 5):
                min_step = 1e-4
            elif (ave_abs_err < 15):
                min_step = 0.01

        if(abs_diff_abs_err > 0.1 and ave_abs_err < 80):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err < 0.01):  # Using 0.01 instead of 0.001 is vital to increase accuracy of the highest order parameter prediction. This, however, makes the first while loop iteration more dynamic or unsteady in terms of numerical movement. Depending on the polynomial coefficients, the program may take several minutes (especially if the coefficients are of the order 10^2 or lower) up to 1 hour or maybe more to complete. Please be patient and wait until it converges.
            min_step = 0.08

        if(abs_diff_abs_err > 0.5 and ave_abs_err > 80):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 0.5 and ave_abs_err > 80):
            min_step = 0.09

        if(abs_diff_abs_err > 1.5 and ave_abs_err > 150):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 1.5 and ave_abs_err > 150):
            min_step = 0.2

        if(abs_diff_abs_err > 2 and ave_abs_err > 250):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 2 and ave_abs_err > 250):
            min_step = 0.3

        if(abs_diff_abs_err > 3 and ave_abs_err > 400):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 3 and ave_abs_err > 400):
            min_step = 0.5

        if(abs_diff_abs_err > 3 and ave_abs_err > 500):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 3 and ave_abs_err > 500):
            min_step = 0.8

        if(abs_diff_abs_err > 5 and ave_abs_err > 1000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 5 and ave_abs_err > 1000):
            min_step = 2

        if(abs_diff_abs_err > 5 and ave_abs_err > 1500):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 5 and ave_abs_err > 1500):
            min_step = 3

        if(abs_diff_abs_err > 100 and ave_abs_err > 2000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 100 and ave_abs_err > 2000):
            min_step = 10

        if(abs_diff_abs_err > 1000 and ave_abs_err > 100000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 1000 and ave_abs_err > 100000):
            min_step = 10

        if(abs_diff_abs_err > 100 and ave_abs_err > 1500000):
            min_step = min_step / 3
            step = step / 3
        if(abs_diff_abs_err <= 100 and ave_abs_err > 1500000):
            min_step = 15

        if(abs_diff_abs_err > 1000 and ave_abs_err > 2000000):
            min_step = min_step / 3
            step = step / 3
        if(abs_diff_abs_err <= 1000 and ave_abs_err > 2000000):
            min_step = 15

        if(ave_abs_err > 100000):
            max_step = 1000
        if(ave_abs_err > 10000):
            max_step = 100

        if(abs_diff_theta[3] < 1e-10 and ave_abs_err > 0.4):
            min_step = 1e-7

        if (step > max_step):
            step = max_step
        if (step < min_step):
            step = min_step

        # if(prev_prev_theta[3] == prev_theta[3] or prev_prev_theta[3] == theta[3]):
        #     break_log = break_log + ['BREAK 3: Cyclic']
        #     break

        prev_prev_theta = prev_theta
        prev_theta = theta
        theta = theta - step * unit_grad_func_loss

        abs_diff_theta = np.abs(theta - prev_theta)
        abs_diff_theta_prime = np.abs(theta - prev_prev_theta)
        abs_diff_theta[3] = min(abs_diff_theta[3], abs_diff_theta_prime[3])  # This relaxes the 'BREAK: Cyclic' condition and lumps it in the other BREAK condition(s).
        print('Average Absolute Error:', ave_abs_err, 'Step Size:', step, '[theta[0], theta[1], theta[2], theta[3]]:', theta)

        if(abs_diff_theta[3] < 1e-4 and ave_abs_err < 0.4):
            break_log = break_log + ['BREAK 3: Small Parameter Difference and Small Average Absolute Error']
            break

    prev_theta3 = round(prev_theta[3], 1)
    min_step = 1e-5
else:
    prev_theta3 = round(0, 1)
theta = np.array([theta[0], theta[1], theta[2]])
prev_theta = np.array([prev_theta[0], prev_theta[1], prev_theta[2]])
prev_prev_theta = np.array([prev_prev_theta[0], prev_prev_theta[1], prev_prev_theta[2]])

if(len(sys.argv) >= 4):
    while ave_abs_err > accuracy:
        func_err = theta[0] * x**0 + theta[1] * x**1 + theta[2] * x**2 + prev_theta3 * x**3 - y
        abs_func_err = np.abs(func_err)

        prev_ave_abs_err = ave_abs_err
        ave_abs_err = np.average(abs_func_err)

        grad_func_loss = np.array([np.average(func_err * x**0), np.average(func_err * x**1), np.average(func_err * x**2)])

        norm_grad_func_loss = np.linalg.norm(grad_func_loss)

        unit_grad_func_loss = grad_func_loss / (norm_grad_func_loss + np.finfo(float).eps)

        step = learn_rate * norm_grad_func_loss
        abs_diff_abs_err = np.abs(prev_ave_abs_err - ave_abs_err)

        if(abs_diff_abs_err <= 0.1):
            if (ave_abs_err < 0.45):
                min_step = 1e-6
            elif(ave_abs_err < 0.5):
                min_step = 1e-5

        if(abs_diff_abs_err > 0.3 and ave_abs_err < 5):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err < 0.001):
            min_step = 1e-5

        if(abs_diff_abs_err > 0.5 and ave_abs_err > 5):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 0.5 and ave_abs_err > 5):
            min_step = 0.1

        if(abs_diff_abs_err > 1.5 and ave_abs_err > 50):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 1.5 and ave_abs_err > 50):
            min_step = 1

        if(abs_diff_abs_err > 2 and ave_abs_err > 100):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 2 and ave_abs_err > 100):
            min_step = 2.5

        if(abs_diff_abs_err > 3 and ave_abs_err > 400):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 3 and ave_abs_err > 400):
            min_step = 3

        if(abs_diff_abs_err > 3 and ave_abs_err > 500):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 3 and ave_abs_err > 500):
            min_step = 4

        if(abs_diff_abs_err > 5 and ave_abs_err > 1000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 5 and ave_abs_err > 1000):
            min_step = 5

        if(abs_diff_abs_err > 5 and ave_abs_err > 1500):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 5 and ave_abs_err > 1500):
            min_step = 7

        if(abs_diff_abs_err > 100 and ave_abs_err > 2000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 100 and ave_abs_err > 2000):
            min_step = 10

        if(abs_diff_abs_err > 1000 and ave_abs_err > 100000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 1000 and ave_abs_err > 100000):
            min_step = 10

        if(abs_diff_abs_err > 100 and ave_abs_err > 1500000):
            min_step = min_step / 3
            step = step / 3
        if(abs_diff_abs_err <= 100 and ave_abs_err > 1500000):
            min_step = 15

        if(abs_diff_abs_err > 1000 and ave_abs_err > 2000000):
            min_step = min_step / 3
            step = step / 3
        if(abs_diff_abs_err <= 1000 and ave_abs_err > 2000000):
            min_step = 15

        if(ave_abs_err > 100000):
            max_step = 1000
        if(ave_abs_err > 10000):
            max_step = 100

        if(abs_diff_theta[2] < 1e-10 and ave_abs_err > 0.37):
            min_step = 1e-6

        if (step > max_step):
            step = max_step
        if (step < min_step):
            step = min_step

        # if(prev_prev_theta[2] == prev_theta[2] or prev_prev_theta[2] == theta[2]):
        #     break_log = break_log + ['BREAK 2: Cyclic']
        #     break

        prev_prev_theta = prev_theta
        prev_theta = theta
        theta = theta - step * unit_grad_func_loss

        abs_diff_theta = np.abs(theta - prev_theta)
        abs_diff_theta_prime = np.abs(theta - prev_prev_theta)
        abs_diff_theta[2] = min(abs_diff_theta[2], abs_diff_theta_prime[2])
        print('Average Absolute Error:', ave_abs_err, 'Step Size:', step, '[theta[0], theta[1], theta[2]]:', theta, 'theta[3]:', prev_theta3)

        if(abs_diff_theta[2] < 1e-8 and ave_abs_err < 0.37):
            break_log = break_log + ['BREAK 2: Small Parameter Difference and Small Average Absolute Error']
            break

    prev_theta2 = round(prev_theta[2], 1)
    min_step = 1e-5
else:
    prev_theta2 = round(0, 1)
theta = np.array([theta[0], theta[1]])
prev_theta = np.array([prev_theta[0], prev_theta[1]])
prev_prev_theta = np.array([prev_prev_theta[0], prev_prev_theta[1]])

if(len(sys.argv) >= 3):
    while ave_abs_err > accuracy:
        func_err = theta[0] * x**0 + theta[1] * x**1 + prev_theta2 * x**2 + prev_theta3 * x**3 - y
        abs_func_err = np.abs(func_err)

        prev_ave_abs_err = ave_abs_err
        ave_abs_err = np.average(abs_func_err)

        grad_func_loss = np.array([np.average(func_err * x**0), np.average(func_err * x**1)])

        norm_grad_func_loss = np.linalg.norm(grad_func_loss)

        unit_grad_func_loss = grad_func_loss / (norm_grad_func_loss + np.finfo(float).eps)

        step = learn_rate * norm_grad_func_loss
        abs_diff_abs_err = np.abs(prev_ave_abs_err - ave_abs_err)

        if(abs_diff_abs_err <= 0.1):
            if(ave_abs_err < 0.12):
                min_step = 1e-8
            elif(ave_abs_err < 0.15):
                min_step = 1e-7
            elif(ave_abs_err < 0.2):
                min_step = 1e-6
            elif (ave_abs_err < 0.4):
                min_step = 1e-5

        if(abs_diff_abs_err > 0.1 and ave_abs_err < 5):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err < 0.001):
            min_step = 1e-5

        if(abs_diff_abs_err > 0.5 and ave_abs_err > 5):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 0.5 and ave_abs_err > 5):
            min_step = 0.1

        if(abs_diff_abs_err > 1.5 and ave_abs_err > 50):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 1.5 and ave_abs_err > 50):
            min_step = 1

        if(abs_diff_abs_err > 2 and ave_abs_err > 100):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 2 and ave_abs_err > 100):
            min_step = 2.5

        if(abs_diff_abs_err > 3 and ave_abs_err > 400):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 3 and ave_abs_err > 400):
            min_step = 3

        if(abs_diff_abs_err > 3 and ave_abs_err > 500):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 3 and ave_abs_err > 500):
            min_step = 4

        if(abs_diff_abs_err > 5 and ave_abs_err > 1000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 5 and ave_abs_err > 1000):
            min_step = 5

        if(abs_diff_abs_err > 5 and ave_abs_err > 1500):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 5 and ave_abs_err > 1500):
            min_step = 7

        if(abs_diff_abs_err > 100 and ave_abs_err > 2000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 100 and ave_abs_err > 2000):
            min_step = 10

        if(abs_diff_abs_err > 1000 and ave_abs_err > 100000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 1000 and ave_abs_err > 100000):
            min_step = 10

        if(abs_diff_abs_err > 100 and ave_abs_err > 1500000):
            min_step = min_step / 3
            step = step / 3
        if(abs_diff_abs_err <= 100 and ave_abs_err > 1500000):
            min_step = 15

        if(abs_diff_abs_err > 1000 and ave_abs_err > 2000000):
            min_step = min_step / 3
            step = step / 3
        if(abs_diff_abs_err <= 1000 and ave_abs_err > 2000000):
            min_step = 15

        if(ave_abs_err > 100000):
            max_step = 1000
        if(ave_abs_err > 10000):
            max_step = 100

        if(abs_diff_theta[1] < 1e-8 and ave_abs_err > 0.35):
            min_step = 1e-6

        if (step > max_step):
            step = max_step
        if (step < min_step):
            step = min_step

        # if(prev_prev_theta[1] == prev_theta[1] or prev_prev_theta[1] == theta[1]):
        #     break_log = break_log + ['BREAK 1: Cyclic']
        #     break

        prev_prev_theta = prev_theta
        prev_theta = theta
        theta = theta - step * unit_grad_func_loss

        abs_diff_theta = np.abs(theta - prev_theta)
        abs_diff_theta_prime = np.abs(theta - prev_prev_theta)
        abs_diff_theta[1] = min(abs_diff_theta[1], abs_diff_theta_prime[1])
        print('Average Absolute Error:', ave_abs_err, 'Step Size:', step, '[theta[0], theta[1]]:', theta, 'theta[2]:', prev_theta2, 'theta[3]:', prev_theta3)

        if(abs_diff_theta[1] < 1e-8 and ave_abs_err < 0.35):
            break_log = break_log + ['BREAK 1: Small Parameter Difference and Small Average Absolute Error']
            break

    prev_theta1 = round(prev_theta[1], 1)
    min_step = 1e-6
else:
    prev_theta1 = round(0, 1)
theta = np.array([theta[0]])
prev_theta = np.array([prev_theta[0]])
prev_prev_theta = np.array([prev_prev_theta[0]])

if(len(sys.argv) >= 2):
    while ave_abs_err > accuracy:
        func_err = theta[0] * x**0 + prev_theta1 * x**1 + prev_theta2 * x**2 + prev_theta3 * x**3 - y
        abs_func_err = np.abs(func_err)

        prev_ave_abs_err = ave_abs_err
        ave_abs_err = np.average(abs_func_err)

        grad_func_loss = np.array([np.average(func_err * x**0)])

        norm_grad_func_loss = np.linalg.norm(grad_func_loss)

        unit_grad_func_loss = grad_func_loss / (norm_grad_func_loss + np.finfo(float).eps)

        step = learn_rate * norm_grad_func_loss
        abs_diff_abs_err = np.abs(prev_ave_abs_err - ave_abs_err)

        if(abs_diff_abs_err <= 0.1):
            if(ave_abs_err < 0.12):
                min_step = 1e-8
            elif(ave_abs_err < 0.15):
                min_step = 1e-7
            elif(ave_abs_err < 0.2):
                min_step = 1e-6
            elif (ave_abs_err < 0.4):
                min_step = 1e-5

        if(abs_diff_abs_err > 0.1 and ave_abs_err < 5):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err < 0.001):
            min_step = 1e-5

        if(abs_diff_abs_err > 0.5 and ave_abs_err > 5):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 0.5 and ave_abs_err > 5):
            min_step = 0.1

        if(abs_diff_abs_err > 1.5 and ave_abs_err > 50):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 1.5 and ave_abs_err > 50):
            min_step = 1

        if(abs_diff_abs_err > 2 and ave_abs_err > 100):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 2 and ave_abs_err > 100):
            min_step = 2.5

        if(abs_diff_abs_err > 3 and ave_abs_err > 400):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 3 and ave_abs_err > 400):
            min_step = 3

        if(abs_diff_abs_err > 3 and ave_abs_err > 500):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 3 and ave_abs_err > 500):
            min_step = 4

        if(abs_diff_abs_err > 5 and ave_abs_err > 1000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 5 and ave_abs_err > 1000):
            min_step = 5

        if(abs_diff_abs_err > 5 and ave_abs_err > 1500):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 5 and ave_abs_err > 1500):
            min_step = 7

        if(abs_diff_abs_err > 100 and ave_abs_err > 2000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 100 and ave_abs_err > 2000):
            min_step = 10

        if(abs_diff_abs_err > 1000 and ave_abs_err > 100000):
            min_step = min_step / 2
            step = step / 2
        if(abs_diff_abs_err <= 1000 and ave_abs_err > 100000):
            min_step = 10

        if(abs_diff_abs_err > 100 and ave_abs_err > 1500000):
            min_step = min_step / 3
            step = step / 3
        if(abs_diff_abs_err <= 100 and ave_abs_err > 1500000):
            min_step = 15

        if(abs_diff_abs_err > 1000 and ave_abs_err > 2000000):
            min_step = min_step / 3
            step = step / 3
        if(abs_diff_abs_err <= 1000 and ave_abs_err > 2000000):
            min_step = 15

        if(ave_abs_err > 100000):
            max_step = 1000
        if(ave_abs_err > 10000):
            max_step = 100

        if(abs_diff_theta[0] < 1e-8 and ave_abs_err > 0.35):
            min_step = 1e-6

        if (step > max_step):
            step = max_step
        if (step < min_step):
            step = min_step

        # if(prev_prev_theta[0] == prev_theta[0] or prev_prev_theta[0] == theta[0]):
        #     break_log = break_log + ['BREAK 0: Cyclic']
        #     break

        prev_prev_theta = prev_theta
        prev_theta = theta
        theta = theta - step * unit_grad_func_loss

        abs_diff_theta = np.abs(theta - prev_theta)
        abs_diff_theta_prime = np.abs(theta - prev_prev_theta)
        abs_diff_theta[0] = min(abs_diff_theta[0], abs_diff_theta_prime[0])
        print('Average Absolute Error:', ave_abs_err, 'Step Size:', step, '[theta[0]]:', theta, 'theta[1]:', prev_theta1, 'theta[2]:', prev_theta2, 'theta[3]:', prev_theta3)

        if(abs_diff_theta[0] < 1e-8 and ave_abs_err < 0.35):
            break_log = break_log + ['BREAK 0: Small Parameter Difference and Small Average Absolute Error']
            break

    prev_theta0 = round(prev_theta[0], 1)
else:
    prev_theta0 = round(0, 1)

for i in break_log:
    print(i)
print('Done:')
print('theta[0]:', prev_theta0, 'theta[1]:', prev_theta1, 'theta[2]:', prev_theta2, 'theta[3]:', prev_theta3)
print('y = theta[0] * x**0 + theta[1] * x**1 + theta[2] ** x**2 +theta[3] * x**3')
