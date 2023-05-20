# -*- coding: utf-8 -*-
"""
Created on Sun Feb 09 13:50:10 2014
"""

import numpy as np


class numerov:
    def __init__(self, func, a, b):
        """

        :param func: function to be integrated
        :param a: lower bound for integration
        :param b: upper bound for integration
        """
        # TODO: make steps an __init__ argument.
        # TODO: don't assign class members in functions

        self.a = float(a)
        self.b = float(b)
        self.int_len = self.b - self.a
        self.func = func
        self.f = None
        self.rpts = None
        self.xpts = None
        self.w = None
        self.g = None

    def back_numerov(self, steps):
        """solve equations of the form y(x)''-f(x)y(x)=0, beginning at end point
       and working backwards. Set initial values in code."""
        # ensure step value is an integer.
        if steps.__class__ != int:
            raise ValueError("You must have an integer numer of steps")
        else:
            pass
        # set step size
        h = (self.b - self.a) / steps
        # initialize lists
        y = np.zeros(steps)
        self.f = np.zeros(steps)
        self.rpts = np.zeros(steps)
        # set sample points and sample function
        for n in range(0, steps):
            self.f[n] = self.func(self.a + n * h)
            self.rpts[n] = self.a + n * h
        # set initial values
        y[steps - 1] = -1e-5
        y[steps - 2] = -1e-2
        # implement numerov
        for n in reversed(range(0, steps - 2)):
            y[n] = (y[n + 1] * (2 + 5 * h ** 2 * self.f[n + 1] / 6) - y[n + 2] *
                    (1 - h ** 2 * self.f[n + 2] / 12)) / \
                    (1 - h ** 2 * self.f[n] / 12)
        return y

    def forw_numerov(self, steps):
        """solve equations of the form y(x)''-f(x)y(x)=0, beginning at inital point
       and moving forwards. Set initial values in code"""
        # ensure step value is an integer.
        if steps.__class__ != int:
            raise ValueError("You must have an integer numer of steps")
        else:
            pass
        # set step size
        h = (self.b - self.a) / float(steps)
        # initialize lists
        y = np.zeros(steps)
        self.f = np.zeros(steps)
        self.rpts = np.zeros(steps)
        # set sample points and sample function
        for n in range(0, steps):
            self.f[n] = self.func(self.a + n * h)
            self.rpts[n] = self.a + n * h
        # set initial values
        y[0] = 0
        y[1] = h
        # implement numerov
        for n in range(2, steps):
            y[n] = (y[n - 1] * (2 + 5 * h ** 2 * self.f[n - 1] / 6) - y[n - 2] *
                    (1 - h ** 2 * self.f[n - 2] / 12)) / \
                   (1 - h ** 2 * self.f[n] / 12)
        return y

    def log_back_numerov(self, steps):
        """Equation of form y(r)''-f(r)y(r)=0 can be transformed into:
       w''-g(x)w(x)=0 where x=log(r), w=y(x)e^(-x/2), and g(x)=(0.25+exp(2x)f(x)).
       Remember that it is now necessary to plot again self.rpts, as points are no longer
       linearly distributed along r (being linearly distributed along x=log(r))"""

        def g(x):
            return 0.25 + np.exp(2 * x) * self.func(np.exp(x))

        # set step size
        h = (np.log(self.b) - np.log(self.a)) / steps
        # initial lists
        w = np.zeros(steps)
        self.g = np.zeros(steps)
        self.xpts = np.zeros(steps)
        # set sample points and sample funtion
        for n in range(0, steps):
            self.g[n] = g(np.log(self.a) + n * h)
            self.xpts[n] = np.log(self.a) + n * h
        self.rpts = np.exp(self.xpts)
        # set initial values
        w[steps - 1] = -1e-5
        w[steps - 2] = -1e-2
        # implement
        for n in reversed(range(0, steps - 2)):
            w[n] = (w[n + 1] * (2 + 5 * h**2 * self.g[n + 1] / 6) - w[n + 2] * (1 - h**2 * self.g[n + 2] / 12)) / \
                   (1 - h**2 * self.g[n] / 12)
        self.w = w
        y = w * np.sqrt(self.rpts)
        return y

    def quarter_back_numerov(self, steps):
        """Equation of the form y(r)''-f(r)y(r)=0 can be transformed into:
       w''-g(x)w(x)=0 where x =sqrt(r), w(x)=y(x)/sqrt(x), and g(x)=4x^2*f(x) +3/(4x^2).
       Remember that it is now necessary to plot again self.rpts, as points are no longer
       linearly distributed along r (being linearly distributed along x=sqrt(r))"""

        def g(x):
            return 4. * x ** 2 * self.func(x ** 2) + 3. / (4. * x ** 2)

        # set step size
        h = (np.sqrt(self.b) - np.sqrt(self.a)) / steps
        # initial lists
        w = np.zeros(steps)
        self.g = np.zeros(steps)
        self.xpts = np.zeros(steps)
        self.rpts = np.zeros(steps)
        # set sample points and sample funtion
        for n in range(0, steps):
            self.g[n] = g(np.sqrt(self.a) + n * h)
            self.xpts[n] = np.sqrt(self.a) + n * h
            self.rpts[n] = self.xpts[n] ** 2
        # set initial values
        w[steps - 1] = -1e-5
        w[steps - 2] = -1e-2
        # implement
        for n in reversed(range(0, steps - 2)):
            w[n] = (w[n + 1] * (2 + 5 * h ** 2 * self.g[n + 1] / 6) - w[n + 2] *
                    (1 - h ** 2 * self.g[n + 2] / 12)) / \
                   (1 - h ** 2 * self.g[n] / 12)
        self.w = w
        y = w * np.sqrt(np.sqrt(self.rpts))
        return y
