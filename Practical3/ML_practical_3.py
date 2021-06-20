#!/usr/bin/env python
# coding: utf-8

# # 1.Implement your own Scalar and Vector classes, without using any other modules:

# In[1]:


from typing import Union, List
from math import sqrt


class Scalar:
    pass
class Vector:
    pass

class Scalar:
    def __init__(self: Scalar, val: float):
        self.val = float(val)

    def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]:
        scalar = isinstance(other, Scalar)
        vector = isinstance(other, Vector)

        if scalar:
            return Scalar(self.val * other.val)
        elif vector:
            return Vector(*[n * self.val for n in other.entries])
        else:
            raise TypeError('should be scalar or vector')

    # hint: use isinstance to decide what `other` is
    # raise an error if `other` isn't Scalar or Vector!
    
    def __add__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val + other.val)
        
    def __sub__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val - other.val)
        
    def __truediv__(self: Scalar, other: Scalar) -> Scalar:
        if isinstance(other, Scalar):
            return Scalar(self.val / other.val) # implement division of scalars
    
    def __rtruediv__(self: Scalar, other: Vector) -> Vector:
        if isinstance(other, Vector):
            return Vector(*[n / self.val for n in other.entries])
     # implement division of vector by scalar
    
    def __repr__(self: Scalar) -> str:
        return "Scalar(%r)" % self.val
    def sign(self: Scalar) -> int:
        pass # returns -1, 0, or 1
    def __float__(self: Scalar) -> float:
        return self.val

class Vector:
    def __init__(self: Vector, *entries: List[float]):
        self.entries = entries
    def zero(size: int) -> Vector:
        return Vector(*[0 for i in range(size)])

    def __add__(self: Vector, other: Vector) -> Vector:
        if len(self.entries) == len(other):
            res = [self.entries[i] + other.entries[i] for i in range(len(self.entries))]
            return Vector(*res)
        else: 
            raise TypeError('The vectors are of different lengths')

    def __sub__(self: Vector, other: Vector) -> Vector:
        if len(self.entries) == len(other):
            res = [self.entries[i] - other.entries[i] for i in range(len(self.entries))]
            return Vector(*res)
        else: 
            raise TypeError('The vectors are of different lengths')
   
    def __mul__(self: Vector, other: Vector) -> Scalar:
        if len(self.entries) == len(other):
            dot_prod = 0
            for i in range(len(self.entries)):
                dot_prod += self.entries[i] * other.entries[i]
            return Scalar(dot_prod)
        else:
            raise TypeError('The vectors are of different lengths')

    def magnitude(self: Vector) -> Scalar:
        summ = 0
        for i in range(len(self.entries)):
            summ += self.entries[i]**2    
        return Scalar(sqrt(summ))
      
    def unit(self: Vector) -> Vector:
        return self / self.magnitude()
    def __len__(self: Vector) -> int:
        return len(self.entries)
    def __repr__(self: Vector) -> str:
        return "Vector%s" % repr(self.entries)
    def __iter__(self: Vector):
        return iter(self.entries)


# #   2.Implement the PerceptronTrain and PerceptronTest functions, using your Vector and Scalar classes. Do not permute the dataset when training; run through it linearly.

# In[2]:


def PerceptronTrain(D, Maxiter):
    w = Vector.zero(len(D[0][0]))
    b = Scalar(0)

    for i in range(Maxiter):
        for X, y in D:
            a = X * w + b
            if (y * a).val <= 0: 
                w += y * X
                b += y
    return w, b


# In[3]:


def PerceptronTest(w, b, X):
    activation = X * weights + bias
    return activation.sign()


# # 3.Make a 90-10 test-train split and evaluate your algorithm on the following dataset:

# In[4]:


from random import randint

v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]


split = int(0.9 * len(xs))
x_train, x_test = xs[:split], xs[split:]
y_train, y_test = ys[:split], ys[split:]

D =  list(zip(x_train, y_train))
test = zip(x_test, y_test)

weights, bias = PerceptronTrain(D, 100)

score = 0
total = len(x_test)
for x_test, label in test:
    score += PerceptronTest(weights, bias, x_test) == label.sign()

print("Correctly predicts: " + str(score) + " out of " + str(total))


# # 4.Make a 90-10 test-train split and evaluate your algorithm on the xor dataset:

# In[12]:


from random import randint
xs_xor = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys_xor = [Scalar(1) if x.entries[0] * x.entries[1] < 0 else Scalar(-1) for x in xs_xor]

split = int(0.9 * len(xs_xor))
xs_xor_train, xs_xor_test = xs_xor[:split], xs_xor[split:]
ys_xor_train, ys_xor_test = ys_xor[:split], ys_xor[split:]

D =  list(zip(xs_xor_train, ys_xor_train))
test = zip(xs_xor_test, ys_xor_test)

w, b = PerceptronTrain(D, 10)


# In[14]:


score = 0
total = len(xs_xor_test)
for xs_xor_test, label in test:
    score += PerceptronTest(w, b, xs_xor_test) == label.sign()

print("Correctly predicts: " + str(score) + " out of " + str(total))






