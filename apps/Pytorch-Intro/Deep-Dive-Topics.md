### Back Propagation:
### Pytorch Autograd: 
### vector-Jacobian product:
### derivatives:
Derivative is the measurement of slope (value) of a curve at an exact point. 

slope for straight line = `Change in y` divided by `Change in x` or `delta y` / `delta x` or `dy`/`dx`. Where y is the vertical line and x is the horizontal line in a graph. 

slope of curved line (Derivative) = `f(a+h) = f(a)` divided by `(a+h) - a` = `f(a+h) - f(a)` / `h`. Where `a` is in `x` (horizontal) axis, `f(a)` is in `y` (vertical) axis and `h` is distance. 

Derivative limit : a h approaches to 0 `zero` from the exact point to the starting point of measurement.

Ex: if f(x) = x^2, measure slope when x = 3
if distance = h
(f(x+h)^2 - f(x)^2) / ((x + h) - x)
= ((3+h)^2 - 3^2) / (3 + h -3)
= (9 + 6h + h^2 - 9) / h
= h(6 + h) / h
= 6 + h

and, limit (when h = 0) will be 6 + 0 = 6

So the slope is 6 where x = 3


* General derivative equation for `f(x) = x^2` is `(f(x+h)^2 - f(x)^2) / h`
### partial derivate:
https://youtu.be/kdMep5GUOBw?si=rUFsipNChaYfvXEm