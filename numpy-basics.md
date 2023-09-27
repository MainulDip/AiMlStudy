### Numpy vs Python List:
Numpy (Numerical Python) is written in c, that's why it's way faster than python's list in terms of list related operations (heavy/large-scale computation). 
- behind the scenes optimization written in C
- Vectorization via broadcasting (avoiding loops)
- Backbone of Python scientific packages as well as AI/ML

### Topics
- most used functions
- NumPy Data Types and attributes (ndarray)
- Creating array
- Viewing arrays & matrices
- Manipulating & Comparing arrays
- Sorting arrays
- Use Cases

### NumPy Dot product:
Another way of finding pattern between 2 sets of numbers

- Rules:
    - Number (shape) on the inside (horizontal) must match vertical shape of the other array. (3,3)x(3,2) || (5,3)x(3,5) will match, but (3,5)x(3,5) will not
    - New size is same as outside number
    - Outside & Inside: if (5,3) x (3,5) : both 5 are outside and both 3 are inside

Matrix multiplication operation (3,3) x (3,2) = (3,2).

|   |3,3|   |  .dot |   |   |        |          |          |
|---|---|---|-------|---|---|--------|----------|----------|
| A | B | C |       | J | k |        | AJ+BL+CN | AK+BM+CO |
| D | E | F |       | L | M |        | DJ+EL+FN | DK+EM+FO |
| G | H | I |       | N | O |        | GJ+HL+IN | GK+HM+IO |