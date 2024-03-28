"""

Source  `Kevin Markham <https://github.com/justmarkham/python-reference>`_

"""

###############################################################################
# Import libraries
# ----------------
#

# 'generic import' of math module
import math
math.sqrt(25)

# import a function
from math import sqrt
sqrt(25)    # no longer have to reference the module

# import multiple functions at once

# import all functions in a module (generally discouraged)
# from os import *

# define an alias

# show all functions in math module
content = dir(math)

###############################################################################
# Basic operations
# ----------------
#

# Numbers
10 + 4          # add (returns 14)
10 - 4          # subtract (returns 6)
10 * 4          # multiply (returns 40)
10 ** 4         # exponent (returns 10000)
10 / 4          # divide (returns 2 because both types are 'int')
10 / float(4)   # divide (returns 2.5)
5 % 4           # modulo (returns 1) - also known as the remainder

10 / 4          # true division (returns 2.5)
10 // 4         # floor division (returns 2)


# Boolean operations
# comparisons (these return True)
5 > 3
5 >= 3
5 != 3
5 == 5

# Boolean operations (these return True)
5 > 3 and 6 > 3
5 > 3 or 5 < 3
not False
False or not False and True     # evaluation order: not, and, or


###############################################################################
# Data types
# ----------
#

# determine the type of an object
type(2)         # returns 'int'
type(2.0)       # returns 'float'
type('two')     # returns 'str'
type(True)      # returns 'bool'
type(None)      # returns 'NoneType'

# check if an object is of a given type
isinstance(2.0, int)            # returns False
isinstance(2.0, (int, float))   # returns True

# convert an object to a given type
float(2)
int(2.9)
str(2.9)

# zero, None, and empty containers are converted to False
bool(0)
bool(None)
bool('')    # empty string
bool([])    # empty list
bool({})    # empty dictionary

# non-empty containers and non-zeros are converted to True
bool(2)
bool('two')
bool([2])


###############################################################################
# Lists
# ~~~~~
#
# Different objects categorized along a certain ordered sequence, lists
# are ordered, iterable, mutable (adding or removing objects changes the
# list size), can contain multiple data types.


# create an empty list (two ways)
empty_list = []
empty_list = list()

# create a list
simpsons = ['homer', 'marge', 'bart']

# examine a list
simpsons[0]     # print element 0 ('homer')
len(simpsons)   # returns the length (3)

# modify a list (does not return the list)
simpsons.append('lisa')                 # append element to end
simpsons.extend(['itchy', 'scratchy'])  # append multiple elements to end
# insert element at index 0 (shifts everything right)
simpsons.insert(0, 'maggie')
# searches for first instance and removes it
simpsons.remove('bart')
simpsons.pop(0)                         # removes element 0 and returns it
# removes element 0 (does not return it)
del simpsons[0]
simpsons[0] = 'krusty'                  # replace element 0

# concatenate lists (slower than 'extend' method)
neighbors = simpsons + ['ned', 'rod', 'todd']

# find elements in a list
'lisa' in simpsons
simpsons.count('lisa')      # counts the number of instances
simpsons.index('itchy')     # returns index of first instance

# list slicing [start:end:stride]
weekdays = ['mon', 'tues', 'wed', 'thurs', 'fri']
weekdays[0]         # element 0
weekdays[0:3]       # elements 0, 1, 2
weekdays[:3]        # elements 0, 1, 2
weekdays[3:]        # elements 3, 4
weekdays[-1]        # last element (element 4)
weekdays[::2]       # every 2nd element (0, 2, 4)
weekdays[::-1]      # backwards (4, 3, 2, 1, 0)

# alternative method for returning the list backwards
list(reversed(weekdays))

# sort a list in place (modifies but does not return the list)
simpsons.sort()
simpsons.sort(reverse=True)     # sort in reverse
simpsons.sort(key=len)          # sort by a key

# return a sorted list (but does not modify the original list)
sorted(simpsons)
sorted(simpsons, reverse=True)
sorted(simpsons, key=len)

###############################################################################
# Reference and copy
# ~~~~~~~~~~~~~~~~~~
#
# `References <https://levelup.gitconnected.com/understanding-reference-and-copy-in-python-c681341a0cd8>`_ are used to access objects in memory, here lists.
# A single object may have multiple references. Modifying the content of the one reference
# will change the content of all other references.


num = [1, 2, 3]
same_num = num   # create a second reference to the same list
same_num[0] = 0  # modifies both 'num' and 'same_num'
print(same_num)

###############################################################################
# Copies are references to different objects.
# Modifying the content of the one reference, will not affect the others.

# copy a list (three ways)
new_num = num.copy()
new_num = num[:]
new_num = list(num)

# examine objects
id(num) == id(same_num)  # returns True
id(num) == id(new_num)  # returns False
num is same_num         # returns True
num is new_num          # returns False
num == same_num         # returns True
num == new_num          # returns True (their contents are equivalent)

# conatenate +, replicate *
[1, 2, 3] + [4, 5, 6]
["a"] * 2 + ["b"] * 3


###############################################################################
# Tuples
# ~~~~~~
#
# Like lists, but their size cannot change: ordered, iterable, immutable,
# can contain multiple data types
#

# create a tuple
digits = (0, 1, 'two')          # create a tuple directly
digits = tuple([0, 1, 'two'])   # create a tuple from a list
# trailing comma is required to indicate it's a tuple
zero = (0,)

# examine a tuple
digits[2]           # returns 'two'
len(digits)         # returns 3
digits.count(0)     # counts the number of instances of that value (1)
digits.index(1)     # returns the index of the first instance of that value (1)

# elements of a tuple cannot be modified
# digits[2] = 2       # throws an error

# concatenate tuples
digits = digits + (3, 4)

# create a single tuple with elements repeated (also works with lists)
(3, 4) * 2          # returns (3, 4, 3, 4)

# tuple unpacking
bart = ('male', 10, 'simpson')  # create a tuple


###############################################################################
# Strings
# ~~~~~~~
#
# A sequence of characters, they are iterable, immutable
#

# create a string
s = str(42)         # convert another data type into a string
s = 'I like you'

# examine a string
s[0]                # returns 'I'
len(s)              # returns 10

# string slicing like lists
s[:6]               # returns 'I like'
s[7:]               # returns 'you'
s[-1]               # returns 'u'

# basic string methods (does not modify the original string)
s.lower()           # returns 'i like you'
s.upper()           # returns 'I LIKE YOU'
s.startswith('I')   # returns True
s.endswith('you')   # returns True
s.isdigit()         # returns False (True if every character is a digit)
s.find('like')      # returns index of first occurrence
s.find('hate')      # returns -1 since not found
s.replace('like', 'love')    # replaces all instances of 'like' with 'love'

# split a string into a list of substrings separated by a delimiter
s.split(' ')        # returns ['I','like','you']
s.split()           # same thing
s2 = 'a, an, the'
s2.split(',')       # returns ['a',' an',' the']

# join a list of strings into one string using a delimiter
stooges = ['larry', 'curly', 'moe']
' '.join(stooges)   # returns 'larry curly moe'

# concatenate strings
s3 = 'The meaning of life is'
s4 = '42'
s3 + ' ' + s4       # returns 'The meaning of life is 42'
s3 + ' ' + str(42)  # same thing

# remove whitespace from start and end of a string
s5 = '  ham and cheese  '
s5.strip()          # returns 'ham and cheese'

###############################################################################
# Strings formatting

# string substitutions: all of these return 'raining cats and dogs'
'raining %s and %s' % ('cats', 'dogs')                       # old way
'raining {} and {}'.format('cats', 'dogs')                   # new way
'raining {arg1} and {arg2}'.format(arg1='cats', arg2='dogs')  # named arguments

# String formatting
# See: https://realpython.com/python-formatted-output/
# Old method
print('6 %s' % 'bananas')
print('%d %s cost $%.1f' % (6, 'bananas', 3.14159))

# Format method positional arguments
print('{0} {1} cost ${2:.1f}'.format(6, 'bananas', 3.14159))

###############################################################################
# `Strings encoding <https://towardsdatascience.com/byte-string-unicode-string-raw-string-a-guide-to-all-strings-in-python-684c4c4960ba>`_

###############################################################################
# Normal strings allow for escaped characters. The default strings use unicode string (u string)
#

print('first line\nsecond line')  # or
print(u'first line\nsecond line')
print('first line\nsecond line' == u'first line\nsecond line')

###############################################################################
# Raw strings treat backslashes as literal characters
#

print(r'first line\nfirst line')
print('first line\nsecond line' == r'first line\nsecond line')

###############################################################################
# Sequence of bytes are not strings, should be decoded before some operations
#

s = b'first line\nsecond line'
print(s)

print(s.decode('utf-8').split())


###############################################################################
# Dictionaries
# ~~~~~~~~~~~~
#
# **Dictionary is the must-known data structure**.
# Dictionaries are structures which can contain multiple data types, and
# is ordered with key-value pairs: for each (unique) key, the dictionary
# outputs one value. Keys can be strings, numbers, or tuples, while the
# corresponding values can be any Python object. Dictionaries are:
# unordered, iterable, mutable
#

###############################################################################
# Creation

# Empty dictionary (two ways)
empty_dict = {}
empty_dict = dict()

simpsons_roles_dict = {'Homer': 'father', 'Marge': 'mother',
                       'Bart': 'son', 'Lisa': 'daughter', 'Maggie': 'daughter'}

simpsons_roles_dict = dict(Homer='father', Marge='mother',
                           Bart='son', Lisa='daughter', Maggie='daughter')

simpsons_roles_dict = dict([('Homer', 'father'), ('Marge', 'mother'),
                            ('Bart', 'son'), ('Lisa', 'daughter'), ('Maggie', 'daughter')])

print(simpsons_roles_dict)


###############################################################################
# Access

# examine a dictionary
simpsons_roles_dict['Homer']   # 'father'
len(simpsons_roles_dict)       # 5
simpsons_roles_dict.keys()     # list: ['Homer', 'Marge', ...]
simpsons_roles_dict.values()   # list:['father', 'mother', ...]
simpsons_roles_dict.items()    # list of tuples: [('Homer', 'father') ...]
'Homer' in simpsons_roles_dict  # returns True
'John' in simpsons_roles_dict  # returns False (only checks keys)

# accessing values more safely with 'get'
simpsons_roles_dict['Homer']                       # returns 'father'
simpsons_roles_dict.get('Homer')                   # same thing

try:
    simpsons_roles_dict['John']               # throws an error
except KeyError as e:
    print("Error", e)

simpsons_roles_dict.get('John')               # None
# returns 'not found' (the default)
simpsons_roles_dict.get('John', 'not found')

###############################################################################
# Modify a dictionary (does not return the dictionary)

simpsons_roles_dict['Snowball'] = 'dog'              # add a new entry
simpsons_roles_dict['Snowball'] = 'cat'              # add a new entry
simpsons_roles_dict['Snoop'] = 'dog'                 # edit an existing entry
del simpsons_roles_dict['Snowball']                  # delete an entry

simpsons_roles_dict.pop('Snoop')  # removes and returns ('dog')
simpsons_roles_dict.update(
    {'Mona': 'grandma', 'Abraham': 'grandpa'})  # add multiple entries
print(simpsons_roles_dict)

###############################################################################
# Intersecting two dictionaries

simpsons_ages_dict = {'Homer': 45, 'Marge': 43,
                      'Bart': 11, 'Lisa': 10, 'Maggie': 1}

print(simpsons_roles_dict.keys() & simpsons_ages_dict.keys())

###############################################################################
# String substitution using a dictionary: syntax ``%(key)format``, where ``format``
# is the formatting character e.g. ``s`` for string.

print('Homer is the %(Homer)s of the family' % simpsons_roles_dict)


###############################################################################
# Sets
# ~~~~
#
# Like dictionaries, but with unique keys only (no corresponding values).
# They are: unordered, iterable, mutable, can contain multiple data types
# made up of unique elements (strings, numbers, or tuples)
#

###############################################################################
# Creation

# create an empty set
empty_set = set()

# create a set
languages = {'python', 'r', 'java'}         # create a set directly
snakes = set(['cobra', 'viper', 'python'])  # create a set from a list

###############################################################################
# Examine a set
len(languages)              # 3
'python' in languages       # True

###############################################################################
# Set operations

languages & snakes          # intersection: {'python'}
languages | snakes          # union: {'cobra', 'r', 'java', 'viper', 'python'}
languages - snakes          # set difference: {'r', 'java'}
snakes - languages          # set difference: {'cobra', 'viper'}

# modify a set (does not return the set)
languages.add('sql')        # add a new element
# try to add an existing element (ignored, no error)
languages.add('r')
languages.remove('java')    # remove an element

try:
    languages.remove('c')   # remove a non-existing element: throws an error
except KeyError as e:
    print("Error", e)

# removes an element if present, but ignored otherwise
languages.discard('c')
languages.pop()             # removes and returns an arbitrary element
languages.clear()           # removes all elements
languages.update('go', 'spark')  # add multiple elements (list or set)

# get a sorted list of unique elements from a list
sorted(set([9, 0, 2, 1, 0]))    # returns [0, 1, 2, 9]

###############################################################################
# Execution control statements
# ----------------------------
#

###############################################################################
# Conditional statements
# ~~~~~~~~~~~~~~~~~~~~~~

###############################################################################
# if statement

x = 3
if x > 0:
    print('positive')

###############################################################################
# if/else statement

if x > 0:
    print('positive')
else:
    print('zero or negative')

###############################################################################
# Single-line if/else statement, known as a 'ternary operator'

sign = 'positive' if x > 0 else 'zero or negative'
print(sign)

###############################################################################
# if/elif/else statement

if x > 0:
    print('positive')
elif x == 0:
    print('zero')
else:
    print('negative')


###############################################################################
# Loops
# ~~~~~
#
# Loops are a set of instructions which repeat until termination
# conditions are met. This can include iterating through all values in an
# object, go through a range of values, etc
#

# range returns a list of integers
# returns [0, 1, 2]: includes first value but excludes second value
range(0, 3)
range(3)        # same thing: starting at zero is the default
range(0, 5, 2)  # returns [0, 2, 4]: third argument specifies the 'stride'

###############################################################################
# Iterate on list values

fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print(fruit.upper())

###############################################################################
# Iterate with index

for i in range(len(fruits)):
    print(fruits[i].upper())

###############################################################################
# Iterate with index and values: ``enumerate``

for i, val in enumerate(fruits):
    print(i, val.upper())

# Use range when iterating over a large sequence to avoid actually
# creating the integer list in memory
v = 0
for i in range(10 ** 6):
    v += 1

###############################################################################
# Example, use loop, dictionary and set to count words in a sentence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

quote = """Tick-tow
our incomes are like our shoes; if too small they gall and pinch us
but if too large they cause us to stumble and to trip
"""

words = quote.split()

count = {word: 0 for word in set(words)}
for word in words:
    count[word] += 1

print(count)
###############################################################################
# List comprehensions, iterators, etc.
# ------------------------------------
#
# List comprehensions
# ~~~~~~~~~~~~~~~~~~~
#
# `List comprehensions <http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html>`_
# provides an elegant syntax for the most common processing pattern:
#
# 1. iterate over a list,
# 2. apply some operation
# 3. store the result in a new list

###############################################################################
# Classical iteration over a list

nums = [1, 2, 3, 4, 5]
cubes = []
for num in nums:
    cubes.append(num**3)

###############################################################################
# Equivalent list comprehension

cubes = [num**3 for num in nums]    # [1, 8, 27, 64, 125]

###############################################################################
# Classical iteration over a list with **if condition**:
# create a list of cubes of even numbers

cubes_of_even = []
for num in nums:
    if num % 2 == 0:
        cubes_of_even.append(num**3)

###############################################################################
# Equivalent list comprehension with **if condition**
# syntax: ``[expression for variable in iterable if condition]``

cubes_of_even = [num**3 for num in nums if num % 2 == 0]    # [8, 64]

###############################################################################
# Classical iteration over a list with **if else condition**:
# for loop to cube even numbers and square odd numbers

cubes_and_squares = []
for num in nums:
    if num % 2 == 0:
        cubes_and_squares.append(num**3)
    else:
        cubes_and_squares.append(num**2)

###############################################################################
# Equivalent list comprehension (using a ternary expression)
# for loop to cube even numbers and square odd numbers
# syntax: ``[true_condition if condition else false_condition for variable in iterable]``

cubes_and_squares = [num**3 if num % 2 == 0 else num**2 for num in nums]
print(cubes_and_squares)

###############################################################################
# Nested loops: flatten a 2d-matrix

matrix = [[1, 2], [3, 4]]
items = []
for row in matrix:
    for item in row:
        items.append(item)

###############################################################################
# Equivalent list comprehension with Nested loops

items = [item for row in matrix
         for item in row]

print(items)

###############################################################################
# Set comprehension
# ~~~~~~~~~~~~~~~~~

fruits = ['apple', 'banana', 'cherry']
unique_lengths = {len(fruit) for fruit in fruits}
print(unique_lengths)

###############################################################################
# Dictionary comprehension
# ~~~~~~~~~~~~~~~~~~~~~~~~~

###############################################################################
# Create a dictionary from a list

fruit_lengths = {fruit: len(fruit) for fruit in fruits}
print(fruit_lengths)

###############################################################################
# Iterate over keys and values. Increase age of each subject:

simpsons_ages_ = {key: val + 1 for key, val in simpsons_ages_dict.items()}
print(simpsons_ages_)

###############################################################################
# Combine two dictionaries sharing key. Example, a function that joins two dictionaries
# (intersecting keys) into a dictionary of lists

simpsons_info_dict = {name: [simpsons_roles_dict[name], simpsons_ages_dict[name]]
                      for name in simpsons_roles_dict.keys() &
                      simpsons_ages_dict.keys()}
print(simpsons_info_dict)

###############################################################################
# Iterators ``itertools`` package
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import itertools

###############################################################################
# Example: Cartesian product

print([[x, y] for x, y in itertools.product(['a', 'b', 'c'], [1, 2])])

###############################################################################
# Exceptions handling
# ~~~~~~~~~~~~~~~~~~~
#

dct = dict(a=[1, 2], b=[4, 5])

key = 'c'
try:
    dct[key]
except:
    print("Key %s is missing. Add it with empty value" % key)
    dct['c'] = []

print(dct)

###############################################################################
# Functions
# ---------
#
# Functions are sets of instructions launched when called upon, they can
# have multiple input values and a return value
#

###############################################################################
# Function with no arguments and no return values


def print_text():
    print('this is text')


# call the function
print_text()

###############################################################################
# Function with one argument and no return values


def print_this(x):
    print(x)


# call the function
print_this(3)       # prints 3
n = print_this(3)   # prints 3, but doesn't assign 3 to n
# because the function has no return statement
print(n)

###############################################################################
# **Dynamic typing**
#
# Important remarque: **Python is a dynamically typed language**, meaning
# that the Python interpreter does type checking at runtime (as opposed to compiled
# language that are statically typed). As a consequence, the function behavior, decided,
# at execution time, will be different and specific to parameters type.
# Python function are polymorphic.


def add(a, b):
    return a + b


print(add(2, 3), add("deux", "trois"), add(["deux", "trois"], [2, 3]))

###############################################################################
# **Default arguments**


def power_this(x, power=2):
    return x ** power


print(power_this(2), power_this(2, 3))

###############################################################################
# **Docstring** to describe the effect of a function
# IDE, ipython (type: ?power_this) to provide function documentation.


def power_this(x, power=2):
    """Return the power of a number.

    Args:
        x (float): the number
        power (int, optional): the power. Defaults to 2.
    """
    return x ** power


###############################################################################
# **Return several values** as tuple


def min_max(nums):
    return min(nums), max(nums)


# return values can be assigned to a single variable as a tuple
min_max_num = min_max([1, 2, 3])         # min_max_num = (1, 3)

# return values can be assigned into multiple variables using tuple unpacking
min_num, max_num = min_max([1, 2, 3])    # min_num = 1, max_num = 3

###############################################################################
# Example, function, and dictionary comprehension
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Example of a function ``join_dict_to_table(dict1, dict2)`` joining two dictionaries
# (intersecting keys) into a table, i.e., a list of tuples, where the first column
# is the key, the second and third columns are the values of the dictionaries.


def join_dict_to_table(dict1, dict2):
    table = [[key] + [dict1[key], dict2[key]]
             for key in dict1.keys() & dict2.keys()]
    return table


print(join_dict_to_table(simpsons_roles_dict, simpsons_ages_dict))

###############################################################################
# Regular expression
# ------------------
# Regular Expression (RE, or RegEx) allow to search and patterns in strings.
# See `this page <https://www.programiz.com/python-programming/regex>`_ for the syntax
# of the RE patterns.

import re

###############################################################################
# **Usual patterns**
#
# - ``.`` period symbol matches any single character (except newline '\n').
# - pattern``+`` plus symbol matches one or more occurrences of the pattern.
# - ``[]`` square brackets specifies a set of characters you wish to match
# - ``[abc]`` matches a, b or c
# - ``[a-c]`` matches a to z
# - ``[0-9]`` matches 0 to 9
# - ``[a-zA-Z0-9]+`` matches words, at least one alphanumeric character (digits and alphabets)
# - ``[\w]+`` matches words, at least one alphanumeric character including underscore.
# - ``\s`` Matches where a string contains any whitespace character, equivalent to [ \t\n\r\f\v].
# - ``[^\s]`` Caret ``^`` symbol (the start of a square-bracket) inverts the pattern selection .

# regex = re.compile("^.+(firstname:.+)_(lastname:.+)_(mod-.+)")
# regex = re.compile("(firstname:.+)_(lastname:.+)_(mod-.+)")

###############################################################################
# **Compile** (``re.compile(string)``) regular expression with a pattern that
# captures the pattern ``firstname:<subject_id>_lastname:<session_id>``
pattern = re.compile("firstname:[\w]+_lastname:[\w]+")

###############################################################################
# **Match** (``re.match(string)``) to be used in test, loop, etc.
# Determine if the RE matches **at the beginning** of the string.

yes_ = True if pattern.match("firstname:John_lastname:Doe") else False
no_ = True if pattern.match("blahbla_firstname:John_lastname:Doe") else False
no2_ = True if pattern.match("OUPS-John_lastname:Doe") else False
print(yes_, no_, no2_)


###############################################################################
# **Match** (``re.search(string)``) to be used in test, loop, etc.
# Determine if the RE matches **at any location** in the string.

yes_ = True if pattern.search("firstname:John_lastname:Doe") else False
yes2_ = True if pattern.search(
    "blahbla_firstname:John_lastname:Doe") else False
no_ = True if pattern.search("OUPS-John_lastname:Doe") else False
print(yes_, yes2_, no_)

###############################################################################
# **Find** (``re.findall(string)``) all substrings where the RE matches,
# and returns them as a list.

# Find the whole pattern within the string
pattern = re.compile("firstname:[\w]+_lastname:[\w]+")
print(pattern.findall("firstname:John_lastname:Doe blah blah"))

# Find words
print(re.compile("[a-zA-Z0-9]+").findall("firstname:John_lastname:Doe"))

# Find words with including underscore
print(re.compile("[\w]+").findall("firstname:John_lastname:Doe"))


###############################################################################
# Extract specific parts of the RE: use parenthesis ``(part of pattern to be matched)``
# Extract John and Doe, such as John is suffixed with firstname:
# and Doe is suffixed with lastname: 

pattern = re.compile("firstname:([\w]+)_lastname:([\w]+)")
print(pattern.findall("firstname:John_lastname:Doe \
    firstname:Bart_lastname:Simpson"))

###############################################################################
# **Split** (``re.split(string)``) splits the string where there is a match and
# returns a list of strings where the splits have occurred. Example, match
# any non alphanumeric character (digits and alphabets) ``[^a-zA-Z0-9]`` to split
# the string.

print(re.compile("[^a-zA-Z0-9]").split("firstname:John_lastname:Doe"))


###############################################################################
# **Substitute** (``re.sub(pattern, replace, string)``) returns a string where
# matched occurrences are replaced with the content of replace variable.

print(re.sub('\s', "_", "Sentence with white      space"))
print(re.sub('\s+', "_", "Sentence with white      space"))

###############################################################################
# Remove all non-alphanumeric characters and space in a string

re.sub('[^0-9a-zA-Z\s]+', '', 'H^&ell`.,|o W]{+orld')

###############################################################################
# System programming
# ------------------
#

###############################################################################
# Operating system interfaces (os)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os


###############################################################################
# Get/set current working directory


# Get the current working directory
cwd = os.getcwd()
print(cwd)

# Set the current working directory
os.chdir(cwd)

###############################################################################
# Temporary directory

import tempfile
tmpdir = tempfile.gettempdir()
print(tmpdir)

###############################################################################
# Join paths

mytmpdir = os.path.join(tmpdir, "foobar")

###############################################################################
# Create a directory

os.makedirs(os.path.join(tmpdir, "foobar", "plop", "toto"), exist_ok=True)

# list containing the names of the entries in the directory given by path.
os.listdir(mytmpdir)

###############################################################################
# File input/output
# ~~~~~~~~~~~~~~~~~

filename = os.path.join(mytmpdir, "myfile.txt")
print(filename)
lines = ["Dans python tout est bon", "Enfin, presque"]

###############################################################################
# Write line by line

fd = open(filename, "w")
fd.write(lines[0] + "\n")
fd.write(lines[1] + "\n")
fd.close()

###############################################################################
# Context manager to automatically close your file

with open(filename, 'w') as f:
    for line in lines:
        f.write(line + '\n')

###############################################################################
# Read
# read one line at a time (entire file does not have to fit into memory)

f = open(filename, "r")
f.readline()    # one string per line (including newlines)
f.readline()    # next line
f.close()

# read the whole file at once, return a list of lines
f = open(filename, 'r')
f.readlines()   # one list, each line is one string
f.close()

# use list comprehension to duplicate readlines without reading entire file at once
f = open(filename, 'r')
[line for line in f]
f.close()

# use a context manager to automatically close your file
with open(filename, 'r') as f:
    lines = [line for line in f]

###############################################################################
# Explore, list directories
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#

###############################################################################
# Walk through directories and subdirectories ``os.walk(dir)``

WD = os.path.join(tmpdir, "foobar")

for dirpath, dirnames, filenames in os.walk(WD):
    print(dirpath, dirnames, filenames)


###############################################################################
# Search for a file using a wildcard ``glob.glob(dir)``

import glob
filenames = glob.glob(os.path.join(tmpdir, "*", "*.txt"))
print(filenames)

###############################################################################
# Manipulating file names, basename and extension

def split_filename_inparts(filename):
    dirname_ = os.path.dirname(filename)
    filename_noext_, ext_ = os.path.splitext(filename)
    basename_ = os.path.basename(filename_noext_)
    return dirname_, basename_, ext_


print(filenames[0], "=>", split_filename_inparts(filenames[0]))


###############################################################################
# File operations: (recursive) copy, move, test if exists: ``shutil`` package

import shutil

###############################################################################
# Copy

src = os.path.join(tmpdir, "foobar",  "myfile.txt")
dst = os.path.join(tmpdir, "foobar",  "plop", "myfile.txt")
shutil.copy(src, dst)
print("copy %s to %s" % (src, dst))

###############################################################################
# Test if file exists ?

print("File %s exists ?" % dst, os.path.exists(dst))

###############################################################################
# Recursive copy,deletion and move

src = os.path.join(tmpdir, "foobar",  "plop")
dst = os.path.join(tmpdir, "plop2")

try:
    print("Copy tree %s under %s" % (src, dst))
    # Note that by default (dirs_exist_ok=True), meaning that copy will fail
    # if destination exists.
    shutil.copytree(src, dst, dirs_exist_ok=True)
    
    print("Delete tree %s" % dst)
    shutil.rmtree(dst)

    print("Move tree %s under %s" % (src, dst))
    shutil.move(src, dst)
except (FileExistsError, FileNotFoundError) as e:
    pass

###############################################################################
# Command execution with subprocess
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For more advanced use cases, the underlying Popen interface can be used directly.

import subprocess

###############################################################################
# ``subprocess.run([command, args*])``
#
# - Run the command described by args.
# - Wait for command to complete
# - return a CompletedProcess instance.
# - Does not capture stdout or stderr by default. To do so, pass PIPE for the stdout and/or stderr arguments.

p = subprocess.run(["ls", "-l"])
print(p.returncode)

###############################################################################
# Run through the shell

subprocess.run("ls -l", shell=True)

###############################################################################
# Capture output

out = subprocess.run(
    ["ls", "-a", "/"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# out.stdout is a sequence of bytes that should be decoded into a utf-8 string
print(out.stdout.decode('utf-8').split("\n")[:5])


###############################################################################
# Multiprocessing and multithreading
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# `Difference between multiprocessing and multithreading <https://techdifferences.com/difference-between-multiprocessing-and-multithreading.html>`_
# is essential to perform efficient parallel processing on multi-cores computers.
#
#    **Multiprocessing**
#
#    A process is  a program instance that has been loaded into memory
#    and managed by the operating system.
#    Process = address space + execution context (thread of control)
#
#    - Process address space is made of (memory) segments for (i) code,
#      (ii) data (static/global), (iii) heap (dynamic memory allocation),
#      and the execution stack (functions' execution context).
#    - Execution context consists of (i) data registers, (ii) Stack Pointer (SP),
#      (iii) Program Counter (PC), and (iv) working Registers.
#
#    OS Scheduling of processes: context switching (ie. save/load Execution context)
#
#    Pros/cons
#
#    - Context switching expensive.
#    - (potentially) complex data sharing (not necessary true).
#    - Cooperating processes - no need for memory protection (separate address spaces).
#    - Relevant for parallel computation with memory allocation.
#
#    **Multithreading**
#
#    - Threads share the same address space (Data registers): access to code, heap and (global) data.
#    - Separate execution stack, PC and Working Registers.
#
#    Pros/cons
#
#    - **Faster context switching** only SP, PC and Working Registers.
#    - Can exploit fine-grain concurrency
#    - Simple data sharing through the shared address space.
#    - **But most of concurrent memory operations are serialized (blocked)
#      by the global interpreter lock (GIL)**.
#      The GIL prevents two threads writing to the same memory at the same time.
#    - Relevant for GUI, I/O (Network, disk) concurrent operation
#
#    **In Python**
#
#    - **As long the GIL exists favor multiprocessing over multithreading**
#    - Multithreading rely on ``threading`` module.
#    - Multiprocessing rely on ``multiprocessing`` module.


###############################################################################
# **Example: Random forest**
# 
# Random forest are the obtained by Majority vote of decision tree on estimated 
# on bootstrapped samples.
#
# Toy dataset

import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

# Toy dataset
X, y = make_classification(n_features=1000, n_samples=5000, n_informative=20,
                           random_state=1, n_clusters_per_class=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,
                                                    random_state=42)


###############################################################################
# Random forest algorithm:
# (i) In parallel, fit decision trees on bootstrapped data samples. Make predictions.
# (ii) Majority vote on predictions

###############################################################################
# 1. In parallel, fit decision trees on bootstrapped data sample. Make predictions.

def boot_decision_tree(X_train, X_test, y_train, predictions_list=None):
    N = X_train.shape[0]
    boot_idx = np.random.choice(np.arange(N), size=N, replace=True)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train[boot_idx], y_train[boot_idx])
    y_pred = clf.predict(X_test)
    if predictions_list is not None:
        predictions_list.append(y_pred)
    return y_pred

###############################################################################
# Independent runs of decision tree, see variability of predictions

for i in range(5):
    y_test_boot = boot_decision_tree(X_train, X_test, y_train)
    print("%.2f" % balanced_accuracy_score(y_test, y_test_boot))

###############################################################################
# 2. Majority vote on predictions

def vote(predictions):
    maj = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)),
        axis=1,
        arr=predictions
    )
    return maj


###############################################################################
# **Sequential execution**
#
# Sequentially fit decision tree on bootstrapped samples, then apply majority vote

nboot = 2
start = time.time()
y_test_boot = np.dstack([boot_decision_tree(X_train, X_test, y_train)
                         for i in range(nboot)]).squeeze()
y_test_vote = vote(y_test_boot)
print("Balanced Accuracy: %.2f" % balanced_accuracy_score(y_test, y_test_vote))
print("Sequential execution, elapsed time:", time.time() - start)


###############################################################################
# **Multithreading**
#
# Concurrent (parallel) execution of the function with two threads.

from threading import Thread

predictions_list = list()
thread1 = Thread(target=boot_decision_tree,
                 args=(X_train, X_test, y_train, predictions_list))
thread2 = Thread(target=boot_decision_tree,
                 args=(X_train, X_test, y_train, predictions_list))

# Will execute both in parallel
start = time.time()
thread1.start()
thread2.start()

# Joins threads back to the parent process
thread1.join()
thread2.join()

# Vote on concatenated predictions
y_test_boot = np.dstack(predictions_list).squeeze()
y_test_vote = vote(y_test_boot)
print("Balanced Accuracy: %.2f" % balanced_accuracy_score(y_test, y_test_vote))
print("Concurrent execution with threads, elapsed time:", time.time() - start)


###############################################################################
# **Multiprocessing**
#
# Concurrent (parallel) execution of the function with
# processes (jobs) executed in different address (memory) space.
# `Process-based parallelism <https://docs.python.org/3/library/multiprocessing.html>`_
#
#
# ``Process()`` for parallel execution and ``Manager()`` for data sharing
#
# **Sharing data between process with Managers**
# Therefore, sharing data requires specific mechanism using  .
# Managers provide a way to create data which can be shared between
# different processes, including sharing over a network between processes
# running on different machines. A manager object controls a server process
# which manages shared objects.

from multiprocessing import Process, Manager

predictions_list = Manager().list()
p1 = Process(target=boot_decision_tree,
             args=(X_train, X_test, y_train, predictions_list))
p2 = Process(target=boot_decision_tree,
             args=(X_train, X_test, y_train, predictions_list))

# Will execute both in parallel
start = time.time()
p1.start()
p2.start()

# Joins processes back to the parent process
p1.join()
p2.join()

# Vote on concatenated predictions
y_test_boot = np.dstack(predictions_list).squeeze()
y_test_vote = vote(y_test_boot)
print("Balanced Accuracy: %.2f" % balanced_accuracy_score(y_test, y_test_vote))
print("Concurrent execution with processes, elapsed time:", time.time() - start)

###############################################################################
# ``Pool()`` of **workers (processes or Jobs)** for concurrent (parallel) execution of multiples
# tasks.
# Pool can be used when *N* independent tasks need to be executed in parallel, when there are
# more tasks than cores on the computer.
#
# 1. Initialize a `Pool(), map(), apply_async(), <https://superfastpython.com/multiprocessing-pool-map-multiple-arguments/>`_
#    of *P* workers (Process, or Jobs), where *P* < number of cores in the computer.
#    Use `cpu_count` to get the number of logical cores in the current system, 
#    See: `Number of CPUs and Cores in Python <https://superfastpython.com/number-of-cpus-python/>`_.
# 2. Map *N* tasks to the *P* workers, here we use the function 
#    `Pool.apply_async() <https://superfastpython.com/multiprocessing-pool-apply_async/>`_ that runs the
#    jobs asynchronously. Asynchronous means that calling `pool.apply_async` does not block the execution
#    of the caller that carry on, i.e., it returns immediately with a `AsyncResult` object for the task.
# 
# that the caller (than runs the sub-processes) is not blocked by the 
# to the process pool does not block, allowing the caller that issued the task to carry on.#
# 3. Wait for all jobs to complete `pool.join()`
# 4. Collect the results

from multiprocessing import Pool, cpu_count
# Numbers of logical cores in the current system.
# Rule of thumb: Divide by 2 to get nb of physical cores
njobs = int(cpu_count() / 2) 
start = time.time()
ntasks = 12
  
pool = Pool(njobs)
# Run multiple tasks each with multiple arguments
async_results = [pool.apply_async(boot_decision_tree,
                                  args=(X_train, X_test, y_train))
                 for i in range(ntasks)]

# Close the process pool & wait for all jobs to complete
pool.close()
pool.join()

# Collect the results
y_test_boot = np.dstack([ar.get() for ar in async_results]).squeeze()

# Vote on concatenated predictions

y_test_vote = vote(y_test_boot)
print("Balanced Accuracy: %.2f" % balanced_accuracy_score(y_test, y_test_vote))
print("Concurrent execution with processes, elapsed time:", time.time() - start)


###############################################################################
# Scripts and argument parsing
# -----------------------------
#
# Example, the word count script ::
#
#        import os
#        import os.path
#        import argparse
#        import re
#        import pandas as pd
#
#        if __name__ == "__main__":
#            # parse command line options
#            output = "word_count.csv"
#            parser = argparse.ArgumentParser()
#            parser.add_argument('-i', '--input',
#                                help='list of input files.',
#                                nargs='+', type=str)
#            parser.add_argument('-o', '--output',
#                                help='output csv file (default %s)' % output,
#                                type=str, default=output)
#            options = parser.parse_args()
#
#            if options.input is None :
#                parser.print_help()
#                raise SystemExit("Error: input files are missing")
#            else:
#                filenames = [f for f in options.input if os.path.isfile(f)]
#
#            # Match words
#            regex = re.compile("[a-zA-Z]+")
#
#            count = dict()
#            for filename in filenames:
#                fd = open(filename, "r")
#                for line in fd:
#                    for word in regex.findall(line.lower()):
#                        if not word in count:
#                            count[word] = 1
#                        else:
#                            count[word] += 1
#
#            fd = open(options.output, "w")
#
#            # Pandas
#            df = pd.DataFrame([[k, count[k]] for k in count], columns=["word", "count"])
#            df.to_csv(options.output, index=False)

###############################################################################
# Networking
# ----------
#

# TODO

###############################################################################
# FTP
# ~~~


###############################################################################
# FTP with ``ftplib``

import ftplib

ftp = ftplib.FTP("ftp.cea.fr")
ftp.login()
ftp.cwd('/pub/unati/people/educhesnay/pystatml')
ftp.retrlines('LIST')

fd = open(os.path.join(tmpdir, "README.md"), "wb")
ftp.retrbinary('RETR README.md', fd.write)
fd.close()
ftp.quit()

###############################################################################
# FTP file download with ``urllib``

import urllib
ftp_url = 'ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/README.md'
urllib.request.urlretrieve(ftp_url, os.path.join(tmpdir, "README2.md"))


###############################################################################
# HTTP
# ~~~~
#

# TODO

###############################################################################
# Sockets
# ~~~~~~~
#

# TODO

###############################################################################
# xmlrpc
# ~~~~~~
#

# TODO


###############################################################################
# Object Oriented Programming (OOP)
# ---------------------------------
#
# **Sources**
#
# -  http://python-textbok.readthedocs.org/en/latest/Object\_Oriented\_Programming.html
#
# **Principles**
#
# -  **Encapsulate** data (attributes) and code (methods) into objects.
#
# -  **Class** = template or blueprint that can be used to create objects.
#
# -  An **object** is a specific instance of a class.
#
# -  **Inheritance**: OOP allows classes to inherit commonly used state
#    and behavior from other classes. Reduce code duplication
#
# -  **Polymorphism**: (usually obtained through polymorphism) calling
#    code is agnostic as to whether an object belongs to a parent class or
#    one of its descendants (abstraction, modularity). The same method
#    called on 2 objects of 2 different classes will behave differently.
#


class Shape2D:
    def area(self):
        raise NotImplementedError()

# __init__ is a special method called the constructor


# Inheritance + Encapsulation
class Square(Shape2D):
    def __init__(self, width):
        self.width = width

    def area(self):
        return self.width ** 2


class Disk(Shape2D):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2


shapes = [Square(2), Disk(3)]

# Polymorphism
print([s.area() for s in shapes])

s = Shape2D()
try:
    s.area()
except NotImplementedError as e:
    print("NotImplementedError", e)


###############################################################################
# Style guide for Python programming
# ----------------------------------
#
# See `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
#
# - Spaces (four) are the preferred indentation method.
# - Two blank lines for top level function or classes definition.
# - One blank line to indicate logical sections.
# - Never use: ``from lib import *``
# - Bad: ``Capitalized_Words_With_Underscores``
# - Function and Variable Names: ``lower_case_with_underscores``
# - Class Names: ``CapitalizedWords`` (aka: ``CamelCase``)


###############################################################################
# Documenting
# -----------
#
# See `Documenting Python <https://realpython.com/documenting-python-code//>`_
# Documenting = comments + docstrings (Python documentation string)
#
# - `Docstrings <https://www.datacamp.com/community/tutorials/docstrings-python>`_
#   are use as documentation for the class, module, and packages.
#   See it as "living documentation".
# - Comments are  used to explain non-obvious portions of the code. "Dead documentation".
#
# Docstrings for functions (same for classes and methods):

def my_function(a, b=2):
    """
    This function ...

    Parameters
    ----------
    a : float
        First operand.
    b : float, optional
        Second operand. The default is 2.

    Returns
    -------
    Sum of operands.

    Example
    -------
    >>> my_function(3)
    5
    """
    # Add a with b (this is a comment)
    return a + b


print(help(my_function))

###############################################################################
# Docstrings for scripts:
#
# At the begining of a script add a pream::
#
#        """
#        Created on Thu Nov 14 12:08:41 CET 2019
#
#        @author: firstname.lastname@email.com
#
#        Some description
#        """


###############################################################################
# Modules and packages
# --------------------
#
# Python `packages and modules <https://docs.python.org/3/tutorial/modules.html>`_
# structure python code into modular "libraries" to be shared.

###############################################################################
# Package
# ~~~~~~~
# 
# Packages are a way of structuring Python’s module namespace by using “dotted module names”.
# A package is a directory (here, ``stat_pkg``) containing a ``__init__.py`` file.

###############################################################################
# Example, ``package``
# ::
#     stat_pkg/
#     ├── __init__.py
#     └── datasets_mod.py
#
# The ``__init__.py`` can be empty.
# Or it can be used to define the package API, i.e., the modules (``*.py`` files)
# that are exported and those that remain internal.

###############################################################################
# Example, file ``stat_pkg/__init__.py``
# ::
#     # 1) import function for modules in the packages
#     from .module import make_regression
#
#     # 2) Make them visible in the package
#     __all__ = ["make_regression"]


###############################################################################
# Module
# ~~~~~~
#
# A module is a python file.
# Example, ``stat_pkg/datasets_mod.py``
# ::
#     import numpy as np
#     def make_regression(n_samples=10, n_features=2, add_intercept=False):
#         ...
#         return X, y, coef


###############################################################################
# Usage
#

import stat_pkg as pkg

X, y, coef = pkg.make_regression()
print(X.shape)

###############################################################################
# The search path
# ~~~~~~~~~~~~~~~
# 
# With a directive like ``import stat_pkg``, Python will searches for
#
# - a module, file named ``stat_pkg.py`` or,
# - a package, directory named ``stat_pkg`` containing a ``stat_pkg/__init__.py`` file.
#
# Python will search in a list of directories given by the variable
# ``sys.path``. This variable is initialized from these locations:
#
#  - The directory containing the input script (or the current directory when no file is specified).
#  - **``PYTHONPATH``** (a list of directory names, with the same syntax as the shell variable ``PATH``).
#
# In our case, to be able to import ``stat_pkg``, the parent directory of ``stat_pkg``
# must be in ``sys.path``.
# You can modify ``PYTHONPATH`` by any method, or access it via ``sys`` package, example:
# ::
#     import sys
#     sys.path.append("/home/ed203246/git/pystatsml/python_lang")

###############################################################################
# Unit testing
# ------------
#
# When developing a library (e.g., a python package) that is bound to evolve and being corrected, we want to ensure that:
# (i) The code correctly implements some expected functionalities;
# (ii) the modifications and additions don't break those functionalities; 
#
# Unit testing is a framework to asses to those two points. See sources:
#  
# - `Unit testing reference doc <https://docs.python.org/3/library/unittest.html>`_
# - `Getting Started With Testing in Python <https://realpython.com/python-testing/>`_

###############################################################################
# Write unit tests (test cases)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In a directory usually called ``tests`` create a `test case <https://docs.python.org/3/library/unittest.html#unittest.TestCase>`_, i.e., a python file 
# ``test_datasets_mod.py`` (general syntax is ``test_<mymodule>.py``) that will execute some
# functionalities of the module and test if the output are as expected. 
# `test_datasets_mod.py` file contains specific directives:
#
# 1. ``import unittest``,
# 2. ``class TestDatasets(unittest.TestCase)``, the test case class. The general syntax is ``class Test<MyModule>(unittest.TestCase)``
# 3. ``def test_make_regression(self)``, test a function of an element of the module. The general syntax is ``test_<my function>(self)``
# 4. ``self.assertTrue(np.allclose(X.shape, (10, 4)))``, test a specific functionality. The general syntax is ``self.assert<True|Equal|...>(<some boolean expression>)``
# 5. ``unittest.main()``, where tests should be executed.
#
# Example:
# ::
#     import unittest
#     import numpy as np
#     from stat_pkg import make_regression
#
#     class TestDatasets(unittest.TestCase):
#
#         def test_make_regression(self):
#             X, y, coefs = make_regression(n_samples=10, n_features=3,
#                                           add_intercept=True)     
#             self.assertTrue(np.allclose(X.shape, (10, 4)))
#             self.assertTrue(np.allclose(y.shape, (10, )))
#             self.assertTrue(np.allclose(coefs.shape, (4, )))
#
#     if __name__ == '__main__':
#         unittest.main()

###############################################################################
# Run the tests  (test runner)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `test runner <https://wiki.python.org/moin/PythonTestingToolsTaxonomy>`_ 
# orchestrates the execution of tests and provides the outcome to the user.
# Many `test runners <https://blog.kortar.org/?p=370>`_ are available.
#
# `unittest <https://docs.python.org/3/library/unittest.html>`_ is the first unit test framework,
# it comes with Python standard library.
# It employs an object-oriented approach, grouping tests into classes known as test cases, 
# each containing distinct methods representing individual tests.
#
# Unitest generally requires that tests are organized as importable modules,
# `see details <https://docs.python.org/3/library/unittest.html#command-line-interface>`_.
# Here, we do not introduce this complexity: we directly execute a test file that isn’t importable
# as a module.
# ::
#     python tests/test_datasets_mod.py
#
# `Unittest test discovery <https://docs.python.org/3/library/unittest.html#unittest-test-discovery>`_:
# (``-m unittest discover``) within (``-s``) ``tests`` directory, with verbose (``-v``) outputs.
# ::
#    python -m unittest discover -v -s tests

###############################################################################
# Doctest: add unit tests in docstring
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# `Doctest <https://docs.python.org/3/library/doctest.html>`_ is an inbuilt test framework that comes bundled with Python by default.
# The doctest module searches for code fragments that resemble interactive Python sessions and runs those sessions to confirm they operate as shown.
# It promotes `Test-driven (TDD) methodology <https://medium.com/@muirujackson/python-test-driven-development-6235c479baa2>`_
#
# Example file: ``python stat_pkg/supervised_models.py``
# ::
#     class LinearRegression:
#         """Ordinary least squares Linear Regression.
#
#         Application Programming Interface (API) is compliant with scikit-learn:
#         fit(X, y), predict(X)
#
#         Parameters
#         ----------
#         fit_intercept : bool, default=True
#             Whether to calculate the intercept for this model. If set
#             to False, no intercept will be used in calculations
#             (i.e. data is expected to be centered).
#
#         Examples
#         --------
#         >>> import numpy as np
#         >>> from stat_pkg import LinearRegression
#         >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
#         >>> # y = 1 * x_0 + 2 * x_1 + 3
#         >>> y = np.dot(X, np.array([1, 2])) + 3
#         >>> reg = LinearRegression().fit(X, y)
#         >>> reg.coef_
#         array([3., 1., 2.0])
#         >>> reg.predict(np.array([[3, 5]]))
#         array([16.])
#         """
#
# test:
# ::
#     python stat_pkg/supervised_models.py
#

###############################################################################
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~
#


###############################################################################
# Exercises
# ---------
#


###############################################################################
# Exercise 1: functions
# ~~~~~~~~~~~~~~~~~~~~~
#
# Create a function that acts as a simple calculator taking three parameters:
# the two operand and the operation in "+", "-", and "*". As default use "+".
# If the operation is misspecified, return a error message Ex: ``calc(4,5,"*")`` returns 20 Ex:
# ``calc(3,5)`` returns 8 Ex: ``calc(1, 2, "something")`` returns error
# message
#


###############################################################################
# Exercise 2: functions + list + loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Given a list of numbers, return a list where all adjacent duplicate
# elements have been reduced to a single element. Ex: ``[1, 2, 2, 3, 2]``
# returns ``[1, 2, 3, 2]``. You may create a new list or modify the passed
# in list.
#
# Remove all duplicate values (adjacent or not) Ex: ``[1, 2, 2, 3, 2]``
# returns ``[1, 2, 3]``
#


###############################################################################
# Exercise 3: File I/O
# ~~~~~~~~~~~~~~~~~~~~
#
# 1. Copy/paste the BSD 4 clause license (https://en.wikipedia.org/wiki/BSD_licenses)
# into a text file. Read, the file and count the occurrences of each
# word within the file. Store the words' occurrence number in a dictionary.
#
# 2. Write an executable python command ``count_words.py`` that parse
# a list of input files provided after ``--input`` parameter.
# The dictionary of occurrence is save in a csv file provides by ``--output``.
# with default value word_count.csv.
# Use:
# - open
# - regular expression
# - argparse (https://docs.python.org/3/howto/argparse.html)


###############################################################################
# Exercise 4: OOP
# ~~~~~~~~~~~~~~~
#
# 1. Create a class ``Employee`` with 2 attributes provided in the
#    constructor: ``name``, ``years_of_service``. With one method
#    ``salary`` with is obtained by ``1500 + 100 * years_of_service``.
#
# 2. Create a subclass ``Manager`` which redefine ``salary`` method
#    ``2500 + 120 * years_of_service``.
#
# 3. Create a small dictionary-nosed database where the key is the
#    employee's name. Populate the database with: samples =
#    Employee('lucy', 3), Employee('john', 1), Manager('julie', 10),
#    Manager('paul', 3)
#
# 4. Return a table of made name, salary rows, i.e. a list of list [[name,
#    salary]]
#
# 5. Compute the average salary
