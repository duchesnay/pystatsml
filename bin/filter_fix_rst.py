#!/usr/bin/env python3

"""
Created on 6 may 2025

@author: edouard.duchesnay@gmail.com

Filters that fi rst files generated by jupyter nbconvert

It is called in the Makefile:
jupyter nbconvert --to rst --stdout $< | bin/filter_fix_rst.py > $@

Filters:
- convert_rst_cross_reference
    Tranform:
    * `Demonstration of Negative Log-Likelihood (NLL) <demonstration-nll>`__
    TO
    * :ref:`Demonstration of Negative Log-Likelihood (NLL) <demonstration-nll>`
"""

#line = "- `Demonstration of Negative Log-Likelihood (NLL) <demonstration-nll>`__

#- :ref:`Demonstration of Negative Log-Likelihood (NLL) <demonstration-nll>`


import sys
import re

def fix_rst_cross_reference(input_string, verbose=False):
    """ Filter RST string to fix cross-reference.
    From:
    `any-text <label-name>`__.
    to 
    :ref:`any-text <label-name>`

    Parameters
    ----------
    input_string : str
        multiline string

    Returns
    -------
    str
        multiline string
    """
    
    # Define the regular expression pattern
    # pattern = r'`([^<]+) <([^>]+)>`__'
    # pattern = r'`(\w[^<`]+) <(ref:[^>]+)>`__'
    # (?<!:ref:) : do not start with :ref:
    # `(\w[^<`]+) : `Letter anything until <
    # <(ref:[^>]+)>`__ : <ref: anything until >`__
    pattern = r'(?<!:ref:)`(\w[^<`]+) <(ref:[^>]+)>`__'

    # Define the replacement pattern with backtickss
    replacement = r':ref:`\1 <\2>`'

    if verbose:
        print(re.compile(pattern).findall(input_string))

    # Use re.sub to replace all occurrences in the input string
    output_string = re.sub(pattern, replacement, input_string)

    return output_string

# Example usage
input_string = """This is a sample text with a cross-reference: `any-text <label-name>`__.
Another cross-reference: `another-text <another-label>`__."""


if __name__ == "__main__":

        input_string = sys.stdin.read()
        output_string = fix_rst_cross_reference(input_string, verbose=False)
        #output_string =  input_string
        sys.stdout.write(output_string)

