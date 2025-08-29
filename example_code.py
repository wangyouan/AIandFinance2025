#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Filename: example_code
# @Date: 2025/08/29
# @Author: Mark Wang
# @Email: wangyouan@xmu.edu.cn


def calculate_sum(numbers):
    # Initialize a variable to store the sum
    total_sum = sum(numbers)
    # Return the calculated sum
    return total_sum


if __name__ == "__main__":

    # Example list of numbers
    numbers_list = [1, 2, 3, 4, 5]

    # Call the function and store the result
    result_sum = calculate_sum(numbers_list)

    # Print the result
    print("Sum of numbers:", result_sum)
