# EARIN_7th_mini-project

## Project Description

This is the 7th mini-project for the EARIN course. We implement a Prolog predicate to calculate the number of days between two dates in 2024 (variant 2).

## Usage
Program was tested (and works as follows):

Copy code from [variant_2.pl](variant_2.pl) file into the online [swi-prolog](https://swish.swi-prolog.org/) environment.

### Testing
You can use the tests that have been commented out (just run them in the online environment), they should return `true`.

### Predicates
We have two predicates you may use.

- `interval(Date1, Date2, Interval)`: Given two dates inside 2024, it binds `Interval` to the number of days between the two dates.
- `interval(Date1, Date2)`: prints the interval between the two dates at the output.

## Remarks

- The return values N are in the range -365 <= N <= 365. Negative numbers happen when the first date is larger than the second date. We decided to keep negative numbers, instead of returning just the absolute value of the difference, because it helps the user understand the precedence of the input dates. Also N cannot be 366 (or -366) because that would mean we went through the whole year and are taking a date from the next year, which is not allowed. Maximum absolute value of N, |N| = 365 happens when the two dates given are: January 1 and December 31.

- Recursions in our predicates are tail recursions. This helps the language with some related optimizations.