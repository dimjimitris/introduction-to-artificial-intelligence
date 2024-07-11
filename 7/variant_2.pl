% so that the online swi-prolog environment can run the code
:- use_module(library(clpb)).

% simple predicate to check leap year
is_leap_year(Year) :-
    0 =:= mod(Year, 4), 0 =\= mod(Year, 100),!.
                                  
is_leap_year(Year) :-
    0 =:= mod(Year, 400).

% simply get the number of days in a month
days_in_month(_Year, Month, Days) :-
    member(Month, [1, 3, 5, 7, 8, 10, 12]),
    Days = 31,!.

days_in_month(_Year, Month, Days) :-
    member(Month, [4, 6, 9, 11]),
    Days = 30,!.

days_in_month(Year, 2, Days) :-
    is_leap_year(Year),
    Days = 29,!.

days_in_month(_Year, 2, Days) :-
    Days = 28.

% given a month, sum all the days in the previous months
% Including the given month
total_days_in_prev_months(_Year, 0, Acc, Res) :- Res is Acc,!.

total_days_in_prev_months(Year, Month, Acc, Res) :-
    Month > 0,
    PrevMonth is Month - 1,
    days_in_month(Year, Month, Days),
    NewAcc is Acc + Days,
    total_days_in_prev_months(Year, PrevMonth, NewAcc, Res).

% given a date, sum all the days in the previous months and
% add the days up to the date in the current month
total_days_in_date(Year, Month, Day, TotalDays) :-
    Month > 0,
    Month < 13,
    days_in_month(Year, Month, Days),
    Day > 0,
    Day =< Days,
    PrevMonth is Month - 1,
    total_days_in_prev_months(Year, PrevMonth, 0, TotalDays1),
    TotalDays is TotalDays1 + Day.

% Date{1,2} are two dates as strings in the format DDMM, year is 2024
% Interval is the number of days between the two dates
interval(Date1, Date2, Interval) :-
    string_length(Date1, 4),
    string_length(Date2, 4),
    sub_string(Date1, 0, 2, _, SDay1),
    sub_string(Date1, 2, 2, _, SMonth1),
    sub_string(Date2, 0, 2, _, SDay2),
    sub_string(Date2, 2, 2, _, SMonth2),
    number_string(Day1, SDay1),
    number_string(Month1, SMonth1),
    number_string(Day2, SDay2),
    number_string(Month2, SMonth2),
    total_days_in_date(2024, Month1, Day1, TotalDays1),
    total_days_in_date(2024, Month2, Day2, TotalDays2),
    Interval is TotalDays2 - TotalDays1.

% Date{1,2} are two dates as strings in the format DDMM, year is 2024
% Write the number of days between the two dates to the output
interval(Date1, Date2) :-
    interval(Date1, Date2, Interval),
    write(Interval).

% test predicate
test([]).
test([(Date1, Date2, Expected) | T]) :-
    interval(Date1, Date2, Result),
    Result = Expected,
    test(T).

% tests that have to fail
% test predicate

test_fail([]).
test_fail([(Date1, Date2) | T]) :-
    (
        interval(Date1, Date2, _Result)
    ->  write("Test failed: "),
        write(Date1), write(" "),
        write(Date2), nl, fail
    ;   true
    ), test_fail(T).

% test cases

/*
% tests that should succeed
test([
    ("0101", "0201", 1), % check that our program actually works
    ("2205", "0506", 14), % problem description test
    ("0102", "1102", 10), % problem description test
    ("2802", "0103", 2), % check february (2024 is a leap year)
    ("2902", "2902", 0), % check february (2024 is a leap year)
    ("0101", "3112", 365), % check the whole year
    ("2504","2504",0), % check the same date
    % now let's check some negatives...
    ("0201", "0101", -1), % check that our program actually works
    ("0506", "2205", -14), % problem description test
    ("1102", "0102", -10), % problem description test
    ("0103", "2802", -2), % check february (2024 is a leap year)
    ("3112", "0101", -365) % check the whole year 
]).

test_fail([
    ("abc", "0201"), % letters not allowed
    ("011", "0201"), % dates have to be 4 character strings
    ("0101", "020"), % dates have to be 4 character strings
    ("0101", "02011"), % dates have to be 4 character strings
    ("0001", "0201"), % days have to be valid
    ("3201", "0201"), % days have to be valid
    ("0100", "0201"), % days have to be valid
    ("0101", "0001"), % months have to be valid
    ("0101", "0113"), % months have to be valid
    ("0101", "0000"), % months have to be valid
    ("0101", "0013"), % months have to be valid
    ("3002", "1301") % febraury has 29 days
]).
*/
