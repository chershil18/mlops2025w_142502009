#!/bin/bash

echo "Enter your Number:"
read num

sum_of_numbers(){
echo "--------SUM OF N NUMBERS IN SHELL SCRIPT-----------"
t=1
total=0
while test $t -le $num
do
        total=`expr $total + $t`
        t=`expr $t + 1`
done
echo "Sum : $total"
}