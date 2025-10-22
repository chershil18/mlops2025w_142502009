#!/bin/bash

echo "Enter your Number:"
read num

sum_of_numbers(){
echo "--------SUM OF N NUMBERS AND FACTORIAL IN SHELL SCRIPT-----------"
t=1
total=0
while test $t -le $num
do
	total=`expr $total + $t`
	t=`expr $t + 1`
done
echo "Sum : $total"
}


factorial(){
fact=1

if [ $num -eq 0 ]; then
  echo "Factorial of 0 is 1."
else
  for (( i=1; i<=num; i++ ))
  do
    fact=$(( fact * i ))
  done
  echo "Factorial of $num is $fact."
fi
}

sum_of_numbers
factorial