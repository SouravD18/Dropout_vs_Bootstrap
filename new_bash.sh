#!/bin/bash

#python multi_3.py & 

counter=1
second_counter=1
while [ $counter -le 12 ]
do
  #python multi_3.py &
  echo $counter
  ((counter++))
  second_counter=1
  while [ $second_counter -le 12 ]
  do
    echo $second_counter
    ((second_counter++))
  done
done

echo All done
