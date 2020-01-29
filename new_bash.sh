#!/bin/bash

counter=1

while [ $counter -le 225 ]
do
  python multi_3.py &
  python multi_3.py &
  python multi_3.py &
  python multi_3.py 
  echo $counter
  ((counter++))
done

echo All done
