#!/bin/bash

rm print_order
j=0
for i in `find ./ -name *.png`
do
    echo "File $i" >> print_order
    lp $i
    let j=$j+1
#    if test $j -ge 2;
#    then
#	exit
#    fi
done
       
