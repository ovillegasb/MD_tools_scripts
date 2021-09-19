#!/bin/bash

# file mol itp name (only name)

mol=$1

# Topology System File
echo -e "\e[1;34mWrtiting Topology File \e[0m"
sleep 10
rm -vf topol.top
touch topol.top
echo -e "\n;" >> topol.top
echo "; System Topology File" >> topol.top
echo "; orlando.villegas@univ-pau.fr" >> topol.top
echo -e ";\n" >> topol.top
echo -e "#include \"oplsaa.ff/forcefield.itp\"\n" >> topol.top
echo -e "#include \"../atomstypes.itp\"\n" >> topol.top
echo -e "#include \"../${mol}.itp\"\n" >> topol.top
echo -e "[ system ]\nSystem\n" >> topol.top
echo -e "[ molecules ]" >> topol.top
printf "%-5s %-5s\n" ${mol^^} $num >> topol.top