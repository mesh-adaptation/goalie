#!/usr/bin/bash

removestar -i $1
sed -i "s/^/    /" $1
sed -i "1idef test_demo():" $1
sed -i "1ios.environ['GOALIE_REGRESSION_TEST'] = '1'" $1
sed -i "1iimport os" $1

# Reduce test runtimes
if [ "$1" == "demos/test_demo_solid_body_rotation.py" ]
then
	sed -i "s/end_time = 2 \* pi/end_time = pi \/ 4/" $1
elif [ "$1" == "demos/test_demo_burgers-hessian.py" ]
then
	sed -i "s/\"maxiter\": 35/\"maxiter\": 3/" $1
elif [ "$1" == "demos/test_demo_point_discharge2d-hessian.py" ]
then
	sed -i "s/\"maxiter\": 35/\"maxiter\": 3/" $1
elif [ "$1" == "demos/test_demo_point_discharge2d-goal_oriented.py" ]
then
	sed -i "s/\"maxiter\": 35/\"maxiter\": 3/" $1
elif [ "$1" == "demos/test_demo_gray_scott.py" ]
then
	sed -i "s/end_time = 2000.0/end_time = 10.0/" $1
elif [ "$1" == "demos/test_demo_gray_scott_split.py" ]
then
	sed -i "s/end_time = 2000.0/end_time = 10.0/" $1
fi
