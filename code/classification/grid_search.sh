#!/bin/bash

mkdir -p data/classification

# specify hyperparameter values
values_of_k=("1 2 3 4 5 6 7 8 9 10")
declare -A values_of_priors=( [0.5]=0.5 [0.6]=0.4 [0.7]=0.3 [0.8]=0.2 [0.9]=0.1)
values_of_var_smooth=("1e-01 1e-02 1e-03 1e-04 1e-05 1e-06 1e-07 1e-08 1e-09")


# different execution modes
if [ $1 = local ]
then
    echo "[local execution]"
    cmd="code/classification/classifier.sge"
elif [ $1 = grid ]
then
    echo "[grid execution]"
    cmd="qsub code/classification/classifier.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search

if [ $2 = knn ]
then
    for k in $values_of_k
    do
        echo $k
        $cmd 'data/classification/clf_'"$k"'.pickle' --knn $k -s 42 --accuracy --kappa --fbeta --sensitivity --run_name knn
    done
elif [ $2 = bayes ]
then

    for p_1 in "${!values_of_priors[@]}"
    do
        p_2=${values_of_priors[$p_1]}
        echo [$p_1, $p_2]
            for v in $values_of_var_smooth
            do
                echo $v
                $cmd 'data/classification/clf_'"$p_1"'_'"$p_2"'_'"$v"'.pickle' --bayes $p_1 $p_2 $v -s 42 --accuracy --kappa --fbeta --sensitivity --run_name bayes
            done
    done
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi   