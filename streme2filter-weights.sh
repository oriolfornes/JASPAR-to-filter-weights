#!/usr/bin/env bash

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

FILTERS_DIR=${SRC_DIR}/streme-filters/Fig3
STREME_DIR=${SRC_DIR}/streme/Fig3

for TF in `ls ${STREME_DIR}/`; do

    for J in `ls ${STREME_DIR}/${TF}`; do

        if ! [ -d ${FILTERS_DIR}/${TF}/${J} ]; then

            mkdir -p ${FILTERS_DIR}/${TF}/${J}

            ${SRC_DIR}/streme2filter-weights.py \
                ${STREME_DIR}/${TF}/${J}/streme.txt.gz \
                ${FILTERS_DIR}/${TF}/${J}/streme.pkl

        fi

    done

done

FILTERS_DIR=${SRC_DIR}/streme-filters/Fig7
STREME_DIR=${SRC_DIR}/streme/Fig7

for SUB_DIR in `ls ${STREME_DIR}`; do

    for TF in `ls ${STREME_DIR}/${SUB_DIR}`; do

        for J in `ls ${STREME_DIR}/${SUB_DIR}/${TF}`; do

            if ! [ -d ${FILTERS_DIR}/${SUB_DIR}/${TF}/${J} ]; then

                mkdir -p ${FILTERS_DIR}/${SUB_DIR}/${TF}/${J}

                ${SRC_DIR}/streme2filter-weights.py \
                    ${STREME_DIR}/${SUB_DIR}/${TF}/${J}/streme.txt.gz \
                    ${FILTERS_DIR}/${SUB_DIR}/${TF}/${J}/streme.pkl

            fi

        done

    done

done
