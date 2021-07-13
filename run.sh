#!/usr/bin/env bash
set -e

virtualenv=$1

if [ ! -d $virtualenv ]; then
  source $virtualenv/bin/activate
fi

source $virtualenv/bin/activate

p.py
