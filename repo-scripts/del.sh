#!/bin/zsh



x="$1"

if [ -z "$x" ]; then
  echo "x is none!"
  exit 1
fi


function f() {
  local x=$1
  local y=$2
  echo $((x + y))
}

f 20 34

function x {
  p=$1
  cd ~/desktop/$1
  pwd
}

x repos