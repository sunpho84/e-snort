#!/bin/bash

if [ -f "$1/lib/resources/gitInfo.cpp" ]
then
    cat "$1/lib/resources/gitInfo.cpp"
else
    
    echo 'namespace esnort::git'
    echo '{'
    echo
    
    echo -n ' const char* hash="'
    git rev-parse HEAD|tr -d "\n"
    echo '";'
    
    echo -n ' const char* time="'
    git log -1 --pretty=%ad|tr -d "\n"
    echo '";'
    
    echo -n ' const char* committer="'
    git log -1 --pretty=%cn|tr -d "\n"
    echo '";'
    
    echo -n ' const char* log="'
    git log -1 --pretty=%B|tr -d "\n"|sed 's|\\|\\\\|g'
    echo '";'
    
    echo
    echo '}'

fi
