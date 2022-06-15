#!/bin/bash
kill -9 $(ps -ef | grep main.py | awk '{print $2}')
