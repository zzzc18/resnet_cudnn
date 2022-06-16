#!/bin/bash
kill -9 $(ps -ef | grep xcnn_cuda | awk '{print $2}')
