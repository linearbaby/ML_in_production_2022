#! /bin/bash

uvicorn --host 0.0.0.0 --port 8000 server:app &
sleep 130;
pkill -2 uvicorn