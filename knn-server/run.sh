screen -S server -dm ulimit -n 4096 && pipenv run authbind uvicorn --host 0.0.0.0 --port 80 --workers 1 server:app
