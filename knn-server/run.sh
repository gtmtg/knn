screen -S server -dm pipenv run authbind uvicorn --port :80 --workers 1 server:app
