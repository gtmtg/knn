branch=`git rev-parse --abbrev-ref HEAD`
project=`gcloud config get-value project 2> /dev/null`
name=spatial-search
url=https://mihir-spatial-search-min-num-instances-test-g6rwrca4fq-uc.a.run.app

export KNN_START_TIME=`python3 -c 'import time;print(time.time())'`
gcloud alpha run deploy mihir-$name-$branch-test --image gcr.io/$project/mihir-$name-$branch --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated --min-instances=1000
export KNN_SLEEP_START_TIME=`python3 -c 'import time;print(time.time())'`
sleep 60
export KNN_INNER_START_TIME=`python3 -c 'import time;print(time.time())'`
pipenv run python benchmark.py -m $url results/$1.json
export KNN_INNER_END_TIME=`python3 -c 'import time;print(time.time())'`
gcloud alpha run deploy mihir-$name-$branch-test --image gcr.io/$project/mihir-$name-$branch --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated --min-instances=0
export KNN_END_TIME=`python3 -c 'import time;print(time.time())'`
