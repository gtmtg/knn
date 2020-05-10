ulimit -n 4096
pipenv run python server.py --interval=5.0 --end_after=15 --delay=20.0 --n_requests=500 --small
pipenv run python server.py --interval=5.0 --end_after=15 --delay=20.0 --n_requests=1000 --small
pipenv run python server.py --interval=5.0 --end_after=15 --delay=20.0 --n_requests=500 --large
pipenv run python server.py --interval=5.0 --end_after=15 --delay=20.0 --n_requests=1000 --large


# gcloud run deploy small-container-500-worker-sleep-microbenchmark-west1 --image gcr.io/mihir-knn/experiments/pubsub --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 256Mi --timeout 900 --region us-west1 --allow-unauthenticated
# gcloud run deploy small-container-1k-worker-sleep-microbenchmark-west1 --image gcr.io/mihir-knn/experiments/pubsub --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 256Mi --timeout 900 --region us-west1 --allow-unauthenticated
# gcloud run deploy large-container-500-worker-sleep-microbenchmark-west1 --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-west1 --allow-unauthenticated
# gcloud run deploy large-container-1k-worker-sleep-microbenchmark-west1 --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-west1 --allow-unauthenticated
# cent

# gcloud run deploy small-container-500-worker-sleep-microbenchmark-cent1 --image gcr.io/mihir-knn/experiments/pubsub --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 256Mi --timeout 900 --region us-central1 --allow-unauthenticated
# gcloud run deploy small-container-1k-worker-sleep-microbenchmark-cent1 --image gcr.io/mihir-knn/experiments/pubsub --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 256Mi --timeout 900 --region us-central1 --allow-unauthenticated
# gcloud run deploy large-container-500-worker-sleep-microbenchmark-cent1 --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated
# gcloud run deploy large-container-1k-worker-sleep-microbenchmark-cent1 --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated


# #
# # gcloud run deploy large-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated
# # pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=10
# # gcloud run deploy large-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated
# # pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=50
# # gcloud run deploy large-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated
# # pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=100
# # gcloud run deploy large-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated
# # pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=500
# # gcloud run deploy large-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated
# # pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=1000
# #
# gcloud run deploy small-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/experiments/pubsub --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 256Mi --timeout 900 --region us-central1 --allow-unauthenticated
# pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=10
# gcloud run deploy small-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/experiments/pubsub --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 256Mi --timeout 900 --region us-central1 --allow-unauthenticated
# pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=50
# gcloud run deploy small-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/experiments/pubsub --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 256Mi --timeout 900 --region us-central1 --allow-unauthenticated
# pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=100
# gcloud run deploy small-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/experiments/pubsub --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 256Mi --timeout 900 --region us-central1 --allow-unauthenticated
# pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=500
# gcloud run deploy small-container-sleep-microbenchmark-central1 --image gcr.io/mihir-knn/experiments/pubsub --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 256Mi --timeout 900 --region us-central1 --allow-unauthenticated
# pipenv run python server.py --interval=5.0 --no_pubsub --end_after=10 --delay=20.0 --n_requests=1000


# # pipenv run python server.py --interval=5.0 --end_after=15 --delay=20.0 --n_requests=1000 --small
