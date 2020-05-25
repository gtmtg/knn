project=`gcloud config get-value project 2> /dev/null`
name=mihir-benchmark-`basename ~+`
folder=mapper
region=us-central1
root_path=../..

# Copy shared resources in
cp -r $root_path/src/knn $folder

# Submit build from within subdirectory
gcloud config set builds/use_kaniko True
(cd $folder && gcloud builds submit --tag gcr.io/$project/mihir-$name)

# Remove shared resources
rm -rf $folder/knn

# Deploy Cloud Run handler
gcloud run deploy mihir-$name --image gcr.io/$project/mihir-$name --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region $region --allow-unauthenticated
