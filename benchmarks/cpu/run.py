import click

from knn import utils
from knn.reducers import StatisticsReducer
from knn.jobs import MapReduceJob


@click.command()
@click.option(
    "-m",
    "--mapper",
    default="https://mihir-mihir-benchmark-cpu-g6rwrca4fq-uc.a.run.app",
)
@click.option("-r", "--runtime", default=10)
@click.option("-n", "--num_trials", default=100)
@utils.unasync
async def main(mapper, runtime, num_trials):
    job = MapReduceJob(
        mapper,
        StatisticsReducer(),
        {"runtime": runtime},
        n_mappers=1000,
        n_retries=10,  # ensure we get all trials in
        chunk_size=1,  # hit different worker instances
    )
    stats = await job.run_until_complete([None] * num_trials)
    print(f"Mean: {stats.mean()}\nStd: {stats.stddev()}")


if __name__ == "__main__":
    main()
