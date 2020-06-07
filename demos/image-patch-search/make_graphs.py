import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup
import collections
import json
import math
import os


@click.command()
@click.argument("input", type=click.File("r"))
@optgroup.group(
    "Graph types", cls=RequiredMutuallyExclusiveOptionGroup,
)
@optgroup.option("-b", "--breakdown", is_flag=True)
@optgroup.option("-t", "--throughput", is_flag=True)
@optgroup.option("-w", "--workers", is_flag=True)
def main(input, breakdown, throughput, workers):
    results = json.load(input)

    if breakdown:
        times = results[-1]["performance"]["profiling"]
        if "boot_time" in times:
            print(times["boot_time"])
            print(len(results[-1]["performance"]["mapper_utilization"]))
            print()
        print(times["total_time"] - times["billed_time"])  # Cloud Run overhead
        print(times["billed_time"] - times["request_time"])  # Model loading
        print(times["request_time"] - times["compute_time"])  # I/O
        print(times["compute_time"])  # Compute
    elif throughput:
        start_time = float(os.getenv("KNN_START_TIME"))
        padding = round((results[0]["progress"]["current_time"] - start_time) / 5)
        for i in range(padding - 1):
            print(0.0)

        prev_progress = {}
        for result in results:
            progress = result["progress"]
            dn = progress["n_processed"] - prev_progress.get("n_processed", 0)
            dt = progress["elapsed_time"] - prev_progress.get("elapsed_time", 0)
            print(dn / dt)
            prev_progress = progress
    elif workers:
        performance = results[-1]["performance"]
        graph_data = collections.defaultdict(int)
        graph_min = 0
        graph_max = 0
        cumulative = False

        if "mapper_utilization" in performance:
            chunks_per_worker = performance["mapper_utilization"].values()

            graph_min = 1
            for bucket in chunks_per_worker:
                graph_data[bucket] += 1
                graph_max = max(graph_max, bucket)
        else:
            boot_times = performance["mapper_boot_times"].values()
            start_time = float(os.getenv("KNN_START_TIME"))
            cumulative = True

            for t in boot_times:
                dt = t - start_time
                bucket = math.ceil(dt / 5)
                graph_data[bucket] += 1
                graph_max = max(graph_max, bucket)

        graph_sum = 0
        for i in range(graph_min, graph_max + 1):
            print(graph_data[i] + graph_sum)
            if cumulative:
                graph_sum += graph_data[i]


if __name__ == "__main__":
    main()
