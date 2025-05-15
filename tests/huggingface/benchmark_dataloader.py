"""
File: test_dataloader.py
Author: Jacob Alexander Rose
Created on Tuesday May 13th, 2025


"""

import torch
import torch.utils.data
import time
from typing import List, Dict
import numpy as np
from collections import defaultdict
from tqdm import trange, tqdm


class DataLoaderBenchmarker:
    def __init__(
        self,
        dataset,
        batch_sizes: List[int],
        num_workers_list: List[int],
        num_iterations: int = 10,
        warmup_iterations: int = 2,
    ):
        """
        Args:
            dataset: PyTorch Dataset instance
            batch_sizes: List of batch sizes to test
            num_workers_list: List of num_workers values to test
            num_iterations: Number of iterations for timing
            warmup_iterations: Number of warmup iterations before timing
        """
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self.num_workers_list = num_workers_list
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.results = defaultdict(list)

    def benchmark_dataloader(self, batch_size: int, num_workers: int) -> Dict:
        """Benchmark a single dataloader configuration."""
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

        print("initiating warmup")
        # Warm up
        # for _ in trange(self.warmup_iterations, desc="Warming up", position=0, leave=True):
        #     for _ in tqdm((dataloader), desc="Warmup", position=1, leave=False):
        #         pass
        dataloader_iter = iter(dataloader)
        for _ in trange(
            self.warmup_iterations, desc="Warming up", position=0, leave=True
        ):
            next(dataloader_iter)

        print("warmup complete, Initiating benchmark")
        times = []
        for _ in trange(
            self.num_iterations,
            desc=f"Benchmarking {self.num_iterations} iterations",
            position=0,
            leave=True,
        ):
            start_time = time.perf_counter()
            for batch in tqdm(dataloader, desc="dataloader", position=1, leave=False):
                pass
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "avg_time": avg_time,
            "std_time": std_time,
            "throughput": len(self.dataset) / avg_time,
        }

    def run_benchmark(self) -> List[Dict]:
        """Run benchmarks for all configurations."""
        results = []
        for batch_size in self.batch_sizes:
            for num_workers in self.num_workers_list:
                result = self.benchmark_dataloader(batch_size, num_workers)
                results.append(result)
                print(
                    f"Batch size: {batch_size}, Workers: {num_workers}, "
                    f"Time: {result['avg_time']:.2f} ± {result['std_time']:.2f}s, "
                    f"Throughput: {result['throughput']:.2f} samples/s"
                )
        return results

    def plot_results(self, results: List[Dict]):
        """Plot benchmark results."""
        import matplotlib.pyplot as plt

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot timing
        for num_workers in self.num_workers_list:
            worker_results = [r for r in results if r["num_workers"] == num_workers]
            batch_sizes = [r["batch_size"] for r in worker_results]
            times = [r["avg_time"] for r in worker_results]
            ax1.plot(batch_sizes, times, marker="o", label=f"Workers: {num_workers}")

        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("DataLoader Performance")
        ax1.legend()
        ax1.grid(True)

        # Plot throughput
        for num_workers in self.num_workers_list:
            worker_results = [r for r in results if r["num_workers"] == num_workers]
            batch_sizes = [r["batch_size"] for r in worker_results]
            throughputs = [r["throughput"] for r in worker_results]
            ax2.plot(
                batch_sizes, throughputs, marker="o", label=f"Workers: {num_workers}"
            )

        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Throughput (samples/s)")
        ax2.set_title("DataLoader Throughput")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("dataloader_benchmark.png")


# Example dataset (replace with your dataset)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(3, 224, 224)  # Simulate 224x224 RGB images


if __name__ == "__main__":
    from plantclef.pytorch.data_catalog import make_dataset
    from plantclef.embed.utils import print_current_time

    name = "plantclef2024"
    subset = "val"

    batch_sizes = [64, 128]
    num_workers_list = [2, 4]
    num_iterations = 5
    warmup_iterations = 2

    # Create dataset and run benchmarks
    # dataset = DummyDataset(1000)

    print(f"Initiating {__file__} Benchmark on subset {subset} of dataset {name}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Num workers: {num_workers_list}")
    print(f"Num iterations: {num_iterations}")
    print(f"Warmup iterations: {warmup_iterations}")
    print_current_time()

    data = make_dataset(name=name, subset=subset, load_all_subsets=False)
    data.set_transform()

    benchmarker = DataLoaderBenchmarker(
        data,
        batch_sizes=batch_sizes,
        num_workers_list=num_workers_list,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
    )

    results = benchmarker.run_benchmark()
    benchmarker.plot_results(results)
