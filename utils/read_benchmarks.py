"""Helper functions to read and plot benchmarking data."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_iteration_data(directory: str, filename: str, labels: list) -> pd.DataFrame:
    """Read benchmarking data from each loop iteration.

    Parameters
    ----------
    directory : str
        Directory of file containing benchmarking data to be read.
    filename : str
        Name of file containing benchmarking data to be read.
    labels : list
        List of labels in output file to read.
        List does not need to be complete, but must be given in order of output.

    Returns
    -------
    df : pd.DataFrame
        Dataframe of durations, with columns corresponding to each input label.
    """
    df = pd.DataFrame(columns=labels)

    path = directory + filename
    print(f"Reading: {path}")

    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            # Get number of runs
            if "Running" in line.split() and "model:" in line.split():
                n_steps = int(line.split()[3])
                print(f"Number of runs: {n_steps}")
                iteration = 0
                durations = np.zeros((len(labels), n_steps))

            # Get durations for each loop
            if "check iteration" in line:
                # Check if line matches each label requested
                for i, label in enumerate(labels):
                    if label in line:
                        # Offset duration by length of label
                        durations[i, iteration] = float(
                            line.split()[4 + len(label.split())]
                        )

                        # If final label, next iteration
                        if label == labels[-1]:
                            iteration += 1

        # Populate dataframe from arrays
        for i, label in enumerate(labels):
            df[label] = durations[i, :]

    return df


def read_summary_data(directory: str, filename: str, labels: list) -> dict:
    """Read benchmarking summary data.

    Parameters
    ----------
    directory : str
        Directory of file containing benchmarking data to be read.
    filename : str
        Path to file containing benchmarking data to be read.
    labels : list
        List of labels to read summary information for.

    Returns
    -------
    results : dict
        Nested dictionary with keys for each label passed, and nested keys for
        the mean, min, max and stddev for each label.
    """
    results = {}
    current_label = None

    path = directory + filename
    print(f"Reading: {path}")

    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            # Get number of runs
            if "Running" in line.split() and "model:" in line.split():
                n_steps = int(line.split()[3])
                print(f"Number of runs: {n_steps}")

            # Get outputs for each label
            for label in labels:
                # Combined mean is output in same line as "Overall mean (s):"
                if "Overall mean" == label and label in line:
                    results[label] = {"mean": float(line.split()[3])}
                    break

                # All other outputs list min, max, mean, stddev on new lines
                if (
                    (current_label is None)
                    and (label in line)
                    and ("check iteration" not in line)
                ):
                    results[label] = {}
                    current_label = label
                    break

            # If current_label set, get min, max, mean, stdev
            if (current_label is not None) and (current_label not in line):
                if line.split()[0] == "sample":
                    current_label = None
                else:
                    results[current_label][line.split()[0]] = float(line.split()[4])

    return results


def plot_df(df: pd.DataFrame, labels: list) -> None:
    """Plot scatter plots for each column in input dataframe.

    Parameters
    ----------
        df : pd.DataFrame)
            Dataframe containing data to be plotted.
        labels : list
            List of columns in dataframe to plot.
    """
    # Create separate plots for each label.
    for label in labels:
        x = np.arange(len(df[label][1:]))
        plt.scatter(x, df[label][1:], s=1, marker="x")
        plt.xlabel("Iteration")
        plt.ylabel("Time / s")
        plt.title(label)
        plt.show()


def plot_summary_means(data: dict, labels: list) -> None:
    """Plot bar chart comparing durations for specified files and keys.

    Parameters
    ----------
    data : dict
        Dictionary of summary data in the form data[file][label][mean].
    labels : list
        List of summary labels to plot bar charts for.
    """
    alpha = 0.9
    bar_width = 1

    # Loop over each label
    for label in labels:
        for i, file in enumerate(data):
            y = data[file][label]["mean"]
            plt.bar(
                np.arange(len(labels))[i],
                y,
                alpha=alpha,
                width=bar_width,
            )
        plt.ylim(0)
        plt.title(label)
        files = list(data.keys())
        plt.xticks(
            ticks=np.arange(len(files)),
            labels=files,
            fontsize=7.5,
        )
        plt.ylabel("Time / s")
        plt.show()


def plot_summary_with_stddev(data: dict, labels: list) -> None:
    """Plot scatter plot with error bars of summary data from benchmarking output files.

    Parameters
    ----------
    data : dict
        Dictionary of summary data in the form data[file][label][mean, stddev].
    labels : list
        List of summary labels to plot on the same graph.
    """
    # Loop over each file
    for file in data:
        y = []
        yerr = []

        # Get means and stddevs for each label
        for label in labels:
            y.append(data[file][label]["mean"])
            # If no stddev, set error bar to 0
            try:
                yerr.append(data[file][label]["stddev"])
            except KeyError:
                yerr.append(0)

        # Plot scatter points for all labels in current file
        plt.scatter(
            np.arange(len(labels)),
            y,
            label=file,
            s=10,
            marker="o",
        )
        # Plot error bars with no lines for all labels in current file
        plt.errorbar(
            np.arange(len(labels)),
            y,
            yerr=yerr,
            capsize=3,
            capthick=1,
            ls="none",
            alpha=0.7,
        )

        plt.legend()
        plt.ylabel("Time / s")
        plt.xticks(
            ticks=np.arange(len(labels)),
            labels=labels,
            fontsize=7.5,
        )
    plt.show()


def read_slurm_walltime(filepath: str, labels: list) -> dict:
    """Read benchmarking data from each loop iteration.

    Parameters
    ----------
    filepath : str
        Path to file containing benchmarking data to be read.
    labels : list
        List of all benchmarks run, matching the run order.
        Typically of the form [model]_[forpy/torch]_[cpu/gpu].

    Returns
    -------
    benchmarks : dict
        Dictionary of times, with keys corresponding to each input label.
    """
    print(f"Reading: {filepath}")

    current_label = ""
    i = 0
    benchmarks = {}

    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            if "Command being timed" in line:
                # Cut from 'Command being timed: "./benchmarker_cgdrag_forpy...'
                # to 'cgdrag_forpy'
                current_label = line.split()[3][15:]
                # print(current_label)
            if "Elapsed (wall clock) time" in line:
                if current_label in labels[i]:
                    benchmarks[labels[i]] = convert_to_seconds(line.split()[7])
                    i += 1

    return benchmarks


def convert_to_seconds(time_str: str):
    """
    Convert wall time string from /usr/bin/time to time in seconds.

    Parameters
    ----------
    time_str : str
        Time in the format h:mm:ss or m:ss.

    Returns
    -------
    time : float
        Time in seconds.
    """
    time = time_str.split(":")
    if len(time) == 3:
        return float(time[0]) * 3600 + float(time[1]) * 60 + float(time[2])
    elif len(time) == 2:
        return float(time[0]) * 60 + float(time[1])
    else:
        raise ValueError("Time format not supported. Expected format: h:mm:ss or m:ss")


def plot_walltimes(benchmarks: dict, labels: list):
    """Plot bar charts comparing walltimes for all labels given.

    Parameters
    ----------
        benchmarks : dict
            Dictionary of times, with keys corresponding to each input label.
        labels : list
            List containing subset of benchmark keys to plot.
    """
    alpha = 0.9
    bar_width = 1

    for i, label in enumerate(labels):
        plt.bar(
            np.arange(len(labels))[i],
            benchmarks[label],
            alpha=alpha,
            width=bar_width,
        )
    # plt.yscale("log")
    plt.ylim(0)
    plt.xticks(
        ticks=np.arange(len(labels)),
        labels=labels,
        fontsize=5,
    )
    plt.ylabel("Time / s")
    plt.show()
