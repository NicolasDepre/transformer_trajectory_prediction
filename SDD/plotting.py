import matplotlib.pyplot as plt
import os
import wandb
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Produce plots from wandb data')

    parser.add_argument('--title', type=str, help='Title')
    parser.add_argument('--tags', type=str, default='exp_dim_model', help='Tags')
    parser.add_argument('--to_plot', type=str, default='val_loss', help='To plot')
    parser.add_argument('--yscale', type=str, default='log', help='Yscale')
    parser.add_argument('--samples', type=int, default=2000, help="Number of samples to download from wandb")

    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--not_smoothed', dest='smoothed', action='store_false')
    parser.set_defaults(smoothed=True)
    parser.add_argument('--smoothing', type=float, default=0.01, help='Alpha factor of smoothing')

    args = parser.parse_args()
    return args


def get_title(args):
    if args.title is not None:
        return args.title
    title = ""
    if args.smoothed:
        title += "Smoothed"

    if args.to_plot == 'val_loss':
        title += " Validation Loss"
    elif args.to_plot ==  'train_loss':
        title += " Training Loss"
    elif args.to_plot == 'test_loss':
        title += " Test Loss"
    else:
        raise Exception("Please provide a title")
    return title

def save_plot(filename, plt):
    saved = False
    while not saved:
        try:
            plt.savefig(filename)
            saved = True
        except PermissionError:
            print("Problem with filename")
            filename = filename.replace(".pdf", "_bis.pdf")
    return

if __name__ == "__main__":
    args = parse_args()

    title = get_title(args)
    tags = args.tags
    to_plot = args.to_plot
    yscale = args.yscale
    samples = args.samples
    complete_history = True if samples == -1 else False
    smoothed = args.smoothed
    alpha = args.smoothing

    project = "thesis_official_runs"
    api = wandb.Api()
    runs = api.runs(path=project, filters={'tags': tags})
    for run in runs:
        if complete_history:
            hist = pd.DataFrame(run.scan_history(keys=['_step', to_plot])) # complete history
        else:
            hist = run.history(samples=samples, keys=['_step', to_plot]).sort_values(['_step'])
        hist['smoothed'] = hist[to_plot].ewm(alpha=alpha, adjust=False).mean()
        data_to_plot = hist['smoothed'] if smoothed else hist[to_plot]
        plt.plot(hist['_step'], data_to_plot, label=run.config['name'])

    plt.title(title)
    plt.legend()
    plt.yscale(yscale)
    save_plot(f"plots/{tags}_{to_plot}.pdf", plt)
    plt.show()
