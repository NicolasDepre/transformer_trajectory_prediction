import wandb
import pandas as pd
import argparse
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

def parse_args():
    parser = argparse.ArgumentParser(description='Produce plots from wandb data')

    parser.add_argument('--title', type=str, help='Title of the plot')
    parser.add_argument('--tags', type=str,  help='Tags to get from wandb', required=True)
    parser.add_argument('--to_plot', type=str, default='val_loss', help='Information to plot')
    parser.add_argument('--yscale', type=str, default='log', help='Scale of the y axis in the plot')
    parser.add_argument('--samples', type=int, default=2000, help="Number of samples to download from wandb for the plot")

    parser.add_argument('--smoothed', action='store_true', help="If needs to be smoothed")
    parser.add_argument('--not_smoothed', dest='smoothed', action='store_false', help="If should not be smoothed")
    parser.set_defaults(smoothed=True)
    parser.add_argument('--smoothing', type=float, default=0.01, help='Alpha factor of smoothing')

    parser.add_argument('--label_format', type=str, help="Format given to the labels of the curves")
    parser.add_argument('--xlabel', type=str, default='step', help="Format given to the labels of the curves")
    parser.add_argument('--ylabel', type=str, default='loss', help="Format given to the labels of the curves")

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

def get_label(run, label_format):
    if label_format is None:
        return run.config['name']
    label = label_format.replace("{", "{run.config['").replace("}", "']}")
    return eval("f\"" + label + "\"")


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
    label_format = args.label_format
    xlabel = args.xlabel
    ylabel = args.ylabel

    project = "thesis_official_runs"
    api = wandb.Api()

    runs = sorted(api.runs(path=project, filters={'tags': tags}), key=lambda run: f"{get_label(run, label_format):0>40}")  # names in a more logical way up to 40 chars
    for run in runs:
        if complete_history:
            hist = pd.DataFrame(run.scan_history(keys=['_step', to_plot])) # complete history
        else:
            hist = run.history(samples=samples, keys=['_step', to_plot]).sort_values(['_step'])
        print(hist)
        hist['smoothed'] = hist[to_plot].ewm(alpha=alpha, adjust=False).mean()
        data_to_plot = hist['smoothed'] if smoothed else hist[to_plot]
        plt.plot(hist['_step'], data_to_plot, label=get_label(run, label_format))
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.yscale(yscale)
    save_plot(f"plots/{tags}_{to_plot}.pdf", plt)
    plt.show()

