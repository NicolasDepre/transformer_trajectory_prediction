import wandb
import pandas as pd
import argparse
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

def parse_args():
    parser = argparse.ArgumentParser(description='Produce plots from wandb data')

    parser.add_argument('--title', type=str, help='Title of the plot')
    parser.add_argument('--tags', type=str,  help='Tags to get from wandb', required=True)
    parser.add_argument('--to_plot_y', type=str, default='val_loss', help='Information to plot on the y axiss')
    parser.add_argument('--yscale', type=str, default='log', help='Scale of the y axis in the plot')
    parser.add_argument('--to_plot_x', type=str, default='_step', help='Information to show on the x axis')
    parser.add_argument('--samples', type=int, default=2000, help="Number of samples to download from wandb for the plot")
    parser.add_argument('--smoothed', action='store_true', help="If needs to be smoothed")
    parser.add_argument('--not_smoothed', dest='smoothed', action='store_false', help="If should not be smoothed")
    parser.set_defaults(smoothed=True)
    parser.add_argument('--smoothing', type=float, default=0.01, help='Alpha factor of smoothing')
    parser.add_argument('--per_epoch', dest='per_epoch', action='store_true', help='Plot mean per epoch')
    parser.set_defaults(per_epoch=False)


    parser.add_argument('--label_format', type=str, help="Format given to the labels of the curves")
    parser.add_argument('--xlabel', type=str, default='Step', help="Format given to the labels of the curves")
    parser.add_argument('--ylabel', type=str, default='Loss', help="Format given to the labels of the curves")
    
    parser.add_argument('--ystart', type=int, default=0, help="Format given to the labels of the curves")

    args = parser.parse_args()
    return args


def get_title(args):
    if args.title is not None:
        return args.title
    title = ""
    if args.to_plot_y == 'val_loss':
        title += "Validation Loss"
    elif args.to_plot_y ==  'train_loss':
        title += "Training Loss"
    elif args.to_plot_y == 'test_loss':
        title += "Test Loss"
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

def avg_per_epoch(stat_length, lst):
    average_per_epoch = []
    n_epochs = len(lst) // stat_length
    #assert n_epochs * stat_length == len(lst)
    for epoch in range(n_epochs):
        epoch_data = lst[epoch*stat_length : min((epoch+1)*stat_length, len(lst))]
        a = sum(epoch_data) / len(epoch_data) 
        average_per_epoch.append(a)
    return average_per_epoch
    

if __name__ == "__main__":
    args = parse_args()

    title = get_title(args)
    tags = args.tags
    to_plot = args.to_plot_y
    yscale = args.yscale
    samples = args.samples
    complete_history = True if samples == -1 else False
    smoothed = args.smoothed
    alpha = args.smoothing
    label_format = args.label_format
    xlabel = args.xlabel
    ylabel = args.ylabel
    ystart = args.ystart
    to_plot_x = args.to_plot_x

    if to_plot_x == "_runtime":
        runtime_plot = True
        xlabel = "Runtime [h]"
    else:
        runtime_plot = False

    group_per_epoch = args.per_epoch
    if group_per_epoch:
        complete_history = True
        xlabel = 'Epoch'

    project = "thesis_official_runs"
    api = wandb.Api()

    runs = sorted(api.runs(path=project, filters={'tags': tags}), key=lambda run: f"{get_label(run, label_format):0>40}")  # names in a more logical way up to 40 chars
    
    for run in runs:
        if complete_history:
            keys = keys=[to_plot_x, to_plot]
            hist = pd.DataFrame(run.scan_history(keys=keys)) # complete history
        else:
            #hist = run.history(samples=samples)
            hist = run.history(samples=samples, keys=[to_plot_x, to_plot]).sort_values([to_plot_x])
        print(hist)
        if runtime_plot:
            hist[to_plot_x] = hist[to_plot_x] / 3600  # go from seconds to hours
        if group_per_epoch:
            data_to_plot = hist[to_plot]
            if 'val' in to_plot:
                length_name = 'val_length'
            elif 'train' in to_plot:
                length_name = 'train_length'
            elif 'test' in to_plot:
                length_name = 'test_length'
            per_epoch = avg_per_epoch(run.config[length_name], hist[to_plot])

            plt.plot(range(ystart, len(per_epoch)), per_epoch[ystart:], label=get_label(run, label_format))
        else:
            hist['smoothed'] = hist[to_plot].ewm(alpha=alpha, adjust=False).mean()
            data_to_plot = hist['smoothed'] if smoothed else hist[to_plot]
            plt.plot(hist[to_plot_x], data_to_plot, label=get_label(run, label_format))
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.yscale(yscale)
    save_plot(f"plots/{tags}_{to_plot}_{to_plot_x.replace('_','')}_{yscale}.pdf", plt)
    plt.show()

