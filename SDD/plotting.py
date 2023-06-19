import matplotlib.pyplot as plt
import os
import wandb


def make_plot(title, training, validation=None, ylim=None):
    try:
        os.mkdir("plots")
    except FileExistsError:
        pass
    feature = title.split('-')[0]
    fig, ax = plt.subplots()

    avg_training = [sum(sublst) / len(sublst) for sublst in training]

    flat_training = flat_list(training)
    print("len flat: ", len(flat_training))
    if ylim is not None:
        plt.ylim(ylim)

    update_steps = range(0, len(flat_training))
    
    n_steps = len(flat_training)
    n_epochs = len(training)

    ax.plot(update_steps, flat_training, label='training ' + feature)
    ax.plot(range(0, n_steps, n_steps//n_epochs), avg_training, label='training avg per epoch ' + feature)

    if validation is not None:
        #flat_validation = flat_list(validation)
        #avg_validation = [sum(sublst) / len(sublst) for sublst in validation]
        ax.plot(range(0, n_steps, n_steps // n_epochs), validation, label='validation avg per epoch ' + feature)
        #ax.plot(update_steps, flat_validation, label='validation ' + feature)
        #ax.plot(range(0, len(flat_validation), len(validation[0])), avg_validation, label='validation avg per epoch ' + feature)

    ax.legend()

    plt.xlabel('update step')
    plt.ylabel(title)

    filename = get_filename_from_params(title)
    save_plot(filename, fig)
    plt.show()
    plt.close(fig)
    return


def plot_accuracies(training_accuracies, validation_accuracies=None):
    make_plot("accuracy", training_accuracies, validation_accuracies)


def plot_losses(training_losses, validation_losses=None):
    make_plot("Loss", training_losses, validation_losses)


def get_filename_from_params(title):
    fn = "plots/" + title
    fn = fn.replace('.', ',')
    return fn + ".pdf"


def save_plot(filename, fig):
    saved = False
    while not saved:
        try:
            plt.savefig(filename)
            saved = True
        except PermissionError:
            print("Problem with filename")
            filename = filename.replace(".pdf", "_bis.pdf")
    return


def flat_list(lst):
    try:
        return [elem for sublst in lst for elem in sublst]
    except Exception as e:
        print(e)
        return lst

import pandas as pd

#endpoint = "/depren/thesis_official_runs/runs/"
#run_ids = ["dmn0il2i"]
#runs = [api.run(endpoint + id) for id in run_ids]
#histories = [run.history().sort_values(['_step']) for run in runs]

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
    plt.savefig(f"plots/{tags}_{to_plot}.pdf")
    plt.show()
