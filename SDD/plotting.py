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

if __name__ == "__main__":
    title = "Smoothed validation loss"
    tags = 'exp_dim_model'
    to_plot = "val_loss"
    yscale = "log"

    project = "thesis_official_runs"
    api = wandb.Api()
    runs = api.runs(path=project, filters={'tags': tags})
    complete_history = False
    for run in runs:
        if complete_history:
            hist = pd.DataFrame(run.scan_history(keys=['_step', to_plot])) # complete history
        else:
            hist = run.history(samples=25000, keys=['_step', to_plot]).sort_values(['_step'])
        #smoothed = data[col].ewm(alpha=0.1)
        print(hist)
        hist['smoothed'] = hist[to_plot].ewm(alpha=0.01, adjust=False).mean()

        plt.plot(hist['_step'], hist['smoothed'], label=run.config['name'])

    plt.title(title)
    plt.legend()
    plt.yscale(yscale)
    plt.show()