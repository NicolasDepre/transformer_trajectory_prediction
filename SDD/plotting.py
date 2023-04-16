import matplotlib.pyplot as plt


def make_plot(title, training, validation=None, ylim=None):
    os.system("mkdir plots")
    feature = title.split('-')[0]
    fig, ax = plt.subplots()

    avg_training = [sum(sublst) / len(sublst) for sublst in training]

    flat_training = flat_list(training)
    print(len(flat_training))

    if ylim is not None:
        plt.ylim(ylim)

    update_steps = range(0, len(flat_training))

    ax.plot(update_steps, flat_training, label='training ' + feature)
    ax.plot(range(0, len(flat_training), len(training[0])), avg_training, label='training avg per epoch ' + feature)
    if validation is not None:
        ax.plot(update_steps, validation, label='validation ' + feature)

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
