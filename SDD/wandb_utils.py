import wandb

api = wandb.Api()
#run_id = "97y0w076"
#run = api.run("/depren/thesis_official_runs/runs/" + run_id)

project = "thesis_official_runs"
runs = api.runs(path=project)

for run in runs:
    print(run)
    run.config['img_step'] = 30
    run.update()