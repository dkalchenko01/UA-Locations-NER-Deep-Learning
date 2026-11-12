"""W&B sweep trials using prepared NER records and loaders."""

import gc
import shutil
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from src.data_io import file_sha256, save_json
from src.dataset import LABEL_TO_ID, make_loader
from src.experiment_tracking import start_run, log_epoch, log_checkpoint
from src.metrics import competition_scores
from src.model import BiLSTMTagger
from src.preprocessing import clean_locations_v3
from src.training import set_seed, train_epoch, predict, save_checkpoint


def _backup_files(directory, *paths):
    if directory is not None:
        for path in paths:
            temporary = directory / (path.name + ".tmp")
            shutil.copy2(path, temporary)
            temporary.replace(directory / path.name)


def run_sweep_trial(base_config, provenance, tokenizer, train_loader, val_loader,
                    val_records, device, root, repo_dir):
    model, optimizer = None, None
    try:
        with start_run(
            model="bilstm", config=base_config,
            validation_version=base_config["validation_version"],
            metric_version=base_config["metric_version"],
            postprocessing_version="raw", repo_dir=repo_dir,
            split_manifest_path=repo_dir / "data/splits/validation_manifest.parquet",
            tokenizer_path=repo_dir / "artifacts/tokenizers" / base_config["tokenizer"],
            tags=["stacked", "sweep"],
        ) as run:
            config = {key: run.config[key] for key in base_config}
            loader_options = {
                "pad_id": tokenizer.token_to_id("[PAD]"),
                "batch_size": config["batch_size"], "seed": config["seed"],
                "max_padded_tokens": config["max_padded_tokens"],
                "pin_memory": device.type == "cuda",
            }
            train_records = train_loader.dataset.records
            train_loader = make_loader(train_records, shuffle=True, **loader_options)
            val_loader = make_loader(val_records, **loader_options)
            run_id = f"sweep_{run.id}"
            run.name = run_id
            run_dir = root / "outputs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            backup_dir = None
            if config.get("checkpoint_backup_dir"):
                backup_dir = Path(config["checkpoint_backup_dir"]) / run_id
                backup_dir.mkdir(parents=True, exist_ok=False)
                run.summary["checkpoints/backup_dir"] = str(backup_dir)
            trial_provenance = {
                **provenance, "run_id": run_id, "sweep_id": run.sweep_id,
                "source_sha256": {path.name: file_sha256(path)
                                  for path in sorted((repo_dir / "src").glob("*.py"))},
                "notebook_sha256": file_sha256(repo_dir / "notebooks/04_bilstm_sweep.ipynb"),
            }
            run.config.update({"provenance": trial_provenance})
            save_json(run_dir / "config.json", config)
            save_json(run_dir / "provenance.json", trial_provenance)
            save_json(run_dir / "wandb_run.json", {"id": run.id, "url": run.url})

            set_seed(config["seed"])
            model_config = {
                "vocab_size": tokenizer.get_vocab_size(),
                "pad_id": tokenizer.token_to_id("[PAD]"),
                **{key: config[key] for key in
                   ("embedding_dim", "hidden_size", "num_layers", "dropout")},
                "num_labels": len(LABEL_TO_ID),
            }
            model = BiLSTMTagger(**model_config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            run.config.update({"model_config": model_config})
            save_json(run_dir / "model_config.json", model_config)
            _backup_files(backup_dir, run_dir / "config.json", run_dir / "provenance.json",
                          run_dir / "wandb_run.json", run_dir / "model_config.json",
                          repo_dir / "artifacts/tokenizers" / config["tokenizer"])

            history, best_score, best_epoch = [], -1, None
            for epoch in range(1, config["epochs"] + 1):
                started = perf_counter()
                train_loader.batch_sampler.set_epoch(epoch)
                train_loss = train_epoch(model, train_loader, optimizer, device,
                                         config["gradient_clip"])
                predictions, val_losses = predict(model, val_loader, device)
                scores = competition_scores(val_records, predictions)
                clean_scores = competition_scores(val_records, predictions,
                                                  cleaner=clean_locations_v3)
                del predictions
                result = {
                    "epoch": epoch, "train_loss": train_loss,
                    "epoch_seconds": perf_counter() - started,
                    **{f"{source}_val_loss": value for source, value in val_losses.items()},
                    **scores, **{f"clean_{key}": value for key, value in clean_scores.items()},
                }
                history.append(result)
                pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
                save_checkpoint(run_dir / f"epoch_{epoch:02d}.pt", model, optimizer, epoch,
                                config, model_config, scores, trial_provenance)
                _backup_files(backup_dir, run_dir / f"epoch_{epoch:02d}.pt",
                              run_dir / "history.csv")
                if scores[config["selection_metric"]] > best_score:
                    best_score, best_epoch = scores[config["selection_metric"]], epoch
                    save_checkpoint(run_dir / "best.pt", model, optimizer, epoch,
                                    config, model_config, scores, trial_provenance)
                    _backup_files(backup_dir, run_dir / "best.pt")
                log_epoch(
                    run, epoch, result, selection_metric=config["selection_metric"],
                    epoch_seconds=result["epoch_seconds"],
                    extra_metrics={f"val_{config['validation_version']}/clean_v3/{key}": value
                                   for key, value in clean_scores.items()},
                )
                print(run_id, result)

            run.summary["best/checkpoint_local"] = str(run_dir / "best.pt")
            save_json(run_dir / "summary.json", {
                "best_epoch": best_epoch, "best_score": best_score,
                "selection_metric": config["selection_metric"],
            })
            _backup_files(backup_dir, run_dir / "summary.json")
            if config["upload_checkpoint"]:
                log_checkpoint(run, run_dir / "best.pt")
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
