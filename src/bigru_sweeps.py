"""Three controlled Stacked BiGRU variants for a small W&B grid sweep.

The only W&B sweep parameter is `trial_variant`. This avoids accidentally
running the 2x2x2 Cartesian product of layers, size and learning rate.
"""

import gc
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from src.bigru_helpers import backup_files
from src.bigru_model import StackedBiGRUTagger
from src.data_io import file_sha256, save_json
from src.dataset import LABEL_TO_ID, make_loader
from src.experiment_tracking import log_checkpoint, log_epoch, start_run
from src.metrics import competition_scores
from src.preprocessing import clean_locations_v3
from src.training import predict, save_checkpoint, set_seed, train_epoch


# A is trained in the main notebook; B/C/D are follow-up experiments.
TRIALS = {
    "B_h384": {"num_layers": 3, "hidden_size": 384, "learning_rate": 0.001},
    "C_4layers": {"num_layers": 4, "hidden_size": 256, "learning_rate": 0.001},
    "D_lr3e-4": {"num_layers": 3, "hidden_size": 256, "learning_rate": 0.0003},
}


def run_sweep_trial(base_config, provenance, tokenizer, train_records,
                    val_records, device, root, repo_dir, notebook_path=None):
    """Run exactly one W&B-agent-selected variant using preprepared records."""
    model = optimizer = None
    try:
        with start_run(
            model="bigru", config=base_config,
            validation_version=base_config["validation_version"],
            metric_version=base_config["metric_version"],
            postprocessing_version="raw", repo_dir=repo_dir,
            split_manifest_path=repo_dir / "data/splits/validation_manifest.parquet",
            tokenizer_path=repo_dir / "artifacts/tokenizers" / base_config["tokenizer"],
            tags=["stacked", "bigru", "whitespace-bpe", "sweep"],
        ) as run:
            variant = run.config.get("trial_variant")
            if variant not in TRIALS:
                raise ValueError(f"Unknown sweep variant {variant!r}; expected one of {list(TRIALS)}")
            config = {**base_config, **TRIALS[variant], "trial_variant": variant}
            # Variant is a sweep parameter, all other hyperparameters are metadata.
            run.config.update({key: config[key] for key in
                               ("hidden_size", "num_layers", "learning_rate")},
                              allow_val_change=True)
            run_id = f"bigru_sweep_{variant}_{run.id}"
            run.name = run_id
            run_dir = Path(root) / "outputs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            backup_dir = (Path(config["checkpoint_backup_dir"]) / run_id
                          if config.get("checkpoint_backup_dir") else None)
            if backup_dir:
                backup_dir.mkdir(parents=True, exist_ok=True)
                run.summary["checkpoints/backup_dir"] = str(backup_dir)
            trial_provenance = {
                **provenance, "run_id": run_id, "sweep_id": run.sweep_id,
                "trial_variant": variant,
                "source_sha256": {p.name: file_sha256(p)
                                  for p in sorted((repo_dir / "src").glob("*.py"))},
                "notebook_sha256": file_sha256(notebook_path or (repo_dir / "notebooks/04_bigru_sweep.ipynb")),
            }
            save_json(run_dir / "config.json", config)
            save_json(run_dir / "provenance.json", trial_provenance)
            save_json(run_dir / "wandb_run.json", {
                "id": run.id, "url": run.url,
                "path": f"{run.entity}/{run.project}/{run.id}",
            })
            pad_id = tokenizer.token_to_id("[PAD]")
            if pad_id is None:
                raise ValueError("[PAD] missing from tokenizer")
            loader_kw = {
                "pad_id": pad_id, "batch_size": config["batch_size"],
                "seed": config["seed"],
                "max_padded_tokens": config["max_padded_tokens"],
                "pin_memory": device.type == "cuda",
            }
            train_loader = make_loader(train_records, shuffle=True, **loader_kw)
            val_loader = make_loader(val_records, **loader_kw)

            set_seed(config["seed"])
            model_config = {
                "vocab_size": tokenizer.get_vocab_size(), "pad_id": pad_id,
                **{key: config[key] for key in
                   ("embedding_dim", "hidden_size", "num_layers", "dropout")},
                "num_labels": len(LABEL_TO_ID),
            }
            model = StackedBiGRUTagger(**model_config).to(device)
            opt_name = config["optimizer"]
            optimizer_type = getattr(torch.optim, opt_name)
            optimizer_kw = {"lr": config["learning_rate"]}
            if opt_name == "AdamW":
                optimizer_kw["weight_decay"] = config.get("weight_decay", 0.01)
            optimizer = optimizer_type(model.parameters(), **optimizer_kw)
            run.config.update({"model_config": model_config, "optimizer_effective": opt_name})
            save_json(run_dir / "model_config.json", model_config)
            tokenizer_path = repo_dir / "artifacts/tokenizers" / config["tokenizer"]
            backup_files(backup_dir, run_dir / "config.json",
                         run_dir / "provenance.json", run_dir / "model_config.json",
                         run_dir / "wandb_run.json", tokenizer_path)

            history, best_score, best_epoch = [], float("-inf"), None
            for epoch in range(1, config["epochs"] + 1):
                started = perf_counter()
                train_loader.batch_sampler.set_epoch(epoch)
                train_loss = train_epoch(model, train_loader, optimizer, device,
                                         config["gradient_clip"])
                predictions, val_losses = predict(model, val_loader, device,
                                                  decoding="argmax")
                scores = competition_scores(val_records, predictions)
                clean_scores = competition_scores(val_records, predictions,
                                                  cleaner=clean_locations_v3)
                del predictions
                result = {
                    "epoch": epoch, "train_loss": train_loss,
                    "epoch_seconds": perf_counter() - started,
                    **{f"{source}_val_loss": value for source, value in val_losses.items()},
                    **scores,
                    **{f"clean_{key}": value for key, value in clean_scores.items()},
                }
                history.append(result)
                pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
                epoch_path = run_dir / f"epoch_{epoch:02d}.pt"
                save_checkpoint(epoch_path, model, optimizer, epoch, config,
                                model_config, scores, trial_provenance)
                if scores[config["selection_metric"]] > best_score:
                    best_score = scores[config["selection_metric"]]
                    best_epoch = epoch
                    save_checkpoint(run_dir / "best.pt", model, optimizer, epoch,
                                    config, model_config, scores, trial_provenance)
                    backup_files(backup_dir, run_dir / "best.pt")
                backup_files(backup_dir, epoch_path, run_dir / "history.csv")
                log_epoch(
                    run, epoch, result, selection_metric=config["selection_metric"],
                    epoch_seconds=result["epoch_seconds"],
                    extra_metrics={
                        f"val_{config['validation_version']}/clean_v3/{key}": value
                        for key, value in clean_scores.items()
                    },
                )
                print(run_id, result)

            save_json(run_dir / "summary.json", {
                "best_epoch": best_epoch, "best_score": best_score,
                "selection_metric": config["selection_metric"],
                "tokenizer": config["tokenizer"],
            })
            backup_files(backup_dir, run_dir / "summary.json")
            run.summary["best/checkpoint_local"] = str(run_dir / "best.pt")
            if config.get("upload_checkpoint", False):
                log_checkpoint(run, run_dir / "best.pt")
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
