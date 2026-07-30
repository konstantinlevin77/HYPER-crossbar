import os
import sys
import math
import pprint
from itertools import islice

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hyper import evaluation, tasks, util
from hyper.models import HYPER, TransductiveHCNet



separator = ">" * 30
line = "-" * 30


def _metric_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return value


def _wandb_log_metrics(wandb_logger, prefix, metrics, epoch):
    if wandb_logger is None or metrics is None:
        return

    payload = {"epoch": epoch}
    for metric_name, value in metrics.items():
        if metric_name.startswith("typed/"):
            # Keep node-specific metrics together in W&B under valid/typed/...
            # or test/typed/... while preserving legacy keys for old metrics.
            key = f"{prefix}/{metric_name}"
        else:
            key = f"{prefix}_{metric_name}"
        payload[key] = _metric_value(value)
    if payload:
        wandb_logger.log(payload)


def _metrics_to_compute(cfg):
    metrics = list(cfg.task.metric)
    for metric in ("mrr", "hits@1", "hits@10"):
        if metric not in metrics:
            metrics.append(metric)
    if cfg.task.get("typed_eval", False):
        for metric in list(metrics):
            if metric == "mrr" or metric.startswith("hits@"):
                typed_metric = f"typed_{metric}"
                if typed_metric not in metrics:
                    metrics.append(typed_metric)
    return metrics


def _score_ranking(metric, ranking, num_negative):
    if metric.startswith("typed_"):
        metric = metric[len("typed_"):]
    if metric == "mr":
        return ranking.float().mean()
    if metric == "mrr":
        return (1 / ranking.float()).mean()
    if metric.startswith("hits@"):
        values = metric[5:].split("_")
        threshold = int(values[0])
        if len(values) > 1:
            num_sample = int(values[1])
            # unbiased estimation
            fp_rate = (ranking - 1).float() / num_negative
            score = 0
            for i in range(threshold):
                # choose i false positive from num_sample - 1 negatives
                num_comb = math.factorial(num_sample - 1) / \
                           math.factorial(i) / math.factorial(num_sample - i - 1)
                score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
            return score.mean()
        return (ranking <= threshold).float().mean()
    raise ValueError(f"Unknown metric {metric!r}")



def train_and_validate(cfg, model, train_data, valid_data, device, logger, filtered_data=None, batch_per_epoch=None, neptune_logger=None, wandb_logger=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    eval_interval = cfg.train.get("eval_interval", math.ceil(cfg.train.num_epoch / 10))
    if eval_interval <= 0:
        raise ValueError("cfg.train.eval_interval must be a positive integer")
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, eval_interval):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + eval_interval)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.get("strict_negative", True),
                                                max_positions_per_edge=cfg.task.get("num_corrupt_positions"),
                                                sampling_mode=cfg.task.get("negative_sampling"))
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                    if wandb_logger is not None:
                        wandb_logger.log({"loss": loss.item(), "train/loss": loss.item()}, step=batch_id)
                losses.append(loss.item())
                if util.get_rank() == 0 and neptune_logger is not None:
                    neptune_logger["train/loss"].append(loss)
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)
                if neptune_logger is not None:
                    neptune_logger["train/epoch_loss"].append(avg_loss)
                if wandb_logger is not None:
                    wandb_logger.log({"epoch": epoch, "epoch_loss": avg_loss, "train/epoch_loss": avg_loss})

        epoch = min(cfg.train.num_epoch, i + eval_interval)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        valid_metrics = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger, return_metrics=True, neptune_logger=neptune_logger, logger_mode="valid")
        result = valid_metrics["mrr"]
        if rank == 0:
            _wandb_log_metrics(wandb_logger, "valid", valid_metrics, epoch)
            if wandb_logger is not None and cfg.train.get("log_train_metrics", True):
                logger.warning(separator)
                logger.warning("Evaluate on train")
                train_metrics = test(cfg, model, train_data, filtered_data=train_data, device=device, logger=logger, return_metrics=True, logger_mode="train")
                _wandb_log_metrics(wandb_logger, "train", train_metrics, epoch)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False, neptune_logger=None, logger_mode = "test"):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    if "num_eval_edges" in cfg.train and cfg.train.num_eval_edges is not None:
        test_triplets = test_triplets[:cfg.train.num_eval_edges]
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    node_specific_typed_eval = cfg.task.get("node_specific_typed_evaluation", False)
    typed_eval = cfg.task.get("typed_eval", False) or node_specific_typed_eval
    node_types = evaluation.node_types_for_dataset(cfg.dataset["class"])
    position_typed_rankings = evaluation.new_position_rankings()
    typed_rankings = []
    typed_num_negatives = []
    for batch in test_loader:
        eval_positions = tasks.get_active_positions(batch[:, :-1], cfg.task.get("num_eval_positions"))
        batch_list = tasks.all_negative(test_data, batch, positions=eval_positions)
        pred_list = []
        for new_batch in batch_list:
            pred = None if new_batch is None else model(test_data, new_batch)
            pred_list.append(pred)
        if filtered_data is None:
            mask_list = tasks.strict_negative_mask(test_data, batch, positions=eval_positions)
        else:
            mask_list = tasks.strict_negative_mask(filtered_data, batch, positions=eval_positions)  
        if typed_eval:
            typed_mask_source = test_data if filtered_data is None else filtered_data
            typed_mask_list = tasks.strict_typed_negative_mask(typed_mask_source, batch, positions=eval_positions)
        else:
            typed_mask_list = [None] * len(mask_list)
        pos_entities_index_list = batch.t()[:-1]

        ranking_list = []
        num_negative_list = []
        typed_ranking_list = []
        typed_num_negative_list = []
        # For each arity
        arity_pos = 0
        for position, (pred, mask, typed_mask, pos_entities_index, batch) in enumerate(zip(pred_list, mask_list, typed_mask_list, pos_entities_index_list, batch_list)):
            if pred is None:
                continue
            # Mask out the un-used arity
            non_zero_mask = pos_entities_index != 0
            pos_entities_index = pos_entities_index[non_zero_mask]
            mask = mask[non_zero_mask,1:] # remove the dummy node for mask and then apply the non_zero_mask
    
            ranking = tasks.compute_ranking(pred, pos_entities_index, mask)
            num_negative = mask.sum(dim=-1)
            ranking_list.append(ranking)
            num_negative_list.append(num_negative)
            if typed_eval:
                typed_mask = typed_mask[non_zero_mask, 1:]
                typed_ranking = tasks.compute_ranking(pred, pos_entities_index, typed_mask)
                typed_num_negative = typed_mask.sum(dim=-1)
                typed_ranking_list.append(typed_ranking)
                typed_num_negative_list.append(typed_num_negative)
                if node_specific_typed_eval:
                    evaluation.add_position_rankings(position_typed_rankings, position, typed_ranking)
            arity_pos += 1
        
        rankings += ranking_list
        num_negatives += num_negative_list
        typed_rankings += typed_ranking_list
        typed_num_negatives += typed_num_negative_list

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    if typed_eval:
        typed_ranking = torch.cat(typed_rankings)
        typed_num_negative = torch.cat(typed_num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    # obtaining all ranks 
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if typed_eval:
        all_typed_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_typed_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = typed_ranking
        all_typed_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_typed_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = typed_num_negative

    metrics = {}
    if node_specific_typed_eval:
        if node_types is None and rank == 0:
            logger.warning("Node-specific typed evaluation is not defined for dataset %s", cfg.dataset["class"])
        metrics.update(evaluation.node_specific_typed_metrics(position_typed_rankings, node_types, device))
    if rank == 0:
        for metric in _metrics_to_compute(cfg):
            if metric.startswith("typed_"):
                score = _score_ranking(metric, all_typed_ranking, all_typed_num_negative)
            else:
                score = _score_ranking(metric, all_ranking, all_num_negative)
            if neptune_logger is not None:
                neptune_logger[f"{logger_mode}/{metric}"]=score
            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
        for metric, score in metrics.items():
            if metric.startswith("typed/"):
                if neptune_logger is not None:
                    neptune_logger[f"{logger_mode}/{metric}"] = score
                logger.warning("%s: %g" % (metric, score))
    mrr = (1 / all_ranking.float()).mean()

    if return_metrics and "mrr" not in metrics:
        metrics["mrr"] = mrr
    return mrr if not return_metrics else metrics


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    neptune_logger = None
    wandb_logger = None
    if util.get_rank() == 0:
        neptune_logger = util.create_neptune_run(args, vars, cfg)
        wandb_logger = util.create_wandb_run(args, vars, cfg, working_dir=working_dir)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    device = util.get_device(cfg)
    dataset = util.build_dataset(cfg, device=device)
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    if cfg.model["class"] == "HYPER":
        model = HYPER(
            rel_model_cfg=cfg.model.relation_model,
            entity_model_cfg=cfg.model.entity_model,
        )
    elif cfg.model["class"] == "TransductiveHCNet":
        assert len(dataset.graphs) == 1, "TransductiveHCNet only supports single graph"
        model = TransductiveHCNet(
            entity_model_cfg=cfg.model.entity_model,
            num_relations=dataset.num_relations,
        )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)
    
    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test) 
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class'] or cfg.dataset['class'] in {"HYPERCOMPLIANT150", "HYPERRULE1", "HYPERRULE2"} or cfg.dataset['class'] in [dataset+version for version in ["100", "75", "50", "25"] for dataset in ["JF", "WP", "MFB"]]:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            # test filtering graph: inference edges + test edges
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

            # validation filtering graph: train edges + validation edges
            val_filtered_data = Data(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type]),
                num_nodes=valid_data.num_nodes
            )
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
        val_filtered_data = test_filtered_data = filtered_data

    
    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)
    
    train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger, neptune_logger=neptune_logger, wandb_logger=wandb_logger )
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    valid_metrics = test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger, neptune_logger=neptune_logger, logger_mode="valid", return_metrics=True)
    if util.get_rank() == 0:
        _wandb_log_metrics(wandb_logger, "valid", valid_metrics, cfg.train.num_epoch)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test_metrics = test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, neptune_logger=neptune_logger, logger_mode="test", return_metrics=True)
    if util.get_rank() == 0:
        _wandb_log_metrics(wandb_logger, "test", test_metrics, cfg.train.num_epoch)
    if util.get_rank() == 0:
        util.finish_wandb_run(wandb_logger)
