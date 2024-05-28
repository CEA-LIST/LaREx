import torch
import numpy as np
from dropblock import DropBlock2D
from torch.utils.data import DataLoader
from tqdm import tqdm
from .uncertainty_estimation import Hook

dropblock_ext = DropBlock2D(drop_prob=0.4, block_size=1)


def get_msp_score_rcnn(dnn_model: torch.nn.Module, input_dataloader: DataLoader):
    """
    Calculates the Maximum softmax probability score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_model.to(device)
    dl_preds_msp_scores = []

    with torch.no_grad():
        for i, image in enumerate(tqdm(input_dataloader, desc="Getting MSP score")):
            # image = image.to(device)
            results, _ = dnn_model(image)

            pred_scores = results.scores
            # ic(pred_score.shape)
            # get the max values:
            if len(pred_scores) > 0:
                dl_preds_msp_scores.append(pred_scores.max().reshape(1))
            else:
                dl_preds_msp_scores.append((torch.Tensor([0.])).to(device))

        dl_preds_msp_scores_t = torch.cat(dl_preds_msp_scores, dim=0)
        # ic(dl_preds_msp_scores_t.shape)
        # pred = np.max(softmax_fn(pred_logits).detach().cpu().numpy(), axis=1)
        dl_preds_msp_scores = dl_preds_msp_scores_t.detach().cpu().numpy()

    return dl_preds_msp_scores


def get_dice_feat_mean_react_percentile_rcnn(
        dnn_model: torch.nn.Module,
        ind_dataloader: DataLoader,
        react_percentile: int = 90
):
    feat_log = []
    dnn_model.model.eval()
    assert dnn_model.dice_react_precompute
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs in tqdm(ind_dataloader, desc="Setting up DICE/ReAct"):
        outputs = dnn_model(inputs)
        out = outputs.mean(0)
        out = out.view(1, -1)
        # score = dnn_model.fc(out)
        feat_log.append(out.data.cpu().numpy())
    feat_log_array = np.array(feat_log).squeeze()
    return feat_log_array.mean(0), np.percentile(feat_log_array, react_percentile)


def get_energy_score_rcnn(dnn_model: torch.nn.Module, input_dataloader: DataLoader):
    """
    Calculates the energy uncertainty score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_det_model.to(device)

    # Here we take the enrgy as a mean of the whole 1000 proposals
    raw_preds_energy_scores = []
    # Here we take the enrgy as a mean of the filtered detections after NMS
    filtered_preds_energy_scores = []

    with torch.no_grad():
        for i, image in enumerate(tqdm(input_dataloader, desc="Getting energy score")):
            results, box_cls = dnn_model(image)
            # Raw energy
            raw_energy_score = torch.logsumexp(box_cls[:, :-1], dim=1)
            raw_preds_energy_scores.append(raw_energy_score.mean().reshape(1))
            # Filtered energy
            filtered_energy_score = torch.logsumexp(results.inter_feat[:, :-1], dim=1)
            filtered_preds_energy_scores.append(filtered_energy_score.mean().reshape(1))

        raw_preds_energy_scores_t = torch.cat(raw_preds_energy_scores, dim=0)
        raw_preds_energy_scores = raw_preds_energy_scores_t.detach().cpu().numpy()
        filtered_preds_energy_scores_t = torch.cat(filtered_preds_energy_scores, dim=0)
        filtered_preds_energy_scores = filtered_preds_energy_scores_t.detach().cpu().numpy()

    return raw_preds_energy_scores, filtered_preds_energy_scores


# Get latent space Monte Carlo Dropout samples
def get_ls_mcd_samples_rcnn(model: torch.nn.Module,
                            data_loader: torch.utils.data.dataloader.DataLoader,
                            mcd_nro_samples: int,
                            hook_dropout_layer: Hook,
                            layer_type: str,
                            return_raw_predictions: bool,
                            return_raw_latent_activations: bool = False) -> torch.tensor:
    """
     Get Monte-Carlo Dropout samples from RCNN's Dropout or Dropblock Layer
     :param model: Torch model
     :type model: torch.nn.Module
     :param data_loader: Input samples (torch) DataLoader
     :type data_loader: DataLoader
     :param mcd_nro_samples: Number of Monte-Carlo Samples
     :type mcd_nro_samples: int
     :param hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
     :type hook_dropout_layer: Hook
     :param layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or Conv (Convolutional)
     :type: str
     :param return_raw_predictions: Returns the raw logits output
     :type: bool
     :param return_raw_latent_activations: Returns the raw latent activations without extracting the mean and
      concatenating
     :type: bool
     :return: Monte-Carlo Dropout samples for the input dataloader
     :rtype: Tensor
     """
    assert layer_type in ("FC", "Conv", "RPN", "backbone"), "Layer type must be either 'FC', 'RPN' or 'Conv'"
    assert return_raw_latent_activations + return_raw_predictions <= 1, "Not implemented otherwise"
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
            dl_imgs_latent_mcd_samples = []
            if return_raw_predictions:
                raw_predictions = []
            if return_raw_latent_activations:
                raw_latent_activations = []
            for i, image in enumerate(data_loader):
                img_mcd_samples = []
                for s in range(mcd_nro_samples):
                    instances, _ = model(image)
                    if return_raw_predictions:
                        raw_predictions.append(instances.inter_feat[:, :-1].mean(0))
                    # pred = torch.argmax(pred_img, dim=1)
                    latent_mcd_sample = hook_dropout_layer.output
                    if layer_type == "Conv":
                        # Get image HxW mean:
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                        # Remove useless dimensions:
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=3)
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                    elif layer_type == "RPN":
                        if return_raw_latent_activations:
                            raw_latent_activations.append(
                                model.model.proposal_generator.rpn_head.rpn_intermediate_output.clone().detach()
                            )
                        latent_mcd_sample = model.model.proposal_generator.rpn_head.rpn_intermediate_output
                        for i in range(len(latent_mcd_sample)):
                            latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=2, keepdim=True)
                            latent_mcd_sample[i] = torch.mean(latent_mcd_sample[i], dim=3, keepdim=True)
                            # Remove useless dimensions:
                            latent_mcd_sample[i] = torch.squeeze(latent_mcd_sample[i])
                        latent_mcd_sample = torch.cat(latent_mcd_sample, dim=0)
                    elif layer_type == "backbone":
                        # Apply dropblock
                        for k, v in latent_mcd_sample.items():
                            latent_mcd_sample[k] = dropblock_ext(v)
                            # Get image HxW mean:
                            latent_mcd_sample[k] = torch.mean(latent_mcd_sample[k], dim=2, keepdim=True)
                            latent_mcd_sample[k] = torch.mean(latent_mcd_sample[k], dim=3, keepdim=True)
                            # Remove useless dimensions:
                            latent_mcd_sample[k] = torch.squeeze(latent_mcd_sample[k])
                        latent_mcd_sample = torch.cat(list(latent_mcd_sample.values()), dim=0)
                    # FC
                    else:
                        # Aggregate the second dimension (dim 1) to keep the proposed boxes dimension
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=1)
                    if (layer_type == "FC" and latent_mcd_sample.shape[0] == 1000) or layer_type == "RPN":
                        img_mcd_samples.append(latent_mcd_sample)
                    elif layer_type == "FC" and latent_mcd_sample.shape[0] != 1000:
                        pass
                    else:
                        raise NotImplementedError
                if layer_type == "Conv":
                    img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
                    dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                else:
                    if (layer_type == "FC" and latent_mcd_sample.shape[0] == 1000) or layer_type == "RPN":
                        img_mcd_samples_t = torch.stack(img_mcd_samples, dim=0)
                        dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                    elif layer_type == "FC" and latent_mcd_sample.shape[0] != 1000:
                        print(f"Omitted image: {image[0]['image_id']}")
                    else:
                        raise NotImplementedError

                # Update progress bar
                pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)

    if return_raw_predictions:
        return dl_imgs_latent_mcd_samples_t, torch.stack(raw_predictions, dim=0)
    elif return_raw_latent_activations:
        return dl_imgs_latent_mcd_samples_t, raw_latent_activations
    else:
        return dl_imgs_latent_mcd_samples_t
