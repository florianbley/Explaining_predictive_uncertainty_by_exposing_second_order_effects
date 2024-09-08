from datasets import EPEX_FR, BiasCorrection, californiaHousing, wineQuality, SeoulBike, kin8nm, YearPredictionMSD
from MLP import MLP
from CNN import CNN
from Ensemble_regressor import Ensemble_regressor
from Ensemble_regressor_MC_dropout import Ensemble_regressor_MC_dropout
import os
import pickle
import torch
import numpy as np
import copy
import captum
from captum.attr import IntegratedGradients
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function
import sklearn
from sklearn.metrics.pairwise import euclidean_distances


import random
from kde import KDE

def provide_data(dataset_name):
    if dataset_name == "EPEX-FR":
        X_train, X_test, X_val, y_train, y_test, y_val = EPEX_FR.serve_dataset(lookback=48)

    if dataset_name == "EPEX-FR_channel":
        X_train, X_test, X_val, y_train, y_test, y_val = EPEX_FR.serve_dataset(lookback=48, format="channel")

    if dataset_name == "Bias Correction":
        X_train, X_test, X_val, y_train, y_test, y_val = BiasCorrection.serve_dataset()

    if dataset_name == "California Housing":
        X_train, X_test, X_val, y_train, y_test, y_val = californiaHousing.serve_dataset()

    if dataset_name == "Wine Quality":
        X_train, X_test, X_val, y_train, y_test, y_val = wineQuality.serve_dataset()

    if dataset_name == "Seoul Bike":
        X_train, X_test, X_val, y_train, y_test, y_val = SeoulBike.serve_dataset(lookback=10)

    if dataset_name == "Seoul Bike_channel":
        X_train, X_test, X_val, y_train, y_test, y_val = SeoulBike.serve_dataset(lookback=10, format="channel")

    if dataset_name == "kin8nm":
        X_train, X_test, X_val, y_train, y_test, y_val = kin8nm.serve_dataset()

    if dataset_name == "YearPredictionMSD":
        X_train, X_test, X_val, y_train, y_test, y_val = YearPredictionMSD.serve_dataset()

    return X_train, X_test, X_val, y_train, y_test, y_val


def get_ensemble_path(dataset_name, n_models, model_type):
    save_dir = "models/pixelflipping"
    # if directory does not exist, create it
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = save_dir + "/{}_ensemble_data={}_nmodels={}.pkl".format(model_type, dataset_name, n_models)
    return model_path


def check_if_ensemble_exists(dataset_name, n_models, model_type="MLP"):
    model_path = get_ensemble_path(dataset_name, n_models, model_type)
    # check if file exists
    if os.path.isfile(model_path):
        return True
    else:
        return False


def check_if_some_ensemble_exists(dataset_name, model_type):
    # check if some ensemble for this dataset and model type exists
    # if so, return True and the number of models in the ensemble
    some_exists = False
    n_models_of_existing_ensemble = None
    for n_models in range(1, 20):
        ensemble_exists = check_if_ensemble_exists(dataset_name, n_models, model_type)
        if ensemble_exists:
            some_exists = True
            n_models_of_existing_ensemble = n_models
            break
    return some_exists, n_models_of_existing_ensemble


def load_ensemble(dataset_name, n_models, model_type):
    model_path = get_ensemble_path(dataset_name, n_models, model_type)
    # load the ensemble
    with open(model_path, "rb") as file:
        ensemble = pickle.load(file)
        file.close()
    return ensemble


def load_training_params(dataset_name):
    params_path = "params/{}_training_params.txt".format(dataset_name)
    try:
        training_params = {}
        with open(params_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split(" = ")
                    training_params[key] = eval(value)
            file.close()
    except FileNotFoundError:
        # training_params is empty dictionary
        raise FileNotFoundError("Training parameters not found. Please provide parameter txt file first.")
    return training_params

def unpack_dataset(dataset):
    X_train, X_test, X_val, y_train, y_test, y_val = dataset
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    X_val = torch.tensor(X_val).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    y_val = torch.tensor(y_val).float()
    return X_train, X_test, X_val, y_train, y_test, y_val


def save_ensemble(ensemble, dataset_name, n_models, model_type="MLP"):

    # count number of relu layers in the ensemble.models[0]
    n_layers = 0
    for layer in ensemble.models[0].get_layers():
        if type(layer) == torch.nn.ReLU:
            n_layers += 1

    save_path = get_ensemble_path(dataset_name, n_models, model_type)
    with open(save_path, "wb") as file:
        pickle.dump(ensemble, file)
        file.close()

def train_ensemble(dataset_name, dataset, n_models, model_type="MLP"):

    # fix the random seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    training_params = load_training_params(dataset_name)

    X_train, X_test, X_val, y_train, y_test, y_val = unpack_dataset(dataset)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_val = X_val.cuda()
        y_val = y_val.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

    # set the training hyperparameters
    dropout_rate = training_params["dropout_rate"]
    input_shape = X_train.shape[1]
    learning_rate = training_params["learning_rate"]
    n_epochs = 100
    n_neurons = 300
    batch_size = training_params["batch_size"]
    n_outputs = y_train.shape[1]

    loss_fn = torch.nn.MSELoss()

    best_models = []
    for seed in range(n_models):
        n_not_improved = 0
        best_val_loss = float("inf")
        torch.manual_seed(seed)
        np.random.seed(seed)

        if model_type == "MLP":
            model = MLP(input_shape, n_outputs, dropout_rate=dropout_rate, n_neurons=n_neurons, bias=True)
        elif model_type == "CNN":
            model = CNN(
                n_channels=X_train.shape[1], n_input_len=X_train.shape[2],
                n_outputs=y_train.shape[1], dropout_rate=dropout_rate, bias=True)
        else:
            raise ValueError("Model type not recognized. Please use 'MLP' or CNN.")
        # move model to GPU if available
        if cuda_available:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            # make a list of batch indices
            train_inds = list(range(len(X_train)))
            # shuffle the list
            np.random.shuffle(train_inds)
            # split the list into batches

            n_batches = int(len(X_train) / batch_size)
            batch_ind_list = [list(train_inds[i * batch_size: (i + 1) * batch_size]) for i in range(n_batches)]

            model.train()
            for batch_inds in batch_ind_list:
                X_batch, y_batch = X_train[batch_inds], y_train[batch_inds]
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                y_pred_train = model(X_train)
                train_loss = loss_fn(y_pred_train, y_train)

                y_pred_val = model(X_val)
                val_loss = loss_fn(y_pred_val, y_val)

            print("Epoch: {}, Training loss: {}, Validation loss: {}".format(epoch, train_loss, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                n_not_improved = 0
            else:
                n_not_improved += 1
                if n_not_improved > 10:
                    break
        best_models.append(best_model)

    ensemble = Ensemble_regressor(best_models)
    # print out average test loss of the models and the ensemble
    with torch.no_grad():
        single_model_losses = []
        for model in best_models:
            model.eval()
            y_pred_test = model(X_test)
            test_loss = loss_fn(y_pred_test, y_test).item()
            single_model_losses.append(test_loss)
        ensemble_pred_test = ensemble.forward(X_test)
        mean_pred_test = ensemble_pred_test.mean(1).cuda()
        test_loss = loss_fn(mean_pred_test, y_test).item()
        print("Average test loss of the models: {}".format(np.mean(single_model_losses)))
        print("Test loss of the ensemble: {}".format(test_loss))

    # save ensemble
    save_ensemble(ensemble, dataset_name, n_models, model_type=model_type)

    return ensemble


def infer_model_type(dataset_name):
    if dataset_name.split("_")[-1] == "channel":
        return "CNN"
    else:
        return "MLP"


def provide_ensemble(dataset_name, dataset, n_models, uncertainty_type):
    # model type is MLP or CNN if dataset indicates channelized format
    model_type = infer_model_type(dataset_name)

    # check if the model has already been trained
    if uncertainty_type == "ensemble":
        ensemble_saved = check_if_ensemble_exists(dataset_name, n_models, model_type)
        if ensemble_saved:
            ensemble = load_ensemble(dataset_name, n_models, model_type)

        else:
            ensemble = train_ensemble(dataset_name, dataset, n_models=n_models, model_type=model_type)
        return ensemble

    elif uncertainty_type == "MC_dropout":
        ensemble_saved, n_models_of_existing_ensmble = \
            check_if_some_ensemble_exists(dataset_name, model_type=model_type)
        if ensemble_saved:
            ensemble = load_ensemble(dataset_name, n_models_of_existing_ensmble, model_type)
        else:
            ensemble = train_ensemble(dataset_name, dataset, n_models=1, model_type=model_type)

        # for MC_dropout we wrap one network instance in a custom ensemble class which
        # initializes the same instance multiple times with different dropout seeds
        nn_instance = ensemble.models[0]
        MC_ensemble = Ensemble_regressor_MC_dropout(nn_instance, MC_passes=n_models)
        return MC_ensemble

    else:
        raise ValueError("Uncertainty type not recognized. Please use 'ensemble' or 'MC_dropout'.")


def add_flipping_dataset(dataset, ensemble, n_samples=100):
    X_train, X_test, X_val, y_train, y_test, y_val = unpack_dataset(dataset)

    # compute epistemic uncertainties
    vars_test = ensemble.forward(X_test).var(1).sum(1).detach().numpy()

    # choose subset of datapoints to explain
    indices = np.argsort(vars_test)[-n_samples:]
    X_flipping, y_flipping = X_test[indices], y_test[indices]
    return X_flipping, y_flipping

def make_heatmap_covariances(R_all):
    R_covariances = []

    n_samples = R_all.shape[1]
    for i in range(n_samples):
        R_i_all = R_all[:, i, :]
        R_output_covs = []
        for d in range(R_i_all.shape[1]):
            R_i_d_all = R_i_all[:, d]
            R_cov_sample_output = np.cov(R_i_d_all, rowvar=False, bias=True)
            R_output_covs.append(R_cov_sample_output)
        R_output_covs = np.stack(R_output_covs).sum(0)
        R_covariances.append(R_output_covs)
    R_covariances = np.stack(R_covariances)
    return R_covariances

def hessian_times_input(ensemble, X):
    ensemble.eval()
    explanations = []
    for sample in X:
        sample = sample[None]
        sample.requires_grad = True
        hessian = torch.autograd.functional.hessian(ensemble.epistemic_variance, sample)
        h_i = sample * hessian[0, :, 0] * sample.T
        explanations.append(h_i)
    H_I = torch.stack(explanations).detach().numpy()
    return H_I

def gradient_times_input(ensemble, X):
    ensemble.eval()
    X = X.clone()
    X.requires_grad = True
    ensemble_output = ensemble.forward(X)
    vars = ensemble_output.var(1).sum(1)
    vars.sum().backward()
    gradient = X.grad
    GI = gradient * X
    return GI.detach().numpy()

def sensitivity(ensemble, X):
    ensemble.eval()
    X = X.clone()
    X.requires_grad = True
    ensemble_output = ensemble.forward(X)
    vars = ensemble_output.var(1).sum(1)
    vars.sum().backward()
    gradient = X.grad
    sensitivity = (gradient**2).detach().numpy()
    return sensitivity


def covariance_explanation(arr_heatmaps, input_shape):
    #Assumes a collection of covariance heatmaps and returns the diagonal of each heatmap
    # and reshapes it to the original shape of the input.
    n_samples = arr_heatmaps.shape[1]
    heatmap_covariances = make_heatmap_covariances(arr_heatmaps)
    R_diag = np.stack([np.diag(heatmap_covariances[i]) for i in range(heatmap_covariances.shape[0])])
    R_marg = heatmap_covariances.sum(1)

    # reshape to original shape
    R_diag = R_diag.reshape((n_samples, *input_shape))
    R_marg = R_marg.reshape((n_samples, *input_shape))

    return R_diag, R_marg


def get_benchmark_explanations(
        ensemble, X_flipping, ensemble_type="Ensemble", gamma=0,
explanation_names=["LRP"]):
    benchmark_explanations = {}
    n_outputs = ensemble.models[0].n_outputs
    for explanation_name in explanation_names:
        if explanation_name == "LRP":
            R_lrp = ensemble.lrp(X_flipping, gamma=gamma)
            benchmark_explanations["LRP"] = R_lrp
            continue

        elif explanation_name == "CovLRP":
            R_lrp_all = ensemble.lrp_all(X_flipping, gamma=gamma, R_upper=None)
            # ensure correct shape, flatten channels if data is channelised
            R_lrp_all = R_lrp_all.reshape((*R_lrp_all.shape[:3], -1))
            R_lrp_diag, R_lrp_marg = covariance_explanation(R_lrp_all, input_shape=X_flipping.shape[1:])

            benchmark_explanations["CovLRP_diag"] = R_lrp_diag
            benchmark_explanations["CovLRP_marg"] = R_lrp_marg
            continue

        elif explanation_name == "GI":
            # gradient times input
            GI = gradient_times_input(ensemble, X_flipping)
            benchmark_explanations["GI"] = GI
            continue

        elif explanation_name == "Hessian" or explanation_name == "CovGI":
            # 1) make collection of GI predictions for each sample
            GIs = []
            for model in ensemble.models:
                output_GIs_of_model = []
                X_flipping_clone = X_flipping.clone()
                X_flipping_clone.requires_grad = True
                model_output = model(X_flipping_clone)
                for out in range(n_outputs):
                    model_output[:, out].sum().backward(retain_graph=True)
                    grad = X_flipping_clone.grad
                    GI = grad * X_flipping_clone
                    output_GIs_of_model.append(GI.detach().numpy())
                # make sure shape is MxNxOx...
                model_GI = np.swapaxes(np.stack(output_GIs_of_model), 0, 1)
                GIs.append(model_GI)
            R_GI_all = np.stack(GIs)

            R_GI_all = R_GI_all.reshape((*R_GI_all.shape[:3], -1))
            R_GI_diag, R_GI_marg = covariance_explanation(R_GI_all, input_shape=X_flipping.shape[1:])

            benchmark_explanations["CovGI_diag"] = R_GI_diag
            benchmark_explanations["CovGI_marg"] = R_GI_marg
            continue


        elif explanation_name == "IG":
            # integrated gradients
            baseline = X_flipping * 0
            IG = IntegratedGradients(ensemble.epistemic_variance).attribute(X_flipping, baseline,
                                                                            n_steps=50).detach().numpy()
            benchmark_explanations["IG"] = IG
            continue

        elif explanation_name == "CovIG":
            baseline = X_flipping * 0
            list_of_IG_exps = []
            for model in ensemble.models:
                list_of_output_dim_exps = []
                for d in range(n_outputs):
                    # create a small wrapper around model.forward to return only the d-th output
                    def forward_with_output_dimension(x):
                        return model.forward(x)[:, d]
                    IG = IntegratedGradients(forward_with_output_dimension)\
                        .attribute(X_flipping, baseline, n_steps=50).detach().numpy()
                    list_of_output_dim_exps.append(IG)
                list_of_IG_exps.append(np.stack(list_of_output_dim_exps))
            # transpose since covariance explanation expects M x N x D_out x D_in
            R_IG_all = np.stack(list_of_IG_exps).swapaxes(1, 2)
            R_IG_all = R_IG_all.reshape((*R_IG_all.shape[:3], -1))
            R_IG_diag, R_IG_marg = covariance_explanation(R_IG_all, input_shape=X_flipping.shape[1:])

            benchmark_explanations["CovIG_diag"] = R_IG_diag
            benchmark_explanations["CovIG_marg"] = R_IG_marg
            continue

        elif explanation_name == "Sensitivity":
            S = sensitivity(ensemble, X_flipping)
            benchmark_explanations["Sensitivity"] = S
            continue

        elif explanation_name == "Shapley":
            baseline = X_flipping.mean(0) * 0
            shapley_sampler = captum.attr.ShapleyValueSampling(ensemble.epistemic_variance)
            R_shapley = shapley_sampler.attribute(inputs=X_flipping, baselines=baseline[None],
                                                  n_samples=25).detach().numpy()
            benchmark_explanations["Shapley"] = R_shapley
            continue

        elif explanation_name == "CovShapley":
            baseline = X_flipping.mean(0) * 0
            list_of_shapley_exps = []
            n_outputs = ensemble.models[0].forward(X_flipping).shape[1]
            for model in ensemble.models:
                list_of_output_dim_exps = []
                for d in range(n_outputs):
                    # create a small wrapper around model.forward to return only the d-th output
                    def forward_with_output_dimension(x):
                        return model.forward(x)[:, d]
                    shapley_sampler = captum.attr.ShapleyValueSampling(forward_with_output_dimension)
                    R_shapley = shapley_sampler.attribute(inputs=X_flipping, baselines=baseline[None],
                                                          n_samples=25).detach().numpy()
                    list_of_output_dim_exps.append(R_shapley)
                list_of_shapley_exps.append(np.stack(list_of_output_dim_exps))
            # transpose since covariance explanation expects M x N x D_out x D_in
            R_shapley_all = np.stack(list_of_shapley_exps).swapaxes(1, 2)
            R_shapley_all = R_shapley_all.reshape((*R_shapley_all.shape[:3], -1))
            R_shapley_diag, R_shapley_marg = covariance_explanation(R_shapley_all, input_shape=X_flipping.shape[1:])

            benchmark_explanations["CovShapley_diag"] = R_shapley_diag
            benchmark_explanations["CovShapley_marg"] = R_shapley_marg
            continue

        elif explanation_name == "LIME":
            # as a crude approximation we use the median squared distance rule on the flipping set
            # to choose the kernel width
            distances = euclidean_distances(
                X_flipping.reshape((X_flipping.shape[0], -1)),
                X_flipping.reshape((X_flipping.shape[0], -1)))
            median_distance = np.median(distances)

            # LIME
            lime = captum.attr.Lime(
                forward_func=ensemble.epistemic_variance,
                interpretable_model=SkLearnLinearRegression(),
                similarity_func=get_exp_kernel_similarity_function("euclidean", kernel_width=median_distance),
            )
            # IMPORTANT: Added seed change in source code of LIME, otherwise the exp
            R_lime = lime.attribute(
                inputs=X_flipping, n_samples=100).detach().numpy()
            benchmark_explanations["LIME"] = R_lime
            continue

        elif explanation_name == "CovLIME":
            distances = euclidean_distances(
                X_flipping.reshape((X_flipping.shape[0], -1)),
                X_flipping.reshape((X_flipping.shape[0], -1)))
            median_distance = np.median(distances)
            n_outputs = ensemble.models[0].forward(X_flipping).shape[1]
            list_of_LIME_exps = []
            for model in ensemble.models:
                list_of_output_dim_exps = []
                for d in range(n_outputs):
                    # create a small wrapper around model.forward to return only the d-th output
                    def forward_with_output_dimension(x):
                        return model.forward(x)[:, d]
                    lime = captum.attr.Lime(
                        forward_func=forward_with_output_dimension,
                        interpretable_model=SkLearnLinearRegression(),
                        similarity_func=get_exp_kernel_similarity_function("euclidean", kernel_width=median_distance),
                    )
                    R_lime = lime.attribute(
                        inputs=X_flipping, n_samples=100).detach().numpy()
                    list_of_output_dim_exps.append(R_lime)
                list_of_LIME_exps.append(np.stack(list_of_output_dim_exps))
            # transpose since covariance explanation expects M x N x D_out x D_in
            R_lime_all = np.stack(list_of_LIME_exps).swapaxes(1, 2)
            R_lime_all = R_lime_all.reshape((*R_lime_all.shape[:3], -1))
            R_lime_diag, R_lime_marg = covariance_explanation(R_lime_all, input_shape=X_flipping.shape[1:])

            benchmark_explanations["CovLIME_diag"] = R_lime_diag
            benchmark_explanations["CovLIME_marg"] = R_lime_marg
            continue

        else:
            raise ValueError("Explanation {} is not recognized".format(explanation_name))

    return benchmark_explanations

def train_kde(X_kde):
    # train kde model on X_train and X_val
    # concatenate and shuffle X_train and X_val
    random.seed(0)
    random.shuffle(X_kde)
    # train a new kde_model

    kde_model = KDE()

    kde_model.fit(X_kde)
    return kde_model


def compute_sample_perturbation_curve(obj_fun, sample, R, kde_model, interval=1):
    # for 1d data
    torch.manual_seed(0)
    np.random.seed(0)
    perturbation_list = []
    perturbation_list.append(obj_fun(sample[None]).item())

    # flatten sample, for tabular data this does not change anything
    sample = sample.flatten()
    # flatten R, for tabular data this does not change anything
    R = R.flatten()
    order_list = R.argsort()[::-1]

    mask = torch.ones_like(sample).bool()
    for num, order_ind in enumerate(order_list):
        mask[order_ind] = False
        if ((num+1) % interval == 0):
            cond_samples = kde_model.conditional_sample(sample, mask=mask, n_samples=50).float()
            perturbation_list.append(obj_fun(cond_samples).mean().item())

    return perturbation_list


def compute_perturbation_curve(obj_fun, X, R, kde_model, interval=1):
    perturbation_list = []
    for i in range(len(X)):
        sample = X[i]
        R_sample = R[i]
        perturbation_list.append(compute_sample_perturbation_curve(obj_fun, sample, R_sample, kde_model, interval))
    mean_perturbation_curve = np.mean(perturbation_list, axis=0)
    return mean_perturbation_curve, perturbation_list


def compute_benchmark_flipping_curves(obj_fun, X_flipping, benchmark_explanations, kde_model, interval):

    # dict to save perturbation curves for each benchmark explanation
    mean_perturbation_curves = {}
    all_perturbation_curves = {}
    # iterate over all benchmark explanations
    for key in benchmark_explanations.keys():
        # compute perturbation curve for the current benchmark explanation
        mean_perturbation_curve, flipping_set_perturbation_curves = compute_perturbation_curve(obj_fun=obj_fun, X=X_flipping,
                                                              R=benchmark_explanations[key], kde_model=kde_model,
                                                              interval=interval)
        mean_perturbation_curves[key] = mean_perturbation_curve
        all_perturbation_curves[key] = flipping_set_perturbation_curves

    return mean_perturbation_curves, all_perturbation_curves


def pixelflipping(dataset_name, X_flipping, ensemble, benchmark_explanations, flipping_interval=1):
    dataset = provide_data(dataset_name)
    X_train, X_val, X_test, y_train, y_val, y_test = unpack_dataset(dataset)

    # if X has 3 dimensions flatten the last two, else do nothing
    #X_flipping = X_flipping.view(X_flipping.shape[0], -1)

    X_kde = torch.cat([X_train, X_val])[:4000]
    X_kde = X_kde.view(X_kde.shape[0], -1)
    # get kde_model for perturbation distribution
    kde_model = train_kde(X_kde)

    def ensemble_variance(x):
        # assume that x is flattened and if models in ensemble are CNNs, expand the second dimension
        # to C and L
        first_model = ensemble.models[0]
        if first_model.__class__.__name__ == "CNN":
            C = first_model.n_channels
            L = first_model.n_input_len
            x = x.view(x.shape[0], C, L)

        return ensemble.epistemic_variance(x)

    mean_flipping_curves, all_flipping_curves = compute_benchmark_flipping_curves(
            ensemble_variance, X_flipping, benchmark_explanations,
                                                            kde_model, interval=flipping_interval)
    return mean_flipping_curves, all_flipping_curves


def norm_flipping_curves(flipping_curves):
    normed_flipping_curves = {}
    # norm the flipping curves by their first value
    for key in flipping_curves.keys():
        normed_flipping_curves[key] = flipping_curves[key] / flipping_curves[key][0]
    return normed_flipping_curves


def calculate_aucs(flipping_curves):
    # compute area under curves
    normed_flipping_curves = norm_flipping_curves(flipping_curves)
    auc = {}
    for key in normed_flipping_curves.keys():
        # norm the flipping curves by their first value
        auc[key] = np.trapz(normed_flipping_curves[key], dx=1) / (len(normed_flipping_curves[key])-1)
    return auc

def calculate_auc_stds(flipping_curves):
    mean_dict = {}
    std_dict = {}
    for key in flipping_curves.keys():
        all_flipping_curves = np.array(flipping_curves[key])
        auc_list = []
        for curve in all_flipping_curves:
            normed_curve = curve / curve[0]
            auc = np.trapz(normed_curve, dx=1) / (len(normed_curve) - 1)
            #auc = calculate_aucs(curve)
            auc_list.append(auc)
        auc_list = np.array(auc_list)
        auc_mean = np.mean(auc_list)
        auc_std = np.std(auc_list)
        mean_dict[key] = auc_mean
        std_dict[key] = auc_std
    return mean_dict, std_dict

def pixelflipping_procedure(ensemble, dataset_name, dataset, explanation_names, n_samples = 100):
    # fix random seed
    torch.manual_seed(1)
    np.random.seed(1)

    # we chose a smaller LRP gamma for CNNs as they have 5 instead of 3 layers
    if ensemble.models[0].__class__.__name__ == "CNN":
        gamma = 0.1
    else:
        gamma = 0.2

    # move ensemble to cpu for pixel flipping
    ensemble.move_to_cpu()
    # record the test data points with highest predictive uncertainty
    X_flipping, y_flipping = add_flipping_dataset(dataset, ensemble, n_samples=n_samples)

    # if data is tabular, n_dim is the number of dims in shape axis 1
    if len(X_flipping.shape) == 2:
        n_features = X_flipping.shape[1]
    # else we need to infer it from number and length of channels
    elif len(X_flipping.shape) == 3:
        n_features = X_flipping.shape[1] * X_flipping.shape[2]

    # as pixel flipping is computationally expensive, we set the interval to 1 for low dimensional data
    # but skip evaluations for high dimensional data
    flipping_interval = 1
    if n_features > 30:
        flipping_interval = 5

    # wrapper method which executes all specified uncertainty explanations on the flipping set.
    benchmark_explanations = \
        get_benchmark_explanations(ensemble, X_flipping,
                                   ensemble_type="Ensemble", gamma=gamma, explanation_names=explanation_names)
    # perform actual pixelflipping with the benchmark explanations
    mean_flipping_curves, all_flipping_curves = pixelflipping(dataset_name, X_flipping, ensemble, benchmark_explanations, flipping_interval=flipping_interval)
    aucs, stds = calculate_auc_stds(all_flipping_curves)

    pixelflipping_dict = \
        {"gamma": gamma, "n_samples": n_samples, "n_features": n_features,
         "interval": flipping_interval, "flipping_curves": mean_flipping_curves,
         "auc": aucs, "auc_stds": stds}

    return pixelflipping_dict


if __name__ == "__main__":

    # name the datasets for which the pixelflipping procedure should be run
    # suffix _channel indicates that the dataset is in channelized format for CNNs
    dataset_names = [
        "Bias Correction",
        "California Housing", "YearPredictionMSD", "Seoul Bike",
        "Wine Quality", "EPEX-FR",
        "EPEX-FR_channel", "Seoul Bike_channel"]

    # name uncertainty type either "ensemble" or "MC_dropout"
    uncertainty_type = "MC_dropout"

    # list of compared uncertainty explanations
    explanation_methods = [
        "LRP", "CovLRP", "GI", "CovGI",
        "Shapley", "CovShapley", "Sensitivity", "IG",
        "CovIG", "LIME"]

    # number of models in the ensemble, adjust as needed
    n_models = 3

    for dataset_name in dataset_names:
        # provide the dataset either from storage or by recreating it
        dataset = provide_data(dataset_name)

        # provide ensemble for uncertainty estimation, either by loading it from storage or by training it
        ensemble = provide_ensemble(dataset_name, dataset, n_models, uncertainty_type)

        # prepare and run pixelflipping procedure and collect various results and meta data such as aucs, stds, etc.
        pixelflipping_dict = pixelflipping_procedure(ensemble, dataset_name, dataset, explanation_methods)

        # save results as pickle file to storage
        save_dir = "results/pixelflipping/{}/{}_models".format(uncertainty_type, n_models)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        result_save_path_pkl = save_dir + "/{}.pkl".format(dataset_name)
        with open(result_save_path_pkl, 'wb') as f:
            pickle.dump(pixelflipping_dict, f)

        print("Pixelflipping on {} done".format(dataset_name))