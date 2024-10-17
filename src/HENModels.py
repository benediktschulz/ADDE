import copy
import logging
import time
from typing import Any

import edward2 as ed
import keras.backend as K
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import tensorflow_probability as tfp
from concretedropout.tensorflow import (
    ConcreteDenseDropout,
    get_dropout_regularizer,
    get_weight_regularizer,
)
from nptyping import Float, NDArray
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.layers import Concatenate, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from BaseModel import BaseModel
from fn_eval import fn_scores_hd

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class HENBaseModel(BaseModel):
    """
    Represents the base class for Histogram Estimation Networks.
    For a documentation of the methods look at BaseModel
    """

    def __init__(
        self,
        n_ens: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        bin_edges: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_ens,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            **kwargs,
        )
        self.hpar = {
            "N_bins": 20,
            "layers": [64, 32],
            "lr_adam": 5e-4,  # -1 for Adam-default
            "n_epochs": 500,
            "n_patience": 10,
            "n_batch": 64,
            "actv": "softplus",
            "nn_verbose": 0,
            "run_eagerly": False,
        }
        if "hpars" in kwargs:
            for key, value in kwargs["hpars"].items():
                self.hpar[key] = value
        # If not given use quantiles of standard normal
        if bin_edges is None:
            # Equidistant bins vs. more bins in tails
            if self.hpar["N_bins"] == -1:
                self.hpar["N_bins"] = 20
                self.bin_edges = np.r_[1e-16, np.linspace(0, 1, num = self.hpar["N_bins"] + 1)[1:-1], 1-1e-16]
            else:
                q_outer = np.r_[1e-16, 1e-8, 1e-4, 1e-2, 0.05]
                self.bin_edges = np.r_[q_outer, 
                                       np.linspace(0, 1, num = self.hpar["N_bins"] + 1 - 2*len(q_outer))[1:-1], 
                                       1-q_outer]
            
            self.bin_edges = np.unique(np.round(ss.norm.ppf(self.bin_edges), decimals=2))
        else:
            self.bin_edges = bin_edges
        self.hpar["N_bins"] = len(self.bin_edges) - 1
        

    def _build(self, n_samples: int, n_features: int) -> Model:
        # Custom optizimer
        if self.hpar["lr_adam"] == -1:
            custom_opt = "adam"
        else:
            custom_opt = Adam(learning_rate=self.hpar["lr_adam"])

        ### Build network ###
        model = self._get_architecture(
            n_samples=n_samples, n_features=n_features
        )

        # Get custom loss
        custom_loss = self._get_loss()

        ### Estimation ###
        # Compile model
        model.compile(
            optimizer=custom_opt,
            loss=custom_loss,
            run_eagerly=self.hpar["run_eagerly"],
        )

        # Return model
        return model

    def _get_loss(self):
        return "categorical_crossentropy"

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_valid: NDArray,
        y_valid: NDArray,
    ) -> None:
        ### Data preparation ###
        # Read out set sizes
        self.n_train = X_train.shape[0]
        self.n_valid = X_valid.shape[0]

        # Save center and scale parameters
        self.tr_center = np.mean(X_train, axis=0)
        self.tr_scale = np.std(X_train, axis=0)
        self.tr_scale[self.tr_scale == 0.0] = 1.0

        # Scale training data
        X_train = (X_train - self.tr_center) / self.tr_scale

        # Scale validation data with training data attributes
        X_valid = (X_valid - self.tr_center) / self.tr_scale

        # Save center and scale parameters of observations
        self.y_center = np.mean(y_train)
        self.y_scale = np.std(y_train)

        # Scale training and validation observations
        y_train = (y_train - self.y_center) / self.y_scale
        y_valid = (y_valid - self.y_center) / self.y_scale

        # Adapt bin edges if cases are outside of the range
        if min(y_train.min(), y_valid.min()) < self.bin_edges[0]:
            self.bin_edges[0] = min(y_train.min(), y_valid.min()) - 1e-4
        if max(y_train.max(), y_valid.max()) > self.bin_edges[-1]:
            self.bin_edges[-1] = max(y_train.max(), y_valid.max()) + 1e-4
        
        # Generate bin edges of forecast
        self.bin_edges_f = self.y_center + self.bin_edges*self.y_scale
 
        # Calculate bin of each observation
        y_train = np.searchsorted(self.bin_edges[1:], y_train, side="right")
        y_valid = np.searchsorted(self.bin_edges[1:], y_valid, side="right")

        # Transform observations to categorical variables for keras
        y_train = tf.keras.utils.to_categorical(y_train, self.hpar["N_bins"])
        y_valid = tf.keras.utils.to_categorical(y_valid, self.hpar["N_bins"])

        ### Build model ###
        self.model = self._build(
            n_samples=X_train.shape[0], n_features=X_train.shape[1]
        )

        # Take time
        start_tm = time.time_ns()

        # Fit model
        self.model.fit(
            x=X_train,
            y=y_train,
            epochs=self.hpar["n_epochs"],
            batch_size=self.hpar["n_batch"],
            validation_data=(X_valid, y_valid),
            verbose=self.hpar["nn_verbose"],
            callbacks=EarlyStopping(
                patience=self.hpar["n_patience"],
                restore_best_weights=True,
                monitor="val_loss",
            ),
        )

        # Take time
        end_tm = time.time_ns()

        # Time needed
        self.runtime_est = end_tm - start_tm

        self._store_params()

    def predict(self, X_test: NDArray) -> None:
        # Scale data for prediction
        self.n_test = X_test.shape[0]
        X_pred = (X_test - self.tr_center) / self.tr_scale

        ### Prediciton ###
        # Take time
        start_tm = time.time_ns()

        # Predict bin probabilities
        n_mean_prediction = self.hpar.get("n_mean_prediction")
        if n_mean_prediction is None:
            self.p_bins = self.model.predict(
                X_pred, verbose=self.hpar["nn_verbose"]
            )
        else:
            # Predict n_mean_prediction times if single model
            mc_pred: NDArray[Any, Float] = np.array(
                [
                    self.model.predict(
                        X_pred, batch_size=500, verbose=self.hpar["nn_verbose"]
                    )
                    for _ in range(n_mean_prediction)
                ]
            )
            self.p_bins = np.mean(mc_pred, axis=0)

        # Take time
        end_tm = time.time_ns()

        # Time needed
        self.runtime_pred = end_tm - start_tm

    def get_results(self, y_test: NDArray) -> dict[str, Any]:
        ### Evaluation ###
        # Calculate evaluation measures of HEN forecasts
        scores = fn_scores_hd(
            f=self.p_bins,
            y=y_test,
            bin_edges=self.bin_edges_f,
        )

        ### Output ###
        # Output
        return {
            "f": self.p_bins,
            "bin_edges": self.bin_edges_f,
            "nn_ls": self.hpar,
            "scores": scores,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_test": self.n_test,
            "runtime_est": self.runtime_est,
            "runtime_pred": self.runtime_pred,
        }

    def _store_params(self):
        pass


class HENRandInitModel(HENBaseModel):
    """
    Class represents the naive ensemble method for HENs.
    """

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=n_features, name="input", dtype=self.dtype)

        # Hidden layers
        for idx, n_layers in enumerate(self.hpar["layers"]):
            # Build layers
            if idx == 0:
                hidden_layer = Dense(
                    units=n_layers,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input)
            else:
                hidden_layer = Dense(
                    units=n_layers,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer  # type: ignore
                )

        # Softmax as output activation
        output = Dense(
            units=self.hpar["N_bins"],
            activation="softmax",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model


class HENDropoutModel(HENBaseModel):
    """
    Class represents the MC dropout method for HENs.
    """

    def __init__(
        self,
        n_ens: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        bin_edges: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_ens,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            bin_edges,
            **kwargs,
        )
        self.hpar.update(
            {
                "training": True,
                "p_dropout": 0.05,
                "p_dropout_input": 0,
                "upscale_units": True,
                "n_mean_prediction": kwargs.get("n_mean_prediction"),
            }
        )
        if "hpars" in kwargs:
            for key in ["p_dropout", "p_dropout_input"]:
                if key in kwargs.get("hpars"):
                    self.hpar[key] = kwargs.get("hpars")[key]

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        # Extract params
        p_dropout = self.hpar["p_dropout"]

        ### Build network ###
        # Input
        input = Input(shape=n_features, name="input", dtype=self.dtype)
        # Input dropout
        input_d = Dropout(
            rate=self.hpar["p_dropout_input"], noise_shape=(n_features,)
        )(input, training=self.hpar["training"])

        # Hidden layers
        for idx, n_layers in enumerate(self.hpar["layers"]):
            # Calculate units
            if self.hpar["upscale_units"]:
                n_units = int(n_layers / (1 - p_dropout))
            else:
                n_units = n_layers
            # Build layers
            if idx == 0:
                hidden_layer = Dense(
                    units=n_units,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input_d)
                hidden_layer_d = Dropout(
                    rate=p_dropout, noise_shape=(n_units,)
                )(hidden_layer, training=self.hpar["training"])
            else:
                hidden_layer = Dense(
                    units=n_units,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer_d  # type: ignore
                )
                hidden_layer_d = Dropout(
                    rate=p_dropout, noise_shape=(n_units,)
                )(hidden_layer, training=self.hpar["training"])

        # Softmax as output activation
        output = Dense(
            units=self.hpar["N_bins"],
            activation="softmax",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model


class HENBayesianModel(HENBaseModel):
    """
    Class represents the Bayesian NN method for HENs.
    """

    def __init__(
        self,
        n_ens: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        bin_edges: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_ens,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            bin_edges,
            **kwargs,
        )
        self.hpar.update(
            {
                "prior": "standard_normal",
                "posterior": "mean_field",
                "post_scale_scaling": 0.001,
                "n_mean_prediction": kwargs.get("n_mean_prediction"),
            }
        )
        if "hpars" in kwargs:
            for key in ["prior", "posterior", "post_scale_scaling"]:
                if key in kwargs.get("hpars"):
                    self.hpar[key] = kwargs.get("hpars")[key]

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=(n_features,), name="input", dtype=self.dtype)

        # Get prior and posterior
        prior_fn = self._get_prior()
        posterior_fn = self._get_posterior()

        # Hidden layers
        for idx, n_layers in enumerate(self.hpar["layers"]):
            # Build layers
            if idx == 0:
                hidden_layer = tfp.layers.DenseVariational(
                    units=n_layers,
                    make_prior_fn=prior_fn,
                    make_posterior_fn=posterior_fn,
                    kl_weight=1 / n_samples,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input)
            else:
                hidden_layer = tfp.layers.DenseVariational(
                    units=n_layers,
                    make_prior_fn=prior_fn,
                    make_posterior_fn=posterior_fn,
                    kl_weight=1 / n_samples,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer  # type: ignore
                )

        # Softmax as output activation
        output = tfp.layers.DenseVariational(
            units=self.hpar["N_bins"],
            make_prior_fn=prior_fn,
            make_posterior_fn=posterior_fn,
            kl_weight=1 / n_samples,
            activation="softmax",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model

    def _get_prior(self):
        """Returns the prior weight distribution

        e.g. Normal of mean=0 and sd=1
        Prior might be trainable or not

        Returns
        -------
        function
            prior function
        """

        def prior_standard_normal(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = Sequential(
                [
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.MultivariateNormalDiag(
                            loc=tf.zeros(n), scale_diag=tf.ones(n)
                        )
                    )
                ]
            )
            return prior_model

        def prior_uniform(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = Sequential(
                [
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.Independent(
                            tfp.distributions.Uniform(
                                low=tf.ones(n) * -3, high=tf.ones(n) * 3
                            ),
                            reinterpreted_batch_ndims=1,
                        )
                    )
                ]
            )
            return prior_model

        def prior_laplace(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = Sequential(
                [
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.Independent(
                            tfp.distributions.Laplace(
                                loc=tf.zeros(n), scale=tf.ones(n)
                            ),
                            reinterpreted_batch_ndims=1,
                        )
                    )
                ]
            )
            return prior_model

        available_priors = {
            "standard_normal": prior_standard_normal,
            "uniform": prior_uniform,
            "laplace": prior_laplace,
        }

        return available_priors.get(self.hpar["prior"])

    def _get_posterior(self):
        """Returns the posterior weight distribution

        e.g. multivariate Gaussian
        Depending on the distribution the learnable parameters vary

        Returns
        -------
        function
            posterior function
        """

        def posterior_mean_field(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.0))
            posterior_model = Sequential(
                [
                    tfp.layers.VariableLayer(2 * n),
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.Independent(
                            tfp.distributions.Normal(
                                loc=t[..., :n],
                                scale=1e-5
                                + self.hpar["post_scale_scaling"]
                                * tf.nn.softplus(c + t[..., n:]),
                            ),
                            reinterpreted_batch_ndims=1,
                        )
                    ),
                ]
            )
            return posterior_model

        return posterior_mean_field


class HENVariationalDropoutModel(HENBaseModel):
    """
    Class represents the variational dropout method for HENs.
    """

    def __init__(
        self,
        n_ens: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        bin_edges: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_ens,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            bin_edges,
            **kwargs,
        )
        self.hpar.update(
            {
                "n_mean_prediction": kwargs.get("n_mean_prediction"),
            }
        )

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=(n_features,), name="input", dtype=self.dtype)

        # Hidden layers
        for idx, n_layers in enumerate(self.hpar["layers"]):
            # Build layers
            if idx == 0:
                hidden_layer = ed.layers.DenseVariationalDropout(
                    units=n_layers,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input)
            else:
                hidden_layer = ed.layers.DenseVariationalDropout(
                    units=n_layers,
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer  # type: ignore
                )

        # Softmax as output activation
        output = Dense(
            units=self.hpar["N_bins"],
            activation="softmax",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model


class HENConcreteDropoutModel(HENBaseModel):
    """
    Class represents the concrete dropout method for HENs.
    """

    def __init__(
        self,
        n_ens: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        bin_edges: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_ens,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            bin_edges,
            **kwargs,
        )
        self.hpar.update(
            {
                "tau": 1.0,
                "n_mean_prediction": kwargs.get("n_mean_prediction"),
            }
        )

    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=(n_features,), name="input", dtype=self.dtype)

        wr = get_weight_regularizer(N=n_samples, l=1e-2, tau=self.hpar["tau"])
        dr = get_dropout_regularizer(
            N=n_samples, tau=self.hpar["tau"], cross_entropy_loss=False
        )
        # Hidden layers
        for idx, n_layers in enumerate(self.hpar["layers"]):
            # Build layers
            if idx == 0:
                hidden_layer = ConcreteDenseDropout(
                    Dense(
                        units=n_layers,
                        activation=self.hpar["actv"],
                        dtype=self.dtype,
                    ),
                    weight_regularizer=wr,
                    dropout_regularizer=dr,
                    is_mc_dropout=True,
                )(input)
            else:
                hidden_layer = ConcreteDenseDropout(
                    Dense(
                        units=n_layers,
                        activation=self.hpar["actv"],
                        dtype=self.dtype,
                    ),
                    weight_regularizer=wr,
                    dropout_regularizer=dr,
                    is_mc_dropout=True,
                )(
                    hidden_layer  # type: ignore
                )

        # Softmax as output activation
        output = Dense(
            units=self.hpar["N_bins"],
            activation="softmax",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model

    def _store_params(self):
        if self.p_dropout is None:
            self.p_dropout = []
        for layer in self.model.layers:
            if isinstance(layer, ConcreteDenseDropout):
                self.p_dropout.append(
                    tf.nn.sigmoid(layer.trainable_variables[0]).numpy()[0]
                )
        with open(
            f"concrete_dropout_rates_{self.dataset}_{self.ens_method}.txt", "a"
        ) as myfile:
            myfile.write(
                "HEN - Dropout_Rates: " + repr(self.p_dropout) + " \n"
            )
        log_message = f"Learned Dropout rates: {self.p_dropout}"
        logging.info(log_message)


class HENBatchEnsembleModel(HENBaseModel):
    """
    Class represents the BatchEnsemble method for HENs.
    """

    def __init__(
        self,
        n_ens: int,
        dataset: str,
        ens_method: str,
        rpy_elements: dict[str, Any],
        dtype: str = "float32",
        bin_edges: NDArray[Any, Float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_ens,
            dataset,
            ens_method,
            rpy_elements,
            dtype,
            bin_edges,
            **kwargs,
        )
        
    def _get_architecture(self, n_samples: int, n_features: int) -> Model:
        tf.keras.backend.set_floatx(self.dtype)

        ### Build network ###
        # Input
        input = Input(shape=(n_features,), name="input", dtype=self.dtype)

        # Make initializer
        def make_initializer(num):
            return ed.initializers.RandomSign(num)

        # Hidden layers
        for idx, n_layers in enumerate(self.hpar["layers"]):
            # Build layers
            if idx == 0:
                hidden_layer = ed.layers.DenseBatchEnsemble(
                    units=n_layers,
                    rank=1,
                    ensemble_size=self.n_ens,
                    use_bias=True,
                    alpha_initializer=make_initializer(0.5),  # type: ignore
                    gamma_initializer=make_initializer(0.5),  # type: ignore
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(input)
            else:
                hidden_layer = ed.layers.DenseBatchEnsemble(
                    units=n_layers,
                    rank=1,
                    ensemble_size=self.n_ens,
                    use_bias=True,
                    alpha_initializer=make_initializer(0.5),  # type: ignore
                    gamma_initializer=make_initializer(0.5),  # type: ignore
                    activation=self.hpar["actv"],
                    dtype=self.dtype,
                )(
                    hidden_layer  # type: ignore
                )

        # Softmax as output activation
        output = ed.layers.DenseBatchEnsemble(
            units=self.hpar["N_bins"],
            rank=1,
            ensemble_size=self.n_ens,
            use_bias=True,
            alpha_initializer=make_initializer(0.5),  # type: ignore
            gamma_initializer=make_initializer(0.5),  # type: ignore
            activation="softmax",
            dtype=self.dtype,
        )(
            hidden_layer  # type: ignore
        )

        # Define model
        model = Model(inputs=input, outputs=output)

        # Return model
        return model

    def predict(self, X_test: NDArray) -> None:
        # Scale data for prediction
        self.n_test = X_test.shape[0]
        X_pred = (X_test - self.tr_center) / self.tr_scale

        ### Prediciton ###
        # Take time
        start_tm = time.time_ns()

        # Extract all trained weights
        weights = self.model.get_weights()
        # Copy weights to adjust them for the ensemble members
        new_weights = copy.deepcopy(weights)
        # Create new model with same architecture but n_ens=1
        new_model = self._build_single_model(
            n_samples=X_pred.shape[0], n_features=X_pred.shape[1]
        )

        # Initialize predictions
        self.predictions = []

        # Iterate and extract each ensemble member
        for i_ens in range(self.n_ens):
            # Iterate over layers and extract new model's weights
            for i_layer_weights, layer_weights in enumerate(weights):
                # Keep shared weights
                if (i_layer_weights % 4) == 0:
                    new_weights[i_layer_weights] = layer_weights
                # Extract alpha, gammas and bias
                elif (i_layer_weights % 4) != 0:
                    new_weights[i_layer_weights] = np.reshape(
                        layer_weights[i_ens],
                        newshape=(1, layer_weights.shape[1]),
                    )

            # Set new weights
            new_model.set_weights(new_weights)

            # Make predictions with temporary models
            # In order to match dimensions in DenseBatchEnsemble use batchsize
            # from training
            self.predictions.append(
                new_model.predict(
                    X_pred,
                    verbose=self.hpar["nn_verbose"],
                    batch_size=self.hpar["n_batch"],
                )
            )

        # Take time
        end_tm = time.time_ns()

        # Time needed
        self.runtime_pred = end_tm - start_tm

    def _build_single_model(self, n_samples, n_features):
        # Save originial n_ens
        n_ens_original = self.n_ens

        # Use n_ens = 1 for temporary model
        self.n_ens = 1
        model = self._build(n_samples=n_samples, n_features=n_features)

        # Reset n_ens to original value
        self.n_ens = n_ens_original

        return model

    def get_results(self, y_test: NDArray) -> list[dict[str, Any]]:
        # Initialize results
        results = []

        ### Evaluation ###
        for p_bins in self.predictions:
            # Calculate evaluation measures of HEN forecasts
            scores = fn_scores_hd(
                f=p_bins,
                y=y_test,
                bin_edges=self.bin_edges_f,
            )

            results.append(
                {
                    "f": p_bins,
                    "bin_edges": self.bin_edges_f,
                    "nn_ls": self.hpar,
                    "scores": scores,
                    "n_train": self.n_train,
                    "n_valid": self.n_valid,
                    "n_test": self.n_test,
                    "runtime_est": self.runtime_est,
                    "runtime_pred": self.runtime_pred,
                }
            )

        ### Output ###
        # Output
        return results
