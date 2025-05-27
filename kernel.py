import jax
import jax.numpy as jnp
from spectral import get_loss_wrap, lanczos_inverse_hvp
from functools import partial
from jax.scipy.special import betainc
import jax.flatten_util as fu


def compute_kernel_inverse(x, kernel_fn, lambda_=1e-3):
    """
    Compute the inverse of (K + lambda * I), where K is computed via a kernel function.

    Parameters:
        x (ndarray): Input data (n_samples x n_features)
        kernel_fn (callable): A function that computes the kernel matrix, e.g. rbf_kernel
        lambda_ (float): Regularization parameter

    Returns:
        inv_matrix (ndarray): The inverse of (K + lambda * I)
        K (ndarray): The computed kernel matrix
    """
    # Compute kernel matrix K(X, X)
    K = kernel_fn(x, x)

    # Add regularization term lambda * I
    n = K.shape[0]
    A = K + lambda_ * jnp.eye(n)

    # Compute inverse using JAX's version of solve
    I = jnp.eye(n)

    # Run the solve on CPU
    with jax.default_device(jax.devices("cpu")[0]):
        inv_matrix = jnp.linalg.solve(A, I)

    return inv_matrix, K



def compute_squared_hilbert_norm(x, y, kernel_fn, lambda_=1e-3):#(y, lambda_, K): 
    """
    Compute the squared Hilbert norm approximation.

    Parameters:
        x (ndarray): Input data (n_samples x n_features)
        y (ndarray): Target values (n_samples x n_features)
        lambda_ (float): Regularization parameter
        K (ndarray): Precomputed kernel matrix (n_samples x n_samples)

    Returns:
        float: The squared RKHS norm of the fitted function
    """
    # Compute the inverse of (K + lambda * I), with K the kernel matrix
    inv, _ = compute_kernel_inverse(x, kernel_fn, lambda_)

    # Compute the squared norm
    mmd = y.T @ inv @ y

    return mmd



def score_function_gradient_kernel_wrapper(log_prob, kernel_fn, N=100):
    """
    Returns a JIT-compiled function that estimates the score-function gradient
    of the expected kernel value E_y[k(y, y')] using the REINFORCE estimator.

    Args:
        log_prob: Function log_prob(params, y) → scalar log-probability.
        kernel_fn: Function k(y, y') → R^1×1 kernel value, can be partially applied with kwargs.

    Returns:
        A JAX-jitted function that takes:
            - state (with .params attribute),
            - logits (from model output),
            - y_prime (reference label),
            - N (number of samples),
        and returns:
            - A PyTree of kernel-scaled gradient estimates.
    """
    @jax.jit
    def score_function_gradient_kernel(
        state,
        logits,
        y_prime,
        rng_key
    ):
        """
        Internal function to estimate ∇θ E_y[k(y, y')] using REINFORCE.

        Args:
            state: Object containing model parameters in state.params.
            logits: Logits for sampling y ~ p(y | logits).
            y_prime: Fixed label y' against which kernel is evaluated.

        Returns:
            PyTree: Averaged score-function gradients scaled by kernel values.
        """
        # Step 1: Sample N class labels y ~ p(y | logits)
        sample_keys = jax.random.split(rng_key, N)
        ys = jax.vmap(lambda k: jax.random.categorical(k, logits))(sample_keys)

        # Step 2: Preprocess y_prime to have shape (1, 1) for kernel input
        y_prime_reshaped = y_prime.reshape(-1, 1)

        # Step 3: Define a function to compute scaled gradient for one sample y
        def single_grad_scaled(y):
            # Compute ∇_θ log p(y | θ)
            grad_logp = jax.grad(lambda p: log_prob(p, y))(state.params)

            # Reshape y to match kernel input shape (1, 1)
            y_reshaped = jnp.array(y).reshape(-1, 1)

            # Evaluate kernel k(y, y')
            k_val = kernel_fn(y_reshaped, y_prime_reshaped)[0, 0]

            # Scale gradient by kernel value
            return jax.tree_util.tree_map(lambda g: k_val * g, grad_logp)

        # Step 4: Apply the gradient-scaling function over all samples
        all_grads = jax.vmap(single_grad_scaled)(ys)

        # Step 5: Average the scaled gradients across samples
        grad_mean = jax.tree_util.tree_map(
            lambda g: jnp.mean(g, axis=0),
            all_grads
        )

        return grad_mean

    return score_function_gradient_kernel



def compute_mmd_for_sample(x, y, state, model, loss_fn, 
                           tridiag_matrix, lanczos_vecs, 
                           log_prob, kernel, rng_init,
                           N_samples=100, lambda_=1e-3, iota_=1e-3):
    """
    Computes the Maximum Mean Discrepancy (MMD) for a single (x, y) pair using
    a kernel-based estimator with score-function gradients.

    Args:
        x : jnp.ndarray
            Input data (shape: [1, ...] for a single sample).
        y : jnp.ndarray
            Target label (shape: [1, ...]).
        state : TrainState
            Contains model parameters and batch statistics.
        model : nn.Module
            The Flax model being evaluated.
        loss_fn : callable
            Loss function used to compute parameter gradients.
        tridiag_matrix : jnp.ndarray
            Tridiagonal matrix from Lanczos algorithm for H⁻¹ approximation.
        lanczos_vecs : jnp.ndarray
            Lanczos basis vectors for H⁻¹ approximation.
        log_prob : callable
            Function log_prob(params, y) → log p(y | params).
        kernel : callable
            Kernel function k(y, y') returning scalar similarity.
        rng_init : jax.random.PRNGKey
            Initial PRNG key for sampling.
        N_samples : int, optional
            Number of categorical samples to draw (default is 100).
        lambda_ : float, optional
            Regularization for Hilbert norm (default is 1e-3).
        iota_ : float, optional
            Regularization parameter for gradient computation (default is 1e-3).

    Returns:
        mmd : float
            Estimated Maximum Mean Discrepancy value.
    """
    # Wrap loss with current state
    loss_wrap = get_loss_wrap(state, loss_fn, bn=True)

    # Forward pass to get model logits
    logits = model.apply({'params': state.params, 'batch_stats': state.batch_stats}, x, train=False)
    logits = jnp.squeeze(logits)

    # Sample N_samples outputs y ~ p(y | logits)
    _, subkey = jax.random.split(rng_init)
    sample_keys = jax.random.split(subkey, N_samples)
    ys = jax.vmap(lambda k: jax.random.categorical(k, logits))(sample_keys)

    # Compute gradient of loss wrt model parameters
    grad_fn = jax.grad(lambda p: loss_wrap(p, (x, y)))
    grad_tree = grad_fn(state.params)
    grad_flat, _ = fu.ravel_pytree(grad_tree)

    # Estimate H⁻¹ ∇loss using Lanczos approximation
    lanczos_product = lanczos_inverse_hvp(tridiag_matrix, lanczos_vecs, grad_flat)

    # Construct kernel function with fixed args
    kernel_with_args = partial(kernel, gamma=1.0)

    # Get score-function gradient estimator
    score_function_gradient_kernel = score_function_gradient_kernel_wrapper(
        log_prob, kernel_with_args, N_samples)

    # Estimate ∇θ E_y[k(y, y')] for each sampled y'
    batched_grads = jax.vmap(
        lambda y_prime: score_function_gradient_kernel(state, logits, y_prime, rng_init)
    )(ys)

    # Flatten PyTrees to vectors
    flattened_grads = jax.vmap(lambda g: fu.ravel_pytree(g)[0])(batched_grads)

    # Small regularization to ensure stability
    flattened_grads = flattened_grads + iota_ * jnp.ones_like(flattened_grads)

    # Compute inner products: ⟨∇_θ k(y, y'), H⁻¹ ∇_θ loss⟩
    etas = jax.vmap(lambda grad: jnp.dot(grad, lanczos_product))(flattened_grads)

    # Compute final squared RKHS distance (MMD)
    mmd = compute_squared_hilbert_norm(ys, etas, kernel, lambda_)

    return mmd



def compute_mmds(x, y, state, model, loss_fn, 
                          tridiag_matrix, lanczos_vecs, 
                          log_prob, kernel, rng_init,
                          N_samples=100, N_trials=100, lambda_=1e-3, iota_=1e-3):
    """
    Computes the mean MMD value over multiple randomized trials using vectorized evaluation.

    Args:
        x : jnp.ndarray
            Input data (single sample, shape [1, ...]).
        y : jnp.ndarray
            Target label (shape [1, ...]).
        state : TrainState
            Model parameters and statistics.
        model : nn.Module
            The Flax model being evaluated.
        loss_fn : callable
            Loss function used in gradient computations.
        tridiag_matrix : jnp.ndarray
            Tridiagonal matrix for H⁻¹ approximation (from Lanczos).
        lanczos_vecs : jnp.ndarray
            Lanczos basis vectors.
        log_prob : callable
            Function log_prob(params, y) → log p(y | params).
        kernel : callable
            Kernel function k(y, y').
        rng_init : jax.random.PRNGKey
            PRNG key for reproducibility.
        N_samples : int, optional
            Number of samples for each MMD computation (default is 100).
        N_trials : int, optional
            Number of trials for averaging MMD (default is 100).
        lambda_ : float, optional
            Regularization for Hilbert norm (default is 1e-3).
        iota_ : float, optional
            Regularization parameter for gradient computation (default is 1e-3).

    Returns:
        mean_mmd : float
            Average MMD over all trials.
    """
    # Generate random keys for each trial
    trial_keys = jax.random.split(rng_init, N_trials)

    # Define a trial function for vectorized evaluation
    trial_fn = lambda rng: compute_mmd_for_sample(
        x, y, state, model, loss_fn,
        tridiag_matrix, lanczos_vecs,
        log_prob, kernel, rng, N_samples, lambda_, iota_,)

    # Vectorize and compute MMD across trials
    mmds = jax.vmap(trial_fn)(trial_keys)

    # Return MMDs
    return mmds



def t_test(samples):
    '''
    Performs a one-sample t-test using JAX to test whether the sample mean 
    differs significantly from zero.

    Args:
        samples (jnp.ndarray): 1D array of sample data (assumed to be from a single group).

    Returns:
        t_stat (float): The computed t-statistic.
        p_value (float): Two-tailed p-value corresponding to the t-statistic.
    '''
    # Number of samples
    n = samples.shape[0]
    
    # Compute sample mean
    mean = jnp.mean(samples)
    
    # Compute unbiased sample variance (using Bessel's correction: ddof=1)
    var = jnp.var(samples, ddof=1)
    
    # Standard error of the mean
    se = jnp.sqrt(var / n)
    
    # Compute t-statistic: mean divided by its standard error
    t_stat = mean / se

    # Degrees of freedom for the t-distribution
    df = n - 1
    
    # Compute the transformation x for the incomplete beta function
    x = df / (t_stat**2 + df)
    
    # Compute one-tailed p-value using the regularized incomplete beta function
    p_one_tail = 0.5 * betainc(df / 2.0, 0.5, x)
    
    # Convert to two-tailed p-value
    p_value = 2 * p_one_tail
    
    return t_stat, p_value


def evaluate_t_test(t_stat, p_value, mmds, alpha=0.05, eps=1e-8):
    """
    Evaluates a one-sample t-test result.

    Args:
        t_stat (float): t-statistic from the t-test.
        p_value (float): p-value from the t-test.
        mmds (array-like): The sample MMD values.
        alpha (float): Significance level (default: 0.05).
        eps (float): Threshold for treating variance or mean as effectively zero.

    Returns:
        accept_null (bool): 
            True if H₀ is accepted (no significant shift), 
            False if H₀ is rejected (shift detected).
    """
    # Compute standard deviation and mean of MMD samples
    std_mmd = jnp.std(mmds, ddof=1)
    mean_mmd = jnp.mean(mmds)

    # Handle NaN values (e.g., due to division by zero in t-stat or p-value)
    if jnp.isnan(p_value) or jnp.isnan(t_stat):
        return True  # Default to accepting H₀ due to undefined test result

    # Handle near-zero variance case
    if std_mmd < eps:
        if mean_mmd < eps:
            return True   # Mean and variance both near zero → no shift detected
        else:
            return False  # Mean non-zero but variance ≈ 0 → strong evidence of shift

    # Apply standard t-test decision rule
    if p_value < alpha:
        return False  # Reject H₀: statistically significant shift detected
    else:
        return True   # Accept H₀: no significant evidence to claim a shift