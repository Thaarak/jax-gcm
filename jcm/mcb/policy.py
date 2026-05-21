"""Neural network policy modules for MCB control.

This module provides Flax-based neural networks that map climate state
observations to MCB forcing decisions. These policies are designed for
training via Backpropagation Through Time (BPTT) through differentiable
climate simulations.

Available architectures:
- MCBPolicyMLP: Simple multi-layer perceptron
- MCBPolicyCNN: Convolutional network preserving spatial structure
- MCBPolicyResNet: ResNet-style network for better gradient flow

Example usage:
    from jcm.mcb.policy import MCBPolicyMLP
    import jax.numpy as jnp

    policy = MCBPolicyMLP(output_shape=(96, 48), hidden_dims=(256, 256))
    params = policy.init(jax.random.PRNGKey(0), jnp.zeros(100))
    mcb_forcing = policy.apply(params, state_features)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple


class MCBPolicyMLP(nn.Module):
    """Multi-layer perceptron policy for MCB control.

    Maps flattened climate state features to a spatial MCB forcing field.
    Simple but effective for moderate state dimensions.

    Attributes:
        output_shape: Target grid shape (ix, il) for MCB forcing output.
        hidden_dims: Sequence of hidden layer dimensions.
        max_perturbation: Maximum albedo perturbation (output clipped to [0, max]).
        dropout_rate: Dropout rate for regularization during training.

    """

    output_shape: Tuple[int, int] = (96, 48)
    hidden_dims: Sequence[int] = (256, 256)
    max_perturbation: float = 0.15
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, state_features: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass: state features → MCB forcing field.

        Args:
            state_features: Flattened climate state features (batch, features) or (features,).
            training: Whether in training mode (enables dropout).

        Returns:
            MCB forcing field of shape output_shape, values in [0, max_perturbation].

        """
        # Handle both batched and unbatched inputs
        input_shape = state_features.shape
        is_batched = len(input_shape) > 1
        if not is_batched:
            state_features = state_features[None, :]  # Add batch dim

        x = state_features

        # Hidden layers
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f'hidden_{i}')(x)
            x = nn.LayerNorm(name=f'ln_{i}')(x)
            x = nn.gelu(x)
            if self.dropout_rate > 0 and training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Output layer
        output_dim = self.output_shape[0] * self.output_shape[1]
        x = nn.Dense(output_dim, name='output')(x)

        # Reshape to spatial grid
        x = x.reshape((-1,) + self.output_shape)

        # Constrain to valid range [0, max_perturbation]
        x = self.max_perturbation * nn.sigmoid(x)

        # Remove batch dim if input was unbatched
        if not is_batched:
            x = x[0]

        return x


class MCBPolicyCNN(nn.Module):
    """Convolutional policy network preserving spatial structure.

    Takes gridded climate state as input and outputs MCB forcing on the
    same grid. Better for capturing spatial correlations in climate patterns.

    Attributes:
        features: Sequence of filter counts for each conv layer.
        max_perturbation: Maximum albedo perturbation.
        kernel_size: Convolution kernel size (applies to all layers).

    """

    features: Sequence[int] = (32, 64, 64, 32)
    max_perturbation: float = 0.15
    kernel_size: Tuple[int, int] = (3, 3)

    @nn.compact
    def __call__(self, state_grid: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass: gridded state → MCB forcing field.

        Args:
            state_grid: Climate state grid (ix, il, channels) or (batch, ix, il, channels).
            training: Whether in training mode.

        Returns:
            MCB forcing field matching input spatial dims, values in [0, max_perturbation].

        """
        # Handle batched and unbatched inputs
        input_shape = state_grid.shape
        is_batched = len(input_shape) == 4
        if not is_batched:
            state_grid = state_grid[None, ...]  # Add batch dim

        x = state_grid

        # Convolutional layers with residual connections where possible
        for i, feat in enumerate(self.features):
            # Convolution
            conv_out = nn.Conv(
                features=feat,
                kernel_size=self.kernel_size,
                padding='SAME',
                name=f'conv_{i}'
            )(x)
            conv_out = nn.LayerNorm(name=f'ln_{i}')(conv_out)
            conv_out = nn.gelu(conv_out)

            # Skip connection if dimensions match
            if x.shape[-1] == feat:
                x = x + conv_out
            else:
                x = conv_out

        # Final 1x1 conv to single channel output
        x = nn.Conv(
            features=1,
            kernel_size=(1, 1),
            padding='SAME',
            name='output_conv'
        )(x)

        # Remove channel dimension
        x = x.squeeze(-1)

        # Constrain to valid range
        x = self.max_perturbation * nn.sigmoid(x)

        # Remove batch dim if input was unbatched
        if not is_batched:
            x = x[0]

        return x


class ResidualBlock(nn.Module):
    """Residual block for MCBPolicyResNet.

    Uses pre-activation residual connections for better gradient flow
    through deep networks during BPTT.

    """

    features: int
    kernel_size: Tuple[int, int] = (3, 3)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply residual block."""
        residual = x

        # Pre-activation design
        y = nn.LayerNorm()(x)
        y = nn.gelu(y)
        y = nn.Conv(self.features, self.kernel_size, padding='SAME')(y)

        y = nn.LayerNorm()(y)
        y = nn.gelu(y)
        y = nn.Conv(self.features, self.kernel_size, padding='SAME')(y)

        # Project residual if channel mismatch
        if x.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1), padding='SAME')(x)

        return residual + y


class MCBPolicyResNet(nn.Module):
    """ResNet-style policy for deep gradient flow.

    Uses residual connections to enable training with BPTT through
    long climate simulation rollouts. Better gradient flow than
    plain CNNs for multi-year simulations.

    Attributes:
        features: Filter count for residual blocks.
        num_blocks: Number of residual blocks.
        max_perturbation: Maximum albedo perturbation.

    """

    features: int = 64
    num_blocks: int = 4
    max_perturbation: float = 0.15

    @nn.compact
    def __call__(self, state_grid: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass: gridded state → MCB forcing field.

        Args:
            state_grid: Climate state grid (ix, il, channels) or (batch, ix, il, channels).
            training: Whether in training mode.

        Returns:
            MCB forcing field, values in [0, max_perturbation].

        """
        # Handle batched and unbatched inputs
        input_shape = state_grid.shape
        is_batched = len(input_shape) == 4
        if not is_batched:
            state_grid = state_grid[None, ...]

        # Initial projection
        x = nn.Conv(self.features, (3, 3), padding='SAME', name='stem')(state_grid)

        # Residual blocks
        for i in range(self.num_blocks):
            x = ResidualBlock(self.features, name=f'resblock_{i}')(x)

        # Final layers
        x = nn.LayerNorm(name='final_ln')(x)
        x = nn.gelu(x)
        x = nn.Conv(1, (1, 1), padding='SAME', name='head')(x)
        x = x.squeeze(-1)

        # Constrain output
        x = self.max_perturbation * nn.sigmoid(x)

        if not is_batched:
            x = x[0]

        return x


class MCBPolicyHybrid(nn.Module):
    """Hybrid policy combining global and local processing.

    Uses an MLP to process global scalar features (mean temperatures,
    total precipitation) and a CNN to process spatial patterns, then
    combines them for the final MCB forcing output.

    Attributes:
        global_dims: Hidden dimensions for global feature MLP.
        spatial_features: Filter counts for spatial CNN.
        output_shape: Target grid shape for output.
        max_perturbation: Maximum albedo perturbation.

    """

    global_dims: Sequence[int] = (64, 64)
    spatial_features: Sequence[int] = (32, 32)
    output_shape: Tuple[int, int] = (96, 48)
    max_perturbation: float = 0.15

    @nn.compact
    def __call__(
        self,
        global_features: jnp.ndarray,
        spatial_features: jnp.ndarray,
        training: bool = False
    ) -> jnp.ndarray:
        """Forward pass with both global and spatial inputs.

        Args:
            global_features: Scalar/vector features (batch, features) or (features,).
            spatial_features: Gridded features (batch, ix, il, channels) or (ix, il, channels).
            training: Whether in training mode.

        Returns:
            MCB forcing field of shape output_shape.

        """
        # Handle unbatched inputs
        global_unbatched = len(global_features.shape) == 1
        spatial_unbatched = len(spatial_features.shape) == 3

        if global_unbatched:
            global_features = global_features[None, :]
        if spatial_unbatched:
            spatial_features = spatial_features[None, ...]

        # Process global features through MLP
        g = global_features
        for i, dim in enumerate(self.global_dims):
            g = nn.Dense(dim, name=f'global_{i}')(g)
            g = nn.gelu(g)

        # Expand global features to spatial grid
        # Shape: (batch, 1, 1, global_dim) for broadcasting
        g = g[:, None, None, :]
        g = jnp.broadcast_to(g, (g.shape[0],) + self.output_shape + (g.shape[-1],))

        # Process spatial features through CNN
        s = spatial_features
        for i, feat in enumerate(self.spatial_features):
            s = nn.Conv(feat, (3, 3), padding='SAME', name=f'spatial_{i}')(s)
            s = nn.gelu(s)

        # Resize spatial features to output shape if needed
        if s.shape[1:3] != self.output_shape:
            s = jax.image.resize(
                s,
                shape=(s.shape[0],) + self.output_shape + (s.shape[-1],),
                method='bilinear'
            )

        # Combine global and spatial
        combined = jnp.concatenate([g, s], axis=-1)

        # Final conv layers
        x = nn.Conv(32, (3, 3), padding='SAME', name='combine_0')(combined)
        x = nn.gelu(x)
        x = nn.Conv(1, (1, 1), padding='SAME', name='combine_out')(x)
        x = x.squeeze(-1)

        # Constrain output
        x = self.max_perturbation * nn.sigmoid(x)

        if global_unbatched and spatial_unbatched:
            x = x[0]

        return x


def create_policy(
    policy_type: str,
    output_shape: Tuple[int, int],
    max_perturbation: float = 0.15,
    **kwargs
) -> nn.Module:
    """Create MCB policy networks.

    Args:
        policy_type: One of 'mlp', 'cnn', 'resnet', 'hybrid'.
        output_shape: Target grid shape (ix, il).
        max_perturbation: Maximum albedo perturbation.
        **kwargs: Additional arguments passed to the policy constructor.

    Returns:
        Configured policy network module.

    Raises:
        ValueError: If policy_type is not recognized.

    """
    policies = {
        'mlp': MCBPolicyMLP,
        'cnn': MCBPolicyCNN,
        'resnet': MCBPolicyResNet,
        'hybrid': MCBPolicyHybrid,
    }

    if policy_type not in policies:
        raise ValueError(
            f"Unknown policy type '{policy_type}'. "
            f"Available: {list(policies.keys())}"
        )

    policy_cls = policies[policy_type]

    if policy_type == 'mlp':
        return policy_cls(
            output_shape=output_shape,
            max_perturbation=max_perturbation,
            **kwargs
        )
    elif policy_type in ('cnn', 'resnet'):
        return policy_cls(
            max_perturbation=max_perturbation,
            **kwargs
        )
    elif policy_type == 'hybrid':
        return policy_cls(
            output_shape=output_shape,
            max_perturbation=max_perturbation,
            **kwargs
        )


def init_policy_params(
    policy: nn.Module,
    rng_key: jax.Array,
    input_shape: Tuple[int, ...],
    input_dtype: jnp.dtype = jnp.float32,
) -> dict:
    """Initialize policy network parameters.

    Args:
        policy: Policy network module.
        rng_key: JAX random key for initialization.
        input_shape: Shape of input features (without batch dimension).
        input_dtype: Data type for dummy input.

    Returns:
        Initialized parameter dictionary.

    """
    dummy_input = jnp.zeros(input_shape, dtype=input_dtype)
    params = policy.init(rng_key, dummy_input)
    return params
