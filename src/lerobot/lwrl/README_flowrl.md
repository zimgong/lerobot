# FlowRL Integration: Exploitation via Weighted Behavior Cloning

* **FlowRL exploitation term** integrated into SAC policy with **minimal code changes**,
* **expectile regression** for buffer-optimal value estimation (Q^π_β*, V^π_β*),
* **weighted BC loss** in actor training for exploitation guidance,
* **configuration knobs** with safe defaults,
* **mathematical foundation** aligned with FlowRL paper equations.

---

# FlowRL: Exploitation via Weighted Behavior Cloning

This integration adds **FlowRL-style exploitation** to your SAC implementation, enabling the policy to leverage high-value actions from the replay buffer while maintaining exploration. The implementation uses **expectile regression** to estimate buffer-optimal values and a **weighted behavior cloning** term to guide the actor toward better actions.

## What FlowRL Adds (One-Screen Summary)

- **Two value heads** (trained with critic):
  - $Q^{\pi_{\beta^*}}(s,a)$: estimates value of buffer-optimal actions
  - $V^{\pi_{\beta^*}}(s)$: estimates state value for buffer-optimal policy
- **Weighted BC loss** in actor training:
  $$\mathbb{E}[\max(Q^{\pi_{\beta^*}}(s,a) - Q^{\pi_\theta}(s,a'), 0) \cdot \|a' - a\|^2]$$
  where the weight $\max(Q^{\pi_{\beta^*}} - Q^{\pi_\theta}, 0)$ guides policy toward higher-value actions
- **Expectile regression** for robust value estimation ($\tau=0.9$ default)
- **Zero changes** to your training loop - all integration is internal to the policy

---

## Mathematical Foundation

### 1.1 SAC Actor Objective (Baseline)

Your standard SAC actor minimizes:
$$\mathcal{L}_{\text{SAC}}(\theta) = \mathbb{E}[\alpha \log \pi_\theta(a'|s) - \min_i Q_i(s,a')]$$

### 1.2 FlowRL Exploitation Constraint

FlowRL adds a **weighted behavior cloning** term that pulls the policy toward high-value buffer actions:

$$\mathcal{L}_{\text{FlowBC}}(\theta) = \mathbb{E}[f(Q^{\pi_{\beta^*}}(s,a) - Q^{\pi_\theta}(s,a')) \cdot \|a' - a\|^2]$$

where:
- $f(x) = \max(x, 0)$ (ReLU) - non-negative weight function
- $a$ is buffer action, $a' \sim \pi_\theta(\cdot|s)$ is policy action
- Weight $f(\cdot)$ is **zero** when policy outperforms buffer, **positive** when buffer is better

### 1.3 Final Actor Loss

$$\mathcal{L}_{\text{actor}}(\theta) = \mathcal{L}_{\text{SAC}}(\theta) - \lambda_{\text{bc}} \cdot \mathcal{L}_{\text{FlowBC}}(\theta)$$

The **negative sign** makes this a **loss** (minimization problem). The BC term pulls the policy toward buffer actions when they have higher value.

### 1.4 Expectile Regression for $Q^{\pi_{\beta^*}}$ and $V^{\pi_{\beta^*}}$

We use **expectile regression** (as in IQL) to estimate buffer-optimal values:

**$V^{\pi_{\beta^*}}$ training:**
$$\mathcal{L}_V = \mathbb{E}[L_2^\tau(Q^{\pi_{\beta^*}}(s,a) - V^{\pi_{\beta^*}}(s))]$$

**$Q^{\pi_{\beta^*}}$ training:**
$$\mathcal{L}_{Q^*} = \mathbb{E}[(r + \gamma V^{\pi_{\beta^*}}(s') - Q^{\pi_{\beta^*}}(s,a))^2]$$

where $L_2^\tau(u) = |\tau - \mathbf{1}(u<0)| \cdot u^2$ is the **asymmetric L2 loss** with expectile parameter $\tau \in (0,1)$.

**Stop-gradients:**
- In $\mathcal{L}_V$: $Q^{\pi_{\beta^*}}(s,a)$ is **detached**
- In $\mathcal{L}_{Q^*}$: target $r + \gamma V^{\pi_{\beta^*}}(s')$ is **detached**

---

## Implementation Details

### 2.1 Architecture Changes

**New components** (only created if `flow_rl_enabled=True`):
- `self.q_beta_star: CriticHead` → $Q^{\pi_{\beta^*}}(s,a)$ 
- `self.v_beta_star: CriticHead` → $V^{\pi_{\beta^*}}(s)$
- `Policy.calc_bc_loss()` → per-sample MSE helper

**Optimizer integration:**
- $Q^{\pi_{\beta^*}}$ and $V^{\pi_{\beta^*}}$ parameters are added to the **critic** optimizer group
- No changes needed to your training loop

### 2.2 Critic Training (`compute_loss_critic`)

**Standard SAC:** unchanged TD loss on critic ensemble

**Added FlowRL losses:**
```python
# Expectile regression for V^π_β*
v_residual = (q_star_sa.detach() - v_star_s)  # stop-grad on Q*
v_loss = expectile_loss(v_residual, tau).mean()

# TD learning for Q^π_β*  
q_target = rewards + (1-done) * γ * v_star_s_next.detach()  # stop-grad on V*
q_loss_star = F.mse_loss(q_star_sa, q_target)

# Combined auxiliary loss
critics_loss = original_critic_loss + flow_rl_qv_weight * (v_loss + q_loss_star)
```

### 2.3 Actor Training (`compute_loss_actor`)

**Standard SAC:** unchanged actor loss

**Added weighted BC:**
```python
# Compute BC loss per sample
bc_per_sample = actor.calc_bc_loss(actions_pi, actions_buffer)  # [B]

# Weight: ReLU(Q^π_β*(s,a_buf) - Q^π_θ(s,a_pi))
with torch.no_grad():
    q_star_sa = q_beta_star_forward(obs, actions_buffer, obs_features)  # [B]
q_pi_sa = min_q_preds.detach()  # [B] - ensemble min
weights = torch.relu(q_star_sa - q_pi_sa)  # [B]

# Weighted BC term
weighted_bc = (weights * bc_per_sample).mean()
actor_loss = original_actor_loss + λ_bc * weighted_bc
```

### 2.4 Discrete Action Handling

For **hybrid action spaces** (continuous + discrete):
- Q^π_β* and BC only use **continuous portion**: `actions[:, :DISCRETE_DIMENSION_INDEX]`
- Discrete critic remains unchanged

---

## Configuration & Usage

### 3.1 Configuration Parameters

Add to your `SACConfig`:

```python
# FlowRL parameters
flow_rl_enabled: bool = False                    # Enable FlowRL exploitation
flowrl_expectile_tau: float = 0.9               # Expectile $\tau \in (0,1)$ for $V^{\pi_{\beta^*}}$
flow_rl_bc_weight: float = 0.1                  # $\lambda_{\text{bc}}$ weight for weighted BC
flow_rl_qv_weight: float = 1.0                  # Weight for auxiliary losses
```

**Safe defaults:** All parameters have safe defaults; missing configs won't crash.

### 3.2 Example Configuration

```yaml
# Enable FlowRL
flow_rl_enabled: true
flowrl_expectile_tau: 0.9
flow_rl_bc_weight: 0.1
flow_rl_qv_weight: 1.0

# Your existing SAC config...
policy:
  type: "sac_flowrl"  # Use FlowRL policy
  # ... other SAC parameters
```

### 3.3 Training Usage

**No changes to your training script!** Once enabled:

- **Critic updates:** train standard critics + Q^π_β*, V^π_β*
- **Actor updates:** minimize SAC loss + weighted BC
- **Inference:** `select_action()` works unchanged
- **Evaluation:** standard policy evaluation

---

## Practical Recommendations

### 4.1 Hyperparameter Tuning

**Expectile $\tau$:**
- $\tau \in [0.7, 0.97]$ are common values
- Larger $\tau$ → $V^{\pi_{\beta^*}}$ approximates **upper expectile** (more optimistic)
- $\tau = 0.9$ is standard default from IQL

**BC weight $\lambda_{\text{bc}}$:**
- Start with `0.1`
- **Too high:** policy collapses to pure imitation
- **Too low:** no exploitation benefit
- If weights explode, try **clipping**: `torch.clamp(weights, max=10.0)`

**Auxiliary loss weight:**
- `flow_rl_qv_weight = 1.0` is usually fine
- Lower if auxiliary losses dominate critic training

### 4.2 Training Tips

**Warm-starting:**
- Train for ~1K steps with $\lambda_{\text{bc}} = 0$ to let critics stabilize
- Then gradually increase $\lambda_{\text{bc}}$

**Weight monitoring:**
- Track `mean(ReLU($Q^{\pi_{\beta^*}} - Q^{\pi_\theta}$))` - should be > 0 when exploitation is active
- Monitor `mean(||a' - a||²)` - BC loss magnitude

**Numerical stability:**
- Weights are **detached** to prevent gradient flow through Q-heads to actor
- Expectile $\tau$ is **clamped** to $(0,1)$ with warning if invalid

### 4.3 Troubleshooting

**"Weights are always zero":**
- Check if $Q^{\pi_{\beta^*}} > Q^{\pi_\theta}$ (policy already better than buffer)
- Verify critics are trained enough
- Try larger $\tau$ (more optimistic $V^{\pi_{\beta^*}}$)

**"Policy collapses to imitation":**
- Reduce $\lambda_{\text{bc}}$ (e.g., 0.01)
- Check if BC loss dominates SAC loss

**"Training is unstable":**
- Clip weights: `torch.clamp(weights, max=10.0)`
- Use softplus instead of ReLU: `torch.log(1 + torch.exp(x))`

---

## API Reference

### 4.1 Policy Class

```python
class SACFlowRLPolicy(SACPolicy):
    name = "sac_flowrl"
    
    # FlowRL components (created if flow_rl_enabled=True)
    q_beta_star: CriticHead      # $Q^{\pi_{\beta^*}}(s,a)$
    v_beta_star: CriticHead      # $V^{\pi_{\beta^*}}(s)$
```

### 4.2 Key Methods

```python
# FlowRL value estimation
q_star_sa = policy._q_beta_star_forward(obs, actions, obs_features)  # [B]
v_star_s = policy._v_beta_star_forward(obs, obs_features)          # [B]

# BC loss computation  
bc_loss = policy.actor.calc_bc_loss(pred_actions, target_actions)  # [B]

# Expectile loss
expectile_loss = policy._expectile_loss(residual, tau)              # [B]
```

### 4.3 Configuration Registration

The policy is registered as:
```python
@PreTrainedConfig.register_subclass("sac")
@PreTrainedConfig.register_subclass("sac_flowrl")  # Same config, different name
class SACConfig(PreTrainedConfig):
    # ... FlowRL parameters
```

---

## Mathematical References

### 5.1 FlowRL Paper Equations

- **Weighted constraint & Lagrangian**: Sec. 4.3-4.4, Eqs. (13)-(22) in *Flow-Based Policy for Online Reinforcement Learning*
- **Expectile estimators**: Eqs. (18)-(19) for $V^{\pi_{\beta^*}}$ and $Q^{\pi_{\beta^*}}$ training
- **Non-negative weight function**: $f(x) = \max(x,0)$ for exploitation guidance

### 5.2 IQL Expectile Regression

- **$L_2^\tau$ loss**: $L_2^\tau(u) = |\tau - \mathbf{1}(u<0)| \cdot u^2$ (asymmetric L2)
- **Expectile interpretation**: $\tau$ controls optimism (higher $\tau$ → more optimistic $V^{\pi_{\beta^*}}$)
- **Standard values**: $\tau \in [0.7, 0.97]$, with $\tau = 0.9$ being common

### 5.3 Implementation Notes

**Stop-gradient conventions:**
- $Q^{\pi_{\beta^*}}$ **detached** in $V^{\pi_{\beta^*}}$ training (prevents circular gradients)
- $V^{\pi_{\beta^*}}$ **detached** in $Q^{\pi_{\beta^*}}$ targets (stable TD learning)
- Weight terms **detached** in actor loss (prevents Q-head gradients to actor)

**Shape consistency:**
- All losses are **mean-reduced** over batch dimension
- BC loss is **per-sample** then **weighted-averaged**
- Expectile loss is **element-wise** then **mean-reduced**

---

## What We Didn't Implement (By Design)

- **No flow-matching velocity residuals**: We use MSE BC proxy $\|a' - a\|^2$ instead of flow-matching $\|v_\theta - (a-a_0)\|^2$
- **No density/OT computations**: Avoids Wasserstein-2 solvers for practical implementation
- **No flow model policy**: Keeps MLP policy with learnable noise (SAC-style)

The `calc_bc_loss()` method is designed to be **swappable** - you can later replace MSE with flow-matching residuals if needed.

---

## References

- **FlowRL paper**: *Flow-Based Policy for Online Reinforcement Learning*, Lv et al., 2025. ([ar5iv](https://ar5iv.org/pdf/2506.12811))
- **IQL paper**: *Offline Reinforcement Learning with Implicit Q-Learning*, Kostrikov et al., 2021. ([arXiv](https://arxiv.org/abs/2110.06169))
- **FlowRL repository**: [bytedance/FlowRL](https://github.com/bytedance/FlowRL)
- **Expectile regression**: *Quantile Regression*, Koenker & Bassett, 1978.

---

## Quick Start Checklist

1. **Enable FlowRL**: Set `flow_rl_enabled: true` in config
2. **Set policy type**: Use `policy.type: "sac_flowrl"`
3. **Tune hyperparameters**: Start with defaults, adjust $\lambda_{\text{bc}}$ based on performance
4. **Monitor weights**: Track `mean(ReLU($Q^{\pi_{\beta^*}} - Q^{\pi_\theta}$))` for exploitation activity
5. **Check stability**: Ensure weights don't explode, clip if needed

The integration is **backward-compatible** - existing SAC configs work unchanged when `flow_rl_enabled: false`.
